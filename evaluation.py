import torch
import torch.nn.functional as F

import model as model_module  # noqa: F401  (needed so torch.load can unpickle the model)

EPS = 1e-8


def _load_model(model_path):
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and isinstance(obj.get("model"), torch.nn.Module):
        model = obj["model"]
    else:
        raise ValueError(f"{model_path} is not a saved nn.Module or {{'model': ...}} checkpoint")
    return model.eval()


def build_Z(N, r_sum, state_mask, r_scale=None):
    """Build the model input Z (B, S, A, 2+S) from exploration statistics.

    N: (B, S, A, S) transition counts, r_sum: (B, S, A) summed rewards,
    state_mask: (B, S) bool. This is the *single* feature builder shared by
    training and evaluation, so the model always sees inputs generated the same
    way. Visited (s,a) get the empirical transition row N/n_sa and mean reward;
    unvisited (s,a) get a uniform row over valid states and zero reward. Rewards
    are normalised by `r_scale` (per-MDP max |mean reward| when None) so the
    feature stays ~[-1, 1] and the model is reward-scale invariant. Returns
    (Z, r_scale).
    """
    n_sa = N.sum(-1)                                          # (B, S, A) visit counts
    visited = n_sa > 0
    valid_s = state_mask.float()                             # (B, S)
    uniform_row = (valid_s / valid_s.sum(-1, keepdim=True).clamp_min(1.0))[:, None, None, :]
    P_hat = torch.where(visited[..., None], N / n_sa[..., None].clamp_min(1), uniform_row)
    P_hat = P_hat * valid_s[:, None, None, :]                # zero invalid successors

    r_mean = r_sum / n_sa.clamp_min(1)                       # mean observed reward per (s,a)
    if r_scale is None:
        r_scale = r_mean.abs().amax(dim=(1, 2), keepdim=True).clamp_min(EPS)
    r_hat = r_mean / r_scale                                  # reward in ~[-1, 1]

    log1p_N = torch.log1p(n_sa)
    Z = torch.cat([log1p_N[..., None], r_hat[..., None], P_hat], dim=-1)
    return Z, r_scale


def _empirical_Z(N, r_sum, r_scale, S):
    """Single-MDP wrapper around `build_Z` for the rollout loop.

    N: (S, A, S) counts, r_sum: (S, A) summed rewards, r_scale: running max
    |reward| seen so far. Returns Z (1, S, A, 2+S).
    """
    state_mask = torch.ones(1, S, dtype=torch.bool)
    Z, _ = build_Z(N[None], r_sum[None], state_mask, r_scale=r_scale)
    return Z


@torch.no_grad()
def rollout_autoregressive(model_path, S, A, P, r, num_episodes, episode_len, temp=None,
                           verbose=True, eval_iters=24, greedy_at=(), probe_reps=1):
    """Roll out the model's policy while it learns the MDP from exploration.

    The model never sees the true (P, r). It starts from a uniform transition
    estimate and zero rewards, and re-plans Q once per episode from the
    empirical statistics gathered so far (hence "autoregressive"); the policy is
    then fixed for that episode while its experience updates the statistics for
    the next one. Statistics persist across episodes. Actions are greedy when
    `temp is None`, else sampled from softmax(Q / temp) -- except on the 1-based
    episode indices in `greedy_at`, which act greedily (strict argmax) so they
    measure exploitation rather than explore. On those greedy episodes the
    reported return is averaged over `probe_reps` rollouts (variance reduction),
    but only the first probe writes to the empirical stats. Returns a
    (num_episodes,) tensor of (true) total reward per episode.
    """
    model = _load_model(model_path)
    P, r = P.float(), r.float()

    N = torch.zeros(S, A, S)        # observed transition counts
    r_sum = torch.zeros(S, A)       # observed reward sums
    r_scale = torch.tensor(EPS)     # running max |reward| for normalisation

    def run_episode(Q, greedy, update_stats):
        nonlocal r_scale
        s, total = 0, 0.0
        for _ in range(episode_len):
            if greedy:
                a = int(Q[s].argmax())
            else:
                a = int(torch.multinomial(F.softmax(Q[s] / temp, dim=-1), 1))
            s_next = int(torch.multinomial(P[s, a], 1))
            reward = r[s, a]
            total += float(reward)
            if update_stats:
                N[s, a, s_next] += 1
                r_sum[s, a] += reward
                r_scale = torch.maximum(r_scale, reward.abs())
            s = s_next
        return total

    returns = torch.zeros(num_episodes)
    for ep in range(num_episodes):
        Q = model(_empirical_Z(N, r_sum, r_scale, S), num_iters=eval_iters)[0]   # (S, A), fixed for the episode
        greedy = temp is None or (ep + 1) in greedy_at   # exploit (argmax) on probe episodes
        reps = probe_reps if greedy else 1               # only greedy probes are averaged
        # first probe records experience; the rest are measurement-only
        probes = [run_episode(Q, greedy, update_stats=(i == 0)) for i in range(reps)]
        returns[ep] = sum(probes) / len(probes)

    if verbose:
        print(f"mean return over {num_episodes} episodes: {returns.mean().item():.3f}")
    return returns


@torch.no_grad()
def optimal_return(P, r, num_episodes, episode_len, gamma=0.95, vi_iters=500):
    """Mean return of the value-iteration-optimal policy, same rollout protocol."""
    P, r = P.float(), r.float()
    S = P.shape[0]
    Q = torch.zeros_like(r)
    for _ in range(vi_iters):
        Q = r + gamma * (P * Q.max(-1).values[None, None, :]).sum(-1)
    policy = Q.argmax(-1)                                 # (S,) greedy action per state

    states = torch.zeros(num_episodes, dtype=torch.long)
    returns = torch.zeros(num_episodes)
    for _ in range(episode_len):
        a = policy[states]
        returns += r[states, a]
        states = torch.multinomial(P[states, a], 1).squeeze(-1)
    return returns.mean().item()


def evaluate_checkpoint(model_path, num_episodes=128, episode_len=50, temp=None, gamma=0.95,
                        eval_iters=None, frozenlake_probes=32):
    """Run the standard eval suite on a checkpoint and print a learning-curve table.

    Rolls out gridworld 3x3, gridworld 5x5 and frozenlake for `num_episodes`
    episodes each, then tabulates the per-episode return at every episode index
    that is a power of two, with the value-iteration optimum in the last column.
    Frozenlake is stochastic, so its measured cells average `frozenlake_probes`
    greedy rollouts (only the first feeds the exploration stats).
    """
    from gridworld import gridworld
    from frozenlake import frozenlake

    # name -> ((S, A, P, r), probe_reps for the measured greedy episodes)
    envs = {
        "gridworld 3x3": (gridworld(3, goal_idx=8, step_cost=-1.0, goal_r=10.0), 1),
        "gridworld 5x5": (gridworld(5, goal_idx=24, step_cost=-1.0, goal_r=10.0), 1),
        "frozenlake": (frozenlake(), frozenlake_probes),
    }
    powers = [2 ** k for k in range(num_episodes.bit_length()) if 2 ** k <= num_episodes]

    header = f"{'env':<16}" + "".join(f"{f'ep{p}':>11}" for p in powers) + f"{'optimal':>11}"
    print(header)
    print("-" * len(header))
    for name, ((S, A, P, r), probe_reps) in envs.items():
        returns = rollout_autoregressive(
            model_path, S, A, P, r, num_episodes, episode_len, temp=temp, verbose=False,
            eval_iters=eval_iters, greedy_at=set(powers), probe_reps=probe_reps,
        )
        opt = optimal_return(P, r, num_episodes, episode_len, gamma=gamma)
        row = f"{name:<16}" + "".join(f"{returns[p - 1].item():>11.1f}" for p in powers)
        print(row + f"{opt:>11.1f}")


if __name__ == "__main__":
    import argparse

    from prior import GAMMA

    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint, optionally sweeping the eval-time unroll depth "
                    "(the GNN layer is weight-tied, so eval can unroll more than training). "
                    "Example: python evaluation.py checkpoints/step_2000.pt --eval-iters 10 20 40 80",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("checkpoint", help="path to a saved .pt checkpoint")
    p.add_argument("--eval-iters", type=int, nargs="+", default=24,
                   help="unroll depths to sweep (24 = the eval depth used in the paper)")
    p.add_argument("--num-episodes", type=int, default=128)
    p.add_argument("--episode-len", type=int, default=50)
    p.add_argument("--temp", type=float, default=0.3, help="softmax temperature for the rollout policy")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--frozenlake-probes", type=int, default=64,
                   help="greedy rollouts averaged per measured frozenlake cell")
    args = p.parse_args()

    for it in (args.eval_iters or [None]):
        print(f"=== eval_iters={it if it is not None else 'train-depth'} ===")
        evaluate_checkpoint(
            args.checkpoint,
            num_episodes=args.num_episodes,
            episode_len=args.episode_len,
            frozenlake_probes=args.frozenlake_probes,
            temp=args.temp,
            gamma=args.gamma,
            eval_iters=it,
        )
