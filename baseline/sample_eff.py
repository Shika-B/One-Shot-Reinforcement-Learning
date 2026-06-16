"""Tabular RL baselines: UCB-VI and Q-learning on the eval benchmarks.

For each benchmark (gridworld 3x3, gridworld 5x5, frozenlake) we run each
algorithm online and report the *minimal number of episodes before approximate
convergence to an optimal policy*. Convergence is defined on the agent's greedy
(exploitation) policy: its exactly-evaluated normalized score -- with random
policy = 0 and the value-iteration optimum = 1 -- must stay at or above
THRESHOLD for WINDOW consecutive episodes. The reported number is the first
episode of that window, taken as the median over several random seeds.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld import gridworld
from frozenlake import frozenlake
from prior import GAMMA
from evaluation import _load_model, _empirical_Z, EPS

EPISODE_LEN = 50
START = 0
THRESHOLD = 0.95   # fraction of (optimal - random) the greedy policy must reach
WINDOW = 8        # length of the trailing window checked for convergence
HOLD_FRAC = 1.0    # fraction of that window that must be >= THRESHOLD (tolerates
                   # the occasional argmax flip from an agent that still explores)


# --------------------------------------------------------------------------- #
# Exact evaluation helpers (no sampling noise)
# --------------------------------------------------------------------------- #
def value_iteration(P, r, gamma, iters=2000, tol=1e-10, V0=None):
    """Discounted value iteration. P: (S,A,S), r: (S,A). Returns (Q, V)."""
    S = P.shape[0]
    V = np.zeros(S) if V0 is None else V0.copy()
    for _ in range(iters):
        Q = r + gamma * (P @ V)         # (S, A)
        Vn = Q.max(axis=1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return r + gamma * (P @ V), V


def policy_value(P, r, pi, gamma, start=START):
    """Exact discounted value V^pi(start) of a deterministic policy.

    Solves the linear system V = r_pi + gamma P_pi V. We score the discounted
    value -- the objective value iteration, Q-learning and the distilled policy
    all actually optimize -- rather than an undiscounted finite-horizon return.
    This matters: V*(start) from value iteration is a provable maximum of this
    objective, so it is a hard ceiling. The optimal policy scores exactly 1 after
    normalization.
    """
    S = P.shape[0]
    idx = np.arange(S)
    Ppi, rpi = P[idx, pi], r[idx, pi]
    return float(np.linalg.solve(np.eye(S) - gamma * Ppi, rpi)[start])


def reference_values(P, r, gamma):
    """Optimal and uniform-random discounted values at the start state."""
    _, Vstar = value_iteration(P, r, gamma)
    S = P.shape[0]
    Pm, rm = P.mean(axis=1), r.mean(axis=1)
    rand = float(np.linalg.solve(np.eye(S) - gamma * Pm, rm)[START])
    return float(Vstar[START]), rand


# --------------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------------- #
def ucb_vi(P, r, gamma, max_episodes, opt, rand, rng, bonus_c, plan_sweeps=80):
    """Optimistic model-based RL (UCB-VI, discounted/stationary form).

    Plans with value iteration on the empirical model plus a Hoeffding
    exploration bonus, executes the resulting optimistic policy, and updates
    counts. Returns the per-episode normalized score of the greedy (no-bonus)
    exploitation policy.
    """
    S, A, _ = P.shape
    Nsas = np.zeros((S, A, S))
    Nsa = np.zeros((S, A))
    Rsum = np.zeros((S, A))
    Vmax = np.abs(r).max() / (1.0 - gamma)
    denom = max(opt - rand, 1e-8)

    V_opt = np.zeros(S)
    V_grd = np.zeros(S)
    scores = np.empty(max_episodes)
    for ep in range(1, max_episodes + 1):
        Nc = np.maximum(Nsa, 1.0)
        unvis = Nsa == 0
        P_hat = Nsas / Nc[..., None]
        P_hat[unvis] = 1.0 / S                       # uniform prior for unseen rows
        r_hat = Rsum / Nc
        bonus = bonus_c * np.sqrt(np.log(S * A * ep + 1.0) / Nc)
        bonus[unvis] = Vmax                          # fully optimistic for unseen

        # optimistic planning (warm-started)
        for _ in range(plan_sweeps):
            Q = np.minimum(r_hat + bonus + gamma * (P_hat @ V_opt), Vmax)
            V_opt = Q.max(axis=1)
        pi_opt = Q.argmax(axis=1)

        # execute the optimistic policy for one episode
        s = START
        for _ in range(EPISODE_LEN):
            a = pi_opt[s]
            s2 = rng.choice(S, p=P[s, a])
            Nsas[s, a, s2] += 1.0
            Nsa[s, a] += 1.0
            Rsum[s, a] += r[s, a]
            s = s2

        # greedy exploitation policy from the empirical model (no bonus)
        for _ in range(plan_sweeps):
            Qg = r_hat + gamma * (P_hat @ V_grd)
            V_grd = Qg.max(axis=1)
        scores[ep - 1] = (policy_value(P, r, Qg.argmax(axis=1), gamma) - rand) / denom
    return scores


def q_learning(P, r, gamma, max_episodes, opt, rand, rng, eps=0.1, alpha=0.1):
    """Tabular Q-learning, epsilon-greedy with optimistic initialization.

    Returns the per-episode normalized score of the greedy policy argmax_a Q.
    """
    S, A, _ = P.shape
    Vmax = np.abs(r).max() / (1.0 - gamma)
    Q = np.full((S, A), Vmax)                        # optimistic initialization
    denom = max(opt - rand, 1e-8)

    scores = np.empty(max_episodes)
    for ep in range(1, max_episodes + 1):
        s = START
        for _ in range(EPISODE_LEN):
            a = rng.integers(A) if rng.random() < eps else int(Q[s].argmax())
            s2 = rng.choice(S, p=P[s, a])
            Q[s, a] += alpha * (r[s, a] + gamma * Q[s2].max() - Q[s, a])
            s = s2
        scores[ep - 1] = (policy_value(P, r, Q.argmax(axis=1), gamma) - rand) / denom
    return scores


def _softmax_rows(x, temp):
    z = x / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def model_agent(model, P, r, gamma, max_episodes, opt, rand, rng, eval_iters, temp,
                threshold=THRESHOLD, window=WINDOW, hold_frac=HOLD_FRAC):
    """In-context model under the online eval protocol.

    Each episode the model re-plans a policy from the empirical statistics
    gathered so far (same ``build_Z`` features as evaluation.py), the greedy
    policy is exactly evaluated for the convergence score, and one episode of
    temperature-sampled exploration accumulates new data. Stops early once
    converged. Returns the per-episode normalized score of the greedy policy.
    """
    S, A, _ = P.shape
    N = torch.zeros(S, A, S)
    r_sum = torch.zeros(S, A)
    r_scale = float(EPS)                             # running max |reward| seen
    denom = max(opt - rand, 1e-8)
    need = int(np.ceil(hold_frac * window))

    scores = []
    for _ in range(max_episodes):
        Z = _empirical_Z(N, r_sum, torch.tensor(r_scale), S)
        logits = model(Z, num_iters=eval_iters)[0].numpy()   # (S, A)
        sc = (policy_value(P, r, logits.argmax(axis=1), gamma) - rand) / denom
        scores.append(sc)
        # early stop once the trailing window is mostly above threshold
        if len(scores) >= window and \
                sum(v >= threshold for v in scores[-window:]) >= need:
            break

        probs = _softmax_rows(logits, temp)
        s = START
        for _ in range(EPISODE_LEN):
            a = int(rng.choice(A, p=probs[s]))
            s2 = int(rng.choice(S, p=P[s, a]))
            N[s, a, s2] += 1.0
            r_sum[s, a] += float(r[s, a])
            r_scale = max(r_scale, abs(float(r[s, a])))
            s = s2
    return np.array(scores)


# --------------------------------------------------------------------------- #
# Convergence detection + driver
# --------------------------------------------------------------------------- #
def episodes_to_converge(scores, threshold=THRESHOLD, window=WINDOW, hold_frac=HOLD_FRAC):
    """First episode (1-based) starting a length-`window` block in which at least
    `hold_frac` of the scores are >= threshold.

    With hold_frac=1.0 this is the strict "window consecutive successes" rule;
    lower values tolerate isolated dips from a policy that has effectively
    converged but whose argmax still flickers while the agent keeps exploring.
    """
    good = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    need = int(np.ceil(hold_frac * window))
    for i in range(window - 1, len(good)):
        if good[i - window + 1: i + 1].sum() >= need:
            return i - window + 2
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="best_model.pt",
                    help="checkpoint whose in-context convergence is reported "
                         "alongside the tabular baselines")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--max-episodes", type=int, default=1500)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--bonus-c", type=float, default=1.0,
                    help="UCB-VI Hoeffding bonus coefficient")
    ap.add_argument("--eval-iters", type=int, default=24,
                    help="propagation depth K unrolled for the in-context model")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="exploration temperature for the in-context model")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="normalized score the greedy policy must reach")
    ap.add_argument("--window", type=int, default=WINDOW,
                    help="length of the trailing window checked for convergence")
    ap.add_argument("--hold-frac", type=float, default=HOLD_FRAC,
                    help="fraction of the window that must clear the threshold "
                         "(1.0 = strict consecutive; lower tolerates dips)")
    args = ap.parse_args()
    conv = dict(threshold=args.threshold, window=args.window, hold_frac=args.hold_frac)

    def npz(env):
        S, A, P, r = env
        P = np.asarray(P, dtype=np.float64)
        P /= P.sum(axis=-1, keepdims=True) # renormalize (kill float drift)
        return P, np.asarray(r, dtype=np.float64)

    envs = {
        "gridworld 3x3": npz(gridworld(3, goal_idx=8, step_cost=-1.0, goal_r=10.0)),
        "gridworld 5x5": npz(gridworld(5, goal_idx=24, step_cost=-1.0, goal_r=10.0)),
        "frozenlake":    npz(frozenlake()),
    }

    algos = {}
    try:
        model = _load_model(args.model)
        algos["in-context"] = lambda P, r, opt, rand, rng: model_agent(
            model, P, r, args.gamma, args.max_episodes, opt, rand, rng,
            args.eval_iters, args.temp, **conv)
        print(f"loaded in-context model from {args.model} (K={args.eval_iters})")
    except FileNotFoundError:
        print(f"(no checkpoint at {args.model!r}; tabular baselines only)")
    algos["UCB-VI"] = lambda P, r, opt, rand, rng: ucb_vi(
        P, r, args.gamma, args.max_episodes, opt, rand, rng, args.bonus_c)
    algos["Q-learning"] = lambda P, r, opt, rand, rng: q_learning(
        P, r, args.gamma, args.max_episodes, opt, rand, rng)

    print(f"episodes to converge (greedy normalized score >= {args.threshold} "
          f"for >= {int(np.ceil(args.hold_frac * args.window))}/{args.window} of a "
          f"trailing window), median over {args.seeds} seeds\n")
    header = f"{'benchmark':<16}" + "".join(f"{name:>14}" for name in algos)
    print(header)
    print("-" * len(header))

    for env_name, (P, r) in envs.items():
        opt, rand = reference_values(P, r, args.gamma)
        cells = []
        for algo_name, run in algos.items():
            results = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                scores = run(P, r, opt, rand, rng)
                results.append(episodes_to_converge(scores, **conv))
            reached = [x for x in results if x is not None]
            if not reached:
                cells.append(f">{args.max_episodes}")
            else:
                med = int(np.median(reached))
                tag = "" if len(reached) == args.seeds else f" ({len(reached)}/{args.seeds})"
                cells.append(f"{med}{tag}")
        row = f"{env_name:<16}" + "".join(f"{c:>14}" for c in cells)
        print(row)


if __name__ == "__main__":
    main()
