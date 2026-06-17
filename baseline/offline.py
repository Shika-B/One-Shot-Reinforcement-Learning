"""Offline RL comparison: in-context model vs. pessimistic value iteration.

The model is, at its core, a map from empirical statistics to a policy. The
offline question is whether a hand-designed tabular estimator can recover as
good a policy *from the same fixed data*. We fix a behavior policy (uniform
random, or epsilon-greedy on a frozen model policy), collect a single offline
dataset of transitions, and at a grid of dataset sizes feed the *identical*
accumulated statistics to both estimators:

  * in-context : the model's greedy policy from the empirical Z it is fed.
  * VI-LCB     : pessimistic value iteration -- value iteration on the empirical
                 model with a Hoeffding penalty subtracted from each reward, so
                 under-visited (s,a) pairs are pushed to a pessimistic floor.

Both produce a policy from the data only; both are scored by the *exact*
discounted value of that policy at the start state, normalized so a random
policy is 0 and the value-iteration optimum is 1 (same metric as baseline.py).
The output is a single panel plotting normalized start-state value against the
number of episodes in the dataset, with color encoding the environment and line
style the method (solid = in-context, dashed = VI-LCB).

    python baseline/offline.py best_model.pt
    python baseline/offline.py best_model.pt --seeds 8 --out off.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld import gridworld
from frozenlake import frozenlake
from prior import GAMMA
from evaluation import _load_model, _empirical_Z, EPS
from sample_eff import policy_value, reference_values, START, EPISODE_LEN

ENV_COLOR = {"gridworld 3x3": "#0072B2", "gridworld 5x5": "#009E73", "frozenlake": "#D55E00"}
# methods are distinguished by line style within a single panel
METHOD_STYLE = {"in-context": "-", "VI-LCB": "--"}


def absorbing_mask(P):
    """States that loop back to themselves under every action (holes, goals)."""
    S, A, _ = P.shape
    diag = P[np.arange(S)[:, None], np.arange(A)[None, :], np.arange(S)[:, None]]
    return np.all(diag > 1.0 - 1e-9, axis=1)


# --------------------------------------------------------------------------- #
# Pessimistic value iteration (VI-LCB)
# --------------------------------------------------------------------------- #
def vi_lcb(N, r_sum, r_scale, terminal, gamma, bonus_c, delta=0.1, sweeps=300):
    """Offline pessimistic VI on the empirical model.

    Plans on the maximum-likelihood model built from the counts, subtracting a
    Hoeffding confidence penalty proportional to 1/sqrt(N(s,a)). Unvisited pairs
    are pinned to the pessimistic value floor. Sees only the dataset (counts,
    empirical rewards, observed reward scale, and episode-termination flags),
    never the true (P, r). `terminal` marks absorbing states, whose continuation
    value is 0; they are never sources of a transition in the data, so without
    this flag the pessimistic floor would poison every backup that reaches them.
    Returns the greedy policy.
    """
    S, A, _ = N.shape
    Nsa = N.sum(-1)                                  # (S, A) visit counts
    visited = Nsa > 0
    Nc = np.maximum(Nsa, 1.0)
    P_hat = N / Nc[..., None]                         # empirical transitions
    r_hat = r_sum / Nc                                # empirical mean reward
    Vmax = float(r_scale) / (1.0 - gamma)
    Vmin = -Vmax
    bonus = bonus_c * np.sqrt(np.log(S * A / delta) / Nc)

    V = np.zeros(S)
    Q = np.zeros((S, A))
    for _ in range(sweeps):
        Q = r_hat - bonus + gamma * (P_hat @ V)
        Q = np.where(visited, Q, Vmin)               # unseen (s,a): fully pessimistic
        Q = np.clip(Q, Vmin, Vmax)
        V = Q.max(axis=1)
        V[terminal] = 0.0                            # absorbing states: no future reward
    return Q.argmax(axis=1)


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
def _step_into(N, r_sum, r_scale, P, r, s, a, rng):
    """Apply one transition, accumulate statistics, return (next_state, r_scale)."""
    s2 = int(rng.choice(P.shape[0], p=P[s, a]))
    N[s, a, s2] += 1.0
    r_sum[s, a] += float(r[s, a])
    return s2, max(r_scale, abs(float(r[s, a])))


def collect_uniform(P, r, n_transitions, rng, absorbing):
    """Collect `n_transitions` under the uniform-random behavior policy."""
    S, A, _ = P.shape
    N = np.zeros((S, A, S)); r_sum = np.zeros((S, A)); r_scale = float(EPS)
    s, steps = START, 0
    for _ in range(n_transitions):
        a = int(rng.integers(A))
        s, r_scale = _step_into(N, r_sum, r_scale, P, r, s, a, rng)
        steps += 1
        if absorbing[s] or steps >= EPISODE_LEN:
            s, steps = START, 0
    return N, r_sum, r_scale


@torch.no_grad()
def frozen_model_policy(model, P, r, eval_iters, pilot, rng, absorbing):
    """Greedy policy from the model after a uniform pilot dataset, frozen for use
    as the (fixed) behavior policy with epsilon-greedy exploration."""
    S = P.shape[0]
    N, r_sum, r_scale = collect_uniform(P, r, pilot, rng, absorbing)
    Z = _empirical_Z(torch.as_tensor(N, dtype=torch.float32),
                     torch.as_tensor(r_sum, dtype=torch.float32),
                     torch.tensor(r_scale), S)
    return model(Z, num_iters=eval_iters)[0].numpy().argmax(axis=1)


# --------------------------------------------------------------------------- #
# One seed: grow the dataset, score both estimators at every checkpoint
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_seed(model, P, r, opt, rand, checkpoints, rng, eval_iters, bonus_c,
             behavior, eps, pilot):
    S, A, _ = P.shape
    denom = max(opt - rand, 1e-8)
    absorbing = absorbing_mask(P)

    if behavior == "model":
        frozen = frozen_model_policy(model, P, r, eval_iters, pilot, rng, absorbing)
        def act(s):
            return int(frozen[s]) if rng.random() > eps else int(rng.integers(A))
    else:
        def act(s):
            return int(rng.integers(A))

    N = np.zeros((S, A, S)); r_sum = np.zeros((S, A)); r_scale = float(EPS)
    model_scores, lcb_scores = [], []
    s, steps, ci = START, 0, 0
    for t in range(1, checkpoints[-1] + 1):
        a = act(s)
        s, r_scale = _step_into(N, r_sum, r_scale, P, r, s, a, rng)
        steps += 1
        if absorbing[s] or steps >= EPISODE_LEN:
            s, steps = START, 0

        if t == checkpoints[ci]:
            Z = _empirical_Z(torch.as_tensor(N, dtype=torch.float32),
                             torch.as_tensor(r_sum, dtype=torch.float32),
                             torch.tensor(r_scale), S)
            pi_m = model(Z, num_iters=eval_iters)[0].numpy().argmax(axis=1)
            pi_l = vi_lcb(N, r_sum, r_scale, absorbing, GAMMA, bonus_c)
            model_scores.append((policy_value(P, r, pi_m, GAMMA) - rand) / denom)
            lcb_scores.append((policy_value(P, r, pi_l, GAMMA) - rand) / denom)
            ci += 1
            if ci == len(checkpoints):
                break
    return np.array(model_scores), np.array(lcb_scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="best_model.pt")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-transitions", type=int, default=2048,
                    help="largest offline dataset size (rounded down to a power of two)")
    ap.add_argument("--min-transitions", type=int, default=8,
                    help="smallest dataset size on the curve")
    ap.add_argument("--eval-iters", type=int, default=24,
                    help="propagation depth K unrolled for the in-context model")
    ap.add_argument("--bonus-c", type=float, default=0.1,
                    help="VI-LCB Hoeffding penalty coefficient (tuned: higher values "
                         "over-penalize rarely-seen goal transitions on frozenlake)")
    ap.add_argument("--behavior", choices=["random", "model"], default="random",
                    help="data-collection policy: uniform random, or epsilon-greedy "
                         "on a frozen model policy")
    ap.add_argument("--eps", type=float, default=0.3,
                    help="exploration rate for the --behavior model collector")
    ap.add_argument("--pilot", type=int, default=512,
                    help="uniform transitions used to freeze the model behavior policy")
    ap.add_argument("--out", default="paper/offline_curves.png")
    args = ap.parse_args()

    model = _load_model(args.model)
    print(f"loaded in-context model from {args.model} (K={args.eval_iters}), "
          f"behavior={args.behavior}")

    checkpoints = [2 ** k for k in range(args.max_transitions.bit_length())
                   if args.min_transitions <= 2 ** k <= args.max_transitions]

    envs = {
        "gridworld 3x3": gridworld(3, goal_idx=8, step_cost=-1.0, goal_r=10.0),
        "gridworld 5x5": gridworld(5, goal_idx=24, step_cost=-1.0, goal_r=10.0),
        "frozenlake":    frozenlake(),
    }

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style in plt.style.available:
            plt.style.use(style)
            break

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 8,
    })
        
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    # each episode is EPISODE_LEN transitions, so report dataset size in episodes
    episodes = np.asarray(checkpoints) / EPISODE_LEN

    for env_name, env in envs.items():
        S, A, P, r = env
        P = np.asarray(P, dtype=np.float64); P /= P.sum(-1, keepdims=True)
        r = np.asarray(r, dtype=np.float64)
        opt, rand = reference_values(P, r, GAMMA)

        m_runs, l_runs = [], []
        for seed in range(args.seeds):
            ms, ls = run_seed(model, P, r, opt, rand, checkpoints,
                              np.random.default_rng(seed), args.eval_iters,
                              args.bonus_c, args.behavior, args.eps, args.pilot)
            m_runs.append(ms); l_runs.append(ls)
        results = {"in-context": np.array(m_runs), "VI-LCB": np.array(l_runs)}

        for method, arr in results.items():
            mean, std = arr.mean(0), arr.std(0)
            c = ENV_COLOR[env_name]
            ax.plot(episodes, mean, marker="o", markersize=4, linewidth=2,
                    color=c, linestyle=METHOD_STYLE[method], zorder=3)
            ax.fill_between(episodes, mean - std, mean + std, color=c, alpha=0.08)

    ax.axhline(1.0, color="0.4", linestyle=(0, (1, 1)), linewidth=1.3, zorder=1)
    ax.axhline(0.0, color="0.6", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    ax.text(episodes[0], 1.0, " optimal", va="bottom", ha="left", fontsize=8, color="0.4")
    ax.text(episodes[0], 0.0, " random", va="bottom", ha="left", fontsize=8, color="0.5")

    ax.set_axisbelow(True)
    ax.set_xlabel("episodes in the offline dataset")
    ax.set_ylabel("normalized start-state value")
    ax.set_title(f"Offline policy recovery vs. dataset size "
                 f"(behavior: {args.behavior}, {args.seeds} seeds)",
                 fontweight="bold")

    # two legends: colour encodes the environment, line style encodes the method
    env_handles = [Line2D([0], [0], color=c, lw=2, label=e)
                   for e, c in ENV_COLOR.items()]
    method_handles = [Line2D([0], [0], color="0.3", lw=2, linestyle=METHOD_STYLE[m], label=m)
                      for m in METHOD_STYLE]
    leg1 = ax.legend(handles=env_handles, title="environment", loc="lower right",
                     frameon=True, fontsize=13)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="method", loc="lower center",
              frameon=True, fontsize=13)

    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
