"""Plot in-context learning curves for a checkpoint, swept over eval depth K.

The propagation layer is weight-tied, so the number of layers K unrolled at
evaluation can differ from training. This script rolls out the model under the
same protocol as ``evaluation.py`` at every K in ``K_VALUES`` and, for each
held-out environment (gridworld 3x3, gridworld 5x5, frozenlake), records the
normalized greedy score at every episode index that is a power of two.

Layout: one panel per evaluation depth K; within each panel, one colored line
per environment (plus a dotted "optimal" and dashed "random" reference line).

    python baseline/num_layers.py checkpoints/step_15000.pt
    python baseline/num_layers.py checkpoints/step_15000.pt --out curve.png
"""
import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe (e.g. on a remote training box)
import matplotlib.pyplot as plt
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import rollout_autoregressive, optimal_return
from gridworld import gridworld
from frozenlake import frozenlake
from prior import GAMMA

K_VALUES = [4, 8, 16, 20, 24, 38]

# Episodes used only to estimate the normalization constants R_opt / R_rand
NORM_EPISODES = 2000

ENV_PALETTE = ["#0072B2", "#FE6100", "#009E73"]


@torch.no_grad()
def random_return(P, r, num_episodes, episode_len):
    """Mean return of a uniform-random policy, same rollout protocol as optimal_return."""
    P, r = P.float(), r.float()
    A = r.shape[1]
    states = torch.zeros(num_episodes, dtype=torch.long)
    returns = torch.zeros(num_episodes)
    for _ in range(episode_len):
        a = torch.randint(0, A, (num_episodes,))
        returns += r[states, a]
        states = torch.multinomial(P[states, a], 1).squeeze(-1)
    return returns.mean().item()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_path", help="path to a saved .pt checkpoint")
    ap.add_argument("--episode-len", type=int, default=50)
    ap.add_argument("--temp", type=float, default=0.3,
                    help="exploration temperature; the measured power-of-two "
                         "episodes are greedy regardless")
    ap.add_argument("--frozenlake-probes", type=int, default=256,
                    help="greedy rollouts averaged per measured frozenlake cell")
    ap.add_argument("--out", default="paper/learning_curves.png")
    args = ap.parse_args()

    num_episodes = 512
    powers = [2 ** k for k in range(10)]  # 1, 2, 4, ..., 512

    # name -> ((S, A, P, r), probe_reps)
    envs = {
        "gridworld 3x3": (gridworld(3, goal_idx=8, step_cost=-1.0, goal_r=10.0), 1),
        "gridworld 5x5": (gridworld(5, goal_idx=24, step_cost=-1.0, goal_r=10.0), 1),
        "frozenlake":    (frozenlake(), args.frozenlake_probes),
    }

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style in plt.style.available:
            plt.style.use(style)
            break

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 10,
        "ytick.labelsize": 13,
        "legend.fontsize": 8,
    })

    env_color = {name: ENV_PALETTE[i % len(ENV_PALETTE)]
                 for i, name in enumerate(envs)}

    # The normalization constants depend only on the environment, not on K, so
    # compute them once per env (NORM_EPISODES each) and reuse across panels.
    norm = {}
    for name, ((S, A, P, r), probe_reps) in envs.items():
        opt = optimal_return(P, r, NORM_EPISODES, args.episode_len, gamma=GAMMA)
        rand = random_return(P, r, NORM_EPISODES, args.episode_len)
        denom = max(opt - rand, 1e-8)
        norm[name] = dict(S=S, A=A, P=P, r=r, rand=rand, denom=denom,
                          probe_reps=probe_reps)

    fig, axes = plt.subplots(
        math.ceil(len(K_VALUES) / 2), 2, figsize=(len(K_VALUES), 7.0),
        sharey=True, constrained_layout=True,
    )
    if len(K_VALUES) == 1:
        axes = [axes]

    for ax, K in zip(axes.flatten(), K_VALUES):
        for name in envs:

            n = norm[name]
            returns = rollout_autoregressive(
                args.model_path, n["S"], n["A"], n["P"], n["r"],
                num_episodes, args.episode_len,
                temp=args.temp, verbose=False, eval_iters=K,
                greedy_at=set(powers), probe_reps=n["probe_reps"],
            )
            ys = [(returns[p - 1].item() - n["rand"]) / n["denom"] for p in powers]
            ax.plot(powers, ys, marker="o", markersize=5, linewidth=2,
                    color=env_color[name], label=name, zorder=3)

        ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.8,
                   label="optimal", zorder=2)
        ax.axhline(0.0, color="0.7", linestyle="--", linewidth=1.2, alpha=0.8,
                   label="random", zorder=1)

        ax.set_xscale("log", base=2)
        ax.set_xticks(powers)
        ax.set_xticklabels([str(p) for p in powers])
        ax.set_title(f"$K = {K}$")
        ax.set_axisbelow(True)

    # fig.set_ylabel("normalized score")
    axes[2][0].legend(frameon=True, loc="lower right")
    fig.supxlabel("number of episodes in context (log scale)")
    fig.supylabel("normalized score")
    fig.suptitle("In-context returns with evaluation depth $K$",
                 fontsize=18, fontweight="bold")

    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()