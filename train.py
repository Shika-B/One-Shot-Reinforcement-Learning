"""Train the GNN planner by amortized value iteration over MDPs from the prior.

Each step samples a batch of MDPs, computes the optimal Q* with value iteration
on the true dynamics, and regresses the model's prediction onto it. Run e.g.:

    python train.py --steps 20000 --d-model 128 --lr 3e-4 --device cuda
"""
import argparse
import math
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model import GNNIterModel
from prior import sample_MDP_batch, GAMMA, _loguniform
from evaluation import evaluate_checkpoint, build_Z, EPS

# Per-(s,a) exploration budget simulated at train time: a per-MDP mean count is
# drawn log-uniformly in this range, spanning the prior regime (~0 data) to
# well-sampled rows so the visit-count feature carries real evidence.
BUDGET_MIN, BUDGET_MAX = 0.1, 10.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    p.add_argument("--d-model", type=int, default=256, help="hidden size (must be divisible by 8 heads)")
    p.add_argument("--num-layers", type=int, default=20, help="number of GNN propagation steps")
    p.add_argument("--eval-iters", type=int, default=None, help="propagation steps at eval (defaults to num-layers; weights are tied)")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=1.0, help="attention regularization: strength of the log-transition bias (1.0 = exact Bellman expectation under P_hat)")
    p.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500, help="linear LR warmup before cosine decay")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128, help="MDPs per step")
    p.add_argument("--steps", type=int, default=15_000, help="number of training steps")
    p.add_argument("--vi-iters", type=int, default=500, help="value-iteration sweeps for the Q* target")
    p.add_argument("--target-tau", type=float, default=0.2, help="softmax temperature for the soft-CE distillation target")
    p.add_argument("--temp", type=float, default=0.3)
    # runtime
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--save-path", default="checkpoints/")
    p.add_argument("--resume", default="", help="checkpoint path to resume model + optimizer + step from")
    return p.parse_args()


def sample_counts(P, r, state_mask, action_mask):
    """Simulate a finite exploration budget against the true dynamics (P, r).

    Returns (N, r_sum): transition counts (B, S, A, S) and reward sums
    (B, S, A). A per-MDP mean budget is drawn log-uniformly, then per-(s,a)
    visit counts are Poisson, so the model sees coverage from near-zero (the
    prior regime, where P_hat falls back to uniform) to well-sampled rows. This
    makes the model's input distribution match what `_empirical_Z` produces at
    eval time. Rewards are deterministic per (s,a), so r_sum = n_sa * r.
    """
    B, S, A, _ = P.shape
    device = P.device
    row_valid = state_mask[:, :, None] & action_mask[:, None, :]          # (B, S, A)

    lam = _loguniform(BUDGET_MIN, BUDGET_MAX, (B, 1, 1)).to(device)
    n_sa = (torch.poisson(lam.expand(B, S, A)).long()) * row_valid        # (B, S, A)
    n_max = int(n_sa.max().clamp_min(1))

    # Sample n_max successors per row, then keep only the first n_sa of each.
    flat_P = P.reshape(B * S * A, S)
    safe_P = torch.where(flat_P.sum(-1, keepdim=True) > 0, flat_P, torch.ones_like(flat_P))
    draws = torch.multinomial(safe_P, n_max, replacement=True).view(B, S, A, n_max)
    keep = (torch.arange(n_max, device=device) < n_sa[..., None]).float()
    N = torch.zeros(B, S, A, S, device=device).scatter_add_(-1, draws, keep)

    r_sum = n_sa.float() * r
    return N, r_sum


def compute_loss(pred, target, state_mask, action_mask, row_valid, tau):
    """Soft cross-entropy distillation against softmax(Q* / tau).

    Target policy is softmax(Q*/tau) over valid actions; loss is the soft CE
    -(target . log_softmax(Q_pred)), averaged over valid states. Also returns
    mse and kl as logged-only diagnostics. Returns (loss, mse, kl).
    """
    masked = (~action_mask[:, None, :]).expand_as(pred)
    log_p = torch.log_softmax(pred.masked_fill(masked, -1e9), dim=-1)
    log_q = torch.log_softmax(target.masked_fill(masked, -1e9), dim=-1)
    sm = state_mask.float()
    denom = sm.sum().clamp_min(1)

    target_pol = torch.softmax(target.masked_fill(masked, -1e9) / tau, dim=-1)
    loss = (-(target_pol * log_p).sum(-1) * sm).sum() / denom

    # diagnostics (not part of the objective)
    mse = ((pred - target) ** 2 * row_valid).sum() / row_valid.sum().clamp_min(1)
    kl = ((log_q.exp() * (log_q - log_p)).sum(-1) * sm).sum() / denom

    return loss, mse, kl


@torch.no_grad()
def q_star(P, r, state_mask, action_mask, gamma, iters):
    """Optimal Q*(s,a) via value iteration on the true dynamics (batched)."""
    row_valid = (state_mask[:, :, None] & action_mask[:, None, :]).float()
    Q = torch.zeros_like(r)
    for _ in range(iters):
        V = Q.masked_fill(~action_mask[:, None, :], float("-inf")).max(-1).values
        V = V * state_mask.float()                       # unreachable states have value 0
        EV = torch.einsum("bsan,bn->bsa", P, V)          # E_{s'} V(s')
        Q = (r + gamma * EV) * row_valid
    return Q, row_valid


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = GNNIterModel(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 1
    sched_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model = ckpt["model"].to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        sched_state = ckpt.get("sched")
        start_step = ckpt.get("step", 0) + 1
        print(f"resumed from {args.resume} at step {start_step}")

    def lr_lambda(s):
        if s < args.warmup_steps:
            return (s + 1) / max(args.warmup_steps, 1)
        progress = min(max((s - args.warmup_steps) / max(args.steps - args.warmup_steps, 1), 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = LambdaLR(opt, lr_lambda)
    if sched_state is not None:
        sched.load_state_dict(sched_state)

    print(f"params: {sum(p.numel() for p in model.parameters()):,} | device: {device}")
    running = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    for step in range(start_step, args.steps + 1):
        S, A, P, r = sample_MDP_batch(args.batch_size)
        Smax, Amax = P.shape[1], P.shape[2]
        P, r = P.to(device), r.to(device)
        state_mask = (torch.arange(Smax, device=device) < S.to(device)[:, None])
        action_mask = (torch.arange(Amax, device=device) < A.to(device)[:, None])

        target, row_valid = q_star(P, r, state_mask, action_mask, args.gamma, args.vi_iters)

        # Normalise by the true per-MDP reward scale so the model predicts
        # reward-scale-invariant Q-values; eval mirrors this with the running
        # max |reward|. Build Z from simulated exploration, not the oracle P.
        r_scale = r.abs().amax(dim=(1, 2), keepdim=True).clamp_min(EPS)     # (B, 1, 1)
        N, r_sum = sample_counts(P, r, state_mask, action_mask)
        Z, _ = build_Z(N, r_sum, state_mask, r_scale=r_scale)
        target = target / r_scale

        pred = model(Z, state_mask, action_mask)
        loss, mse, kl = compute_loss(pred, target, state_mask, action_mask, row_valid, args.target_tau)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        running["loss"] += loss.item()
        running["mse"] += mse.item()
        running["kl"] += kl.item()

        if step % args.log_interval == 0:
            n = args.log_interval
            print(f"step {step:>6}/{args.steps} | loss {running['loss']/n:.4f} "
                  f"| mse {running['mse']/n:.4f} | kl {running['kl']/n:.4f} "
                  f"| lr {sched.get_last_lr()[0]:.2e}")
            running = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
        if step % args.save_interval == 0 or step == args.steps:
            os.makedirs(args.save_path, exist_ok=True)
            ckpt_path = os.path.join(args.save_path, f"step_{step}.pt")
            torch.save({"model": model, "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "cfg": vars(args), "step": step}, ckpt_path)
            print(f"--- eval @ step {step} ---")
            evaluate_checkpoint(ckpt_path, gamma=args.gamma, temp=args.temp, eval_iters=args.eval_iters)

    print(f"done. saved to {args.save_path}")


if __name__ == "__main__":
    main()
