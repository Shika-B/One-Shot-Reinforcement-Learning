import math
import torch

GAMMA = 0.95

S_MIN, S_MAX = 2, 32
A_MIN, A_MAX = 2, 4
K_MIN, K_MAX = 1, 6
ALPHA_MIN, ALPHA_MAX = 0.05, 5.0

P_KEEP_A, P_KEEP_B = 2.0, 4.0
REWARD_A, REWARD_B = 2.0, 5.0
REWARD_SCALE = 1.0

GEOM_DIMS = torch.tensor([1, 2, 3, 3])  # chain, grid, mesh, random
RANDOM_GEOM_IDX = 3


def _loguniform(lo, hi, shape):
    """Draw `shape` samples uniform on a log scale between `lo` and `hi`."""
    return torch.empty(shape).uniform_(math.log(lo), math.log(hi)).exp()


def sample_MDP_batch(B):
    """Sample a batch of B random MDPs.

    Returns (S, A, P, r): state counts (B,), action counts (B,), transition
    probabilities (B, S_MAX, A_MAX, S_MAX) and rewards (B, S_MAX, A_MAX). 
    Each MDP gets its own size, outdegree, latent geometry, 
    transition stochasticity and reward sparsity.
    """
    # Size parameters
    S = _loguniform(S_MIN, S_MAX, (B,)).round().long().clamp(1, S_MAX)
    A = torch.randint(A_MIN, A_MAX + 1, (B,))

    ks = torch.arange(K_MIN, K_MAX + 1).float()
    K = ks[torch.distributions.Categorical(1.0 / ks).sample((B,))].long().minimum(S)
    M = (K * A_MAX // 2).minimum(S)  # |N(s,a)| = K*A_max/2, capped at S
    alpha = _loguniform(ALPHA_MIN, ALPHA_MAX, (B,))

    # Masks
    valid_s = torch.arange(S_MAX) < S[:, None]
    valid_a = torch.arange(A_MAX) < A[:, None]
    row_valid = valid_s[:, :, None] & valid_a[:, None, :]

    # Nearest neighbours in [0,1]^d, or random ordering.
    geom = torch.randint(len(GEOM_DIMS), (B,))
    x = torch.rand(B, S_MAX, 3)
    x = x * (torch.arange(3) < GEOM_DIMS[geom][:, None]).float()[:, None, :]
    dist = torch.cdist(x, x)
    dist = torch.where(
        (geom == RANDOM_GEOM_IDX)[:, None, None], torch.rand_like(dist), dist
    )
    dist = dist.masked_fill(~valid_s[:, None, :], float("inf"))
    rank = dist.argsort(-1).argsort(-1)
    pool = rank < M[:, None, None]

    # Pick K successors per (s,a) uniformly from each state's pool
    score = torch.rand(B, S_MAX, A_MAX, S_MAX).masked_fill(~pool[:, :, None, :], -1.0)
    sel_rank = score.argsort(-1, descending=True).argsort(-1)
    chosen = sel_rank < K[:, None, None, None]

    # Transitions
    # Dirichlet(alpha) over the chosen support
    # is the same as normalised Gamma(alpha) samples
    conc = alpha[:, None, None, None].expand(B, S_MAX, A_MAX, S_MAX)
    w = torch.distributions.Gamma(conc, 1.0).sample()
    w = torch.where(chosen, w.clamp_min(1e-8), torch.zeros_like(w))
    P = w / w.sum(-1, keepdim=True)
    P = P * row_valid[..., None]

    # Rewards. Each (s,a) gets an independent reward sign, biased by a per-MDP
    # positive fraction p_pos ~ U[0,1]. This spans all-negative, all-positive
    # and mixed-sign MDPs (p_pos near 0.5), the last of which is needed to cover
    # step-cost-plus-goal structures where penalties and a sparse reward coexist.
    p_pos = torch.rand(B, 1, 1)
    sigma = torch.where(torch.rand(B, S_MAX, A_MAX) < p_pos, 1.0, -1.0)
    p_keep = torch.distributions.Beta(P_KEEP_A, P_KEEP_B).sample((B,))
    z = torch.distributions.Beta(REWARD_A, REWARD_B).sample((B, S_MAX, A_MAX))
    keep = torch.rand(B, S_MAX, A_MAX) < p_keep[:, None, None]
    r = keep * sigma * REWARD_SCALE * z
    r = r * row_valid

    return S, A, P, r
