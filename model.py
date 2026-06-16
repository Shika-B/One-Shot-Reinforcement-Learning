import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim, act_cls, dropout):
    """Two-layer MLP: Linear -> activation -> dropout -> Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        act_cls(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class SuccessorAttention(nn.Module):
    """Attention over successor states, biased by the transition probabilities.

    Each (s,a) attends to successor states s' with logits q.k plus a fixed
    weight `beta` (a constant hyperparameter, not learned) times log P(s'|s,a).
    """

    def __init__(self, d_model: int, num_heads: int = 8, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(2 * d_model, d_model)  # FIXED
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Attention regularization: strength of the log-transition bias that
        # pulls successor attention toward the empirical dynamics. beta=1
        # recovers the exact Bellman expectation under P_hat; smaller values
        # free the attention, larger values pin it harder to P_hat.
        self.beta = beta

    def forward(self, h_s, h_sa, e_sa, P_hat):
        """Aggregate successor messages for every (s,a). Returns (B, S, A, d)."""
        B, S, d = h_s.shape
        A = h_sa.shape[2]
        H = self.num_heads
        Dh = self.head_dim

        # queries
        q = self.q_proj(torch.cat([h_sa, e_sa], dim=-1))
        q = q.view(B, S, A, H, Dh)

        # keys/values
        k = self.k_proj(h_s).view(B, S, H, Dh)
        v = self.v_proj(h_s).view(B, S, H, Dh)

        # attention logits (n indexes successor states s')
        attn = torch.einsum("bsahd,bnhd->bsahn", q, k) / (Dh**0.5)

        attn = attn + self.beta * torch.log(P_hat.clamp_min(self.eps)).unsqueeze(3)

        attn = F.softmax(attn, dim=-1)

        msg = torch.einsum("bsahn,bnhd->bsahd", attn, v)

        msg = msg.reshape(B, S, A, d)

        return self.out_proj(msg)


class GNNLayer(nn.Module):
    """One propagation step: successor attention, then state-action and state updates.

    Both updates are residual + LayerNorm, with padding masked out via the
    state mask `sm` and action mask `am`.
    """

    def __init__(
        self,
        d: int,
        dropout: float,
        act_cls,
        num_heads: int = 8,
        beta: float = 1.0,
    ):
        super().__init__()

        self.attn = SuccessorAttention(
            d_model=d,
            num_heads=num_heads,
            beta=beta,
        )

        self.sa_update = _mlp(
            3 * d,
            d,
            d,
            act_cls,
            dropout,
        )

        self.s_update = _mlp(
            2 * d,
            d,
            d,
            act_cls,
            dropout,
        )

        self.ln_sa = nn.LayerNorm(d)
        self.ln_s = nn.LayerNorm(d)

    def forward(
        self,
        h_s,
        h_sa,
        e_sa,
        P_hat,
        sm,
        am,
    ):
        """Update (h_s, h_sa) one step and return the new pair."""
        msg_sa = self.attn(
            h_s,
            h_sa,
            e_sa,
            P_hat,
        )

        h_sa_new = h_sa + self.sa_update(
            torch.cat(
                [h_sa, msg_sa, e_sa],
                dim=-1,
            )
        )

        h_sa_new = self.ln_sa(h_sa_new)

        sa_mask = (sm[:, :, None] * am[:, None, :]).unsqueeze(-1)

        h_sa_new = h_sa_new * sa_mask

        h_sa_real = h_sa_new * am[:, None, :, None]

        agg_a = h_sa_real.sum(dim=2) / am.sum(dim=1, keepdim=True).clamp_min(
            1.0
        ).unsqueeze(-1)

        h_s_new = h_s + self.s_update(
            torch.cat(
                [h_s, agg_a],
                dim=-1,
            )
        )

        h_s_new = self.ln_s(h_s_new)
        h_s_new = h_s_new * sm.unsqueeze(-1)

        return h_s_new, h_sa_new


class GNNIterModel(nn.Module):
    """Iterated GNN that predicts Q-values over an MDP graph.

    Encodes edge features, runs `cfg.num_layers` shared propagation steps to
    spread reward and transition information across the graph, then reads out
    Q(s,a) from the final state and state-action embeddings.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        d = cfg.d_model

        act_cls = nn.GELU if cfg.activation == "gelu" else nn.ReLU

        self.edge_enc = _mlp(
            2,
            d,
            d,
            act_cls,
            cfg.dropout,
        )

        self.s_init = _mlp(
            d,
            d,
            d,
            act_cls,
            cfg.dropout,
        )

        self.layer = GNNLayer(
            d=d,
            dropout=cfg.dropout,
            act_cls=act_cls,
            num_heads=8,
            beta=getattr(cfg, "beta", 1.0),
        )

        self.q_head = _mlp(
            3 * d,
            d,
            1,
            act_cls,
            cfg.dropout,
        )

    def forward(
        self,
        Z,
        state_mask=None,
        action_mask=None,
        num_iters=None,
    ):
        """Map edge features Z (B, S, A, 2+S) to Q-values (B, S, A).

        Z packs log(1+N), reward and the transition row P(.|s,a) per edge.
        Missing actions are masked to a large negative Q. `num_iters` overrides
        how many times the shared layer is unrolled (defaults to cfg.num_layers);
        the weights are tied, so eval can propagate further than training.
        """
        B, S, A, F_ = Z.shape

        if state_mask is None:
            state_mask = Z.new_ones(
                B,
                S,
                dtype=torch.bool,
            )

        if action_mask is None:
            action_mask = Z.new_ones(
                B,
                A,
                dtype=torch.bool,
            )

        sm = state_mask.float()
        am = action_mask.float()

        sa_mask = sm[:, :, None] * am[:, None, :]

        log1p_N = Z[..., 0]
        r_hat = Z[..., 1]
        P_hat = Z[..., 2 : 2 + S]

        e_feat = torch.stack(
            [log1p_N, r_hat],
            dim=-1,
        )

        e_sa = self.edge_enc(e_feat) * sa_mask.unsqueeze(-1)

        a_denom = am.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)

        h_s_in = (e_sa * am[:, None, :, None]).sum(dim=2) / a_denom

        h_s = self.s_init(h_s_in)
        h_s = h_s * sm.unsqueeze(-1)

        h_sa = e_sa

        iters = self.cfg.num_layers if num_iters is None else num_iters
        for _ in range(iters):
            h_s, h_sa = self.layer(
                h_s,
                h_sa,
                e_sa,
                P_hat,
                sm,
                am,
            )

        h_s_bcast = h_s[:, :, None, :].expand(
            B,
            S,
            A,
            h_s.shape[-1],
        )

        q_in = torch.cat(
            [
                h_s_bcast,
                h_sa,
                e_sa,
            ],
            dim=-1,
        )

        Q = self.q_head(q_in).squeeze(-1)

        Q = Q.masked_fill(
            ~action_mask[:, None, :],
            -1e9,
        )

        return Q
