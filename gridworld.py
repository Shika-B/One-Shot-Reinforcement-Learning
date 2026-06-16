import torch

# Actions: 0=up, 1=down, 2=left, 3=right
_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def gridworld(N, goal_idx, step_cost, goal_r):
    """Build a deterministic N x N gridworld MDP.

    Returns (S, A, P, r): S = N^2 states, A = 4 actions, transitions
    P (S, A, S) and rewards r (S, A). Moves that leave the grid stay in place.
    The goal at `goal_idx` is self-absorbing (every action loops back to it
    with zero reward); every other step yields `step_cost`, plus `goal_r` when
    the move lands on the goal.
    """
    S, A = N * N, 4
    P = torch.zeros(S, A, S)
    r = torch.full((S, A), float(step_cost))

    for s in range(S):
        if s == goal_idx:
            P[s, :, goal_idx] = 1.0
            r[s, :] = 0.0
            continue
        row, col = divmod(s, N)
        for a, (dr, dc) in enumerate(_MOVES):
            nr, nc = min(max(row + dr, 0), N - 1), min(max(col + dc, 0), N - 1)
            ns = nr * N + nc
            P[s, a, ns] = 1.0
            if ns == goal_idx:
                r[s, a] += goal_r

    return S, A, P, r
