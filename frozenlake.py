import torch

# Actions: 0=up, 1=down, 2=left, 3=right
_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# the two perpendicular ("slip") directions for each action
_PERP = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


def frozenlake(N=4, holes=(5, 7, 11, 12), goal_idx=None, slip=0.2,
               step_cost=0.0, hole_r=-1.0, goal_r=1.0):
    """Slippery N x N gridworld (FrozenLake-style), an exploitation benchmark.

    Returns (S, A, P, r): S = N^2 states, A = 4 actions, transitions P (S, A, S)
    and rewards r (S, A). A move succeeds with prob 1-slip; otherwise the agent
    slips to one of the two perpendicular directions (each prob slip/2). Off-grid
    moves stay in place. Holes and the goal are absorbing. Rewards are
    expected-over-next-state, so the value of a move already accounts for slip
    risk -- the optimal policy must steer around holes rather than beeline to the
    goal, even though every state is trivial to reach during exploration.
    """
    S, A = N * N, 4
    if goal_idx is None:
        goal_idx = S - 1
    holes = set(holes)
    absorbing = holes | {goal_idx}

    def cell_reward(s):
        return goal_r if s == goal_idx else hole_r if s in holes else step_cost

    P = torch.zeros(S, A, S)
    r = torch.zeros(S, A)
    for s in range(S):
        if s in absorbing:
            P[s, :, s] = 1.0          # absorbing: stay put, no further reward
            continue
        row, col = divmod(s, N)
        for a in range(A):
            outcomes = [(a, 1.0 - slip), (_PERP[a][0], slip / 2), (_PERP[a][1], slip / 2)]
            for move, prob in outcomes:
                dr, dc = _MOVES[move]
                nr, nc = min(max(row + dr, 0), N - 1), min(max(col + dc, 0), N - 1)
                ns = nr * N + nc
                P[s, a, ns] += prob
                r[s, a] += prob * cell_reward(ns)
    return S, A, P, r
