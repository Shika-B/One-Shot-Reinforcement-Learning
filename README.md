## Overview

Prior-fitted networks amortize Bayesian inference: instead of fitting a model to
one dataset, you train a network on data sampled from a *prior* so that a single
forward pass approximates the posterior on any new dataset drawn from that prior.

Here we apply that idea to reinforcement learning. The "dataset" is the
experience gathered in a MDP, and the quantity we want is the optimal policy
$\pi^\star(\cdot \mid s)$. We sample a broad prior over finite MDPs, compute the
exact optimal action values $Q^\star$ for each with value iteration, and train a
network to predict the induced policy $\pi^\star = \mathrm{softmax}(Q^\star/\tau)$
from *finite, noisy exploration statistics* of that MDP. At test time the model
is dropped into an unseen environment, gathers experience online, and re-plans
its policy in a forward pass. Planning becomes amortized inference rather than
gradient-based learning.

This is the approach described in [our paper](TODO-add-paper-link).

## Running the experiment

Train:

```bash
python train.py --num-layers 20 --steps 15000 --warmup-steps 500 --lr 3e-4 --log-interval 50 --save-interval 500 --eval-iters 24
```

Recreate the evaluations (scripts live in [`baseline/`](baseline/)):

```bash
python baseline/num_layers.py checkpoints/step_15000.pt   # learning curves vs. eval depth K
python baseline/sample_eff.py best_model.pt               # episodes-to-converge vs. UCB-VI / Q-learning
python baseline/offline.py best_model.pt                  # offline policy recovery vs. VI-LCB
```

## Pipeline overview

Four stages, one per source file:

1. **Prior** ([`prior.py`](prior.py)). `sample_MDP_batch` draws a batch of random
   finite MDPs $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ with controlled
   diversity in size, connectivity, geometry, stochasticity and reward sparsity.

2. **Targets** ([`train.py`](train.py), `q_star`). Exact value iteration on the
   *true* dynamics gives the optimal $Q^\star(s,a)$, the label the model learns to
   reproduce.

3. **Training** ([`train.py`](train.py) + [`model.py`](model.py)). The model never
   sees the oracle dynamics. Each step simulates a finite exploration budget
   against $(P, r)$, turns the resulting empirical statistics into model input,
   and regresses the predicted $Q$ onto $Q^\star$.

4. **Evaluation** ([`evaluation.py`](evaluation.py)). The model is dropped into
   held-out environments ([`gridworld.py`](gridworld.py),
   [`frozenlake.py`](frozenlake.py)), explores online, and re-plans once per
   episode. The learning curve is tabulated against the value-iteration optimum.

### Model input

Both training and evaluation feed the model the *same* per-edge features, built
by a single `build_Z` from exploration statistics, so the model's input
distribution matches at train and test time. Each state-action pair $(s,a)$
carries three features:

- $\log(1 + N_{s,a})$ the visit count, which says how much evidence backs this row;
- $\hat r_{s,a}$ the mean observed reward, normalized by the per-MDP reward scale;
- $\hat P(\cdot \mid s,a)$ the empirical transition row (a uniform prior when unvisited).

The exploration budget is randomized per MDP (a log-uniform mean count, then a
Poisson count per $(s,a)$), so the model sees everything from the near-zero-data
regime, where it must fall back to the prior, up to well-sampled rows. Rewards
and the $Q^\star$ target are both normalized by the per-MDP reward scale, so the
model is reward-scale invariant.

### Loss

A soft cross-entropy distillation: the target is the Boltzmann policy
$\mathrm{softmax}(Q^\star/\tau)$ over valid actions, and the loss is its cross
entropy against the model's predicted policy $\pi_\theta(\cdot \mid s)$ (the
softmax of the head's logits). The KL between the predicted and target policies
is logged as a diagnostic only. Optimized with AdamW under linear-warmup /
cosine-decay.

## Prior over MDPs

We sample finite MDPs $M = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$
with controlled variation across:

- state scale $S$, action multiplicity $A$;
- connectivity / outdegree $K$ and latent geometry $g$;
- transition stochasticity $\alpha$;
- reward sparsity and skew.

$\gamma$ is held constant for simplicity. The training distribution spans
near-deterministic structured problems (chains, grids) and unstructured
stochastic graphs, so the model has to handle multi-step propagation under a
wide range of dynamics.

### Sampling procedure

Sizes are drawn per MDP, with an outdegree $O$ shared across all states of a
given MDP:

$$
S \sim \text{LogUniform}(S_{\min}, S_{\max}), \quad
A \sim \text{Uniform}(A_{\min}, A_{\max}), \quad
O \sim \text{PowerLaw}(o) \propto \tfrac{1}{o}.
$$

Each MDP gets a latent geometry $g \in \{\text{chain}, \text{grid}, \text{mesh},
\text{random}\}$, with states placed as points $x_s \in [0,1]^d$ ($d = 1, 2, 3$
respectively; random has no geometry). For each $(s,a)$ the successor set
$\mathcal{N}(s,a)$, of size $O A_{\max}/2$, is chosen as nearest neighbours in
latent space (geometric cases) or uniformly at random.

Transition probabilities are Dirichlet over a size-$O$ support sampled uniformly
from $\mathcal{N}(s,a)$:

$$
P(\cdot \mid s,a) \sim \text{Dirichlet}(\alpha \mathbf{1}_O), \qquad
\alpha \sim \text{LogUniform}(\alpha_{\min}, \alpha_{\max}).
$$

The concentration $\alpha$ interpolates between near-deterministic
($\alpha \ll 1$) and diffuse stochastic ($\alpha \gg 1$) transitions.

Rewards are fixed per $(s,a)$. A per-MDP keep probability controls sparsity, and
each pair's sign is drawn independently with a per-MDP positive fraction
$p_{\text{pos}}$:

$$
r(s,a) = \mathbf{1}\{u_{s,a} < p_{\text{keep}}\}\cdot \sigma_{s,a} \cdot z_{s,a}, \quad
z_{s,a} \sim \text{Beta}(a_r, b_r), \;
\Pr(\sigma_{s,a} = +1) = p_{\text{pos}}.
$$

### Constants used

| Symbol | Meaning | Distribution / value |
|---|---|---|
| $\gamma$ | discount factor | $0.95$ (constant) |
| $S$ | state count | $\text{LogUniform}(2, 32)$ |
| $A$ | action count | $\text{Uniform}(2, 4)$ |
| $O$ | outdegree | $\text{PowerLaw}(o) \propto 1/o$, $o \in [\![1, 6]\!]$ |
| $\alpha$ | transition concentration | $\text{LogUniform}(0.05, 5.0)$ |
| $g$ | latent geometry | uniform over $\{\text{chain}, \text{grid}, \text{mesh}, \text{random}\}$ |
| $p_{\text{keep}}$ | reward keep prob. | $\text{Beta}(2.0, 4.0)$ |
| $z_{s,a}$ | reward magnitude | $\text{Beta}(2.0, 5.0)$ |
| $p_{\text{pos}}$ | positive-reward fraction (per-pair sign) | $\text{Uniform}(0, 1)$ |

## Model

The model is a graph neural network that plans over the MDP graph by repeatedly
propagating information through the transition structure. The core idea is a
hybrid Bellman operator: instead of a fixed expectation over successors, it uses
a learned attention over successor states *biased by the log-transition
probabilities*. This interpolates between classical value iteration (pure
expectation under $P$) and transformer-style attention (data-dependent
reweighting). Iterating the operator $K$ times lets reward and long-range
consequences propagate backwards across the graph; a softmax readout head then
produces the policy $\pi(\cdot \mid s)$ over actions.

Concretely:

- Each edge $(s,a)$ is embedded from its features into $e_{s,a}$; state
  embeddings are initialized by pooling their outgoing edges.
- At each step, every $(s,a)$ attends to successor states $s'$ with logits
  $q_{s,a}\cdot k_{s'}$ plus $\beta \log P(s' \mid s,a)$, aggregates the
  successor messages, and updates the state-action and state embeddings with
  residual MLPs + LayerNorm.
- After $K$ steps, the policy $\pi(\cdot \mid s)$ is read off, as a softmax over
  per-action logits computed from the final state and state-action embeddings.

The propagation layer is **weight-tied** across the $K$ iterations, so the model
can be unrolled deeper at evaluation than during training (`--eval-iters`).

See [`model.py`](model.py) for the exact attention, update and readout
equations.
