"""Exact tabular checks of the PIRC maths summarised in docs/pirc/README.md.

Everything here is solved exactly with linear algebra on small random MDPs, so a
non-zero distance is a property of the maths and not of sampling or of a learned
value function. Standalone: needs Python 3.10 or later and numpy only, and does
not import the rest of this repository. Run from the repository root:

    python docs/pirc/verify_maths.py

The numbered checks match the numbers in square brackets in docs/pirc/README.md.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
SEED = 0


@dataclass(frozen=True)
class MDP:
    """A finite MDP. A state flagged terminal has no outgoing reward and a fixed value."""

    transition: Array  # (S, A, S), each row sums to 1
    reward: Array  # (S, A, S)
    gamma: float
    terminal: NDArray[np.bool_]  # (S,)

    @property
    def n_states(self) -> int:
        return self.transition.shape[0]

    @property
    def n_actions(self) -> int:
        return self.transition.shape[1]


def uniform_policy(mdp: MDP) -> Array:
    return np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)


def as_reward(per_state_action: Array, mdp: MDP) -> Array:
    """Lift a function of (s, a) to a reward on (s, a, s')."""
    return np.repeat(per_state_action[:, :, None], mdp.n_states, axis=2)


def policy_value(mdp: MDP, policy: Array, reward: Array, terminal_value: float = 0.0) -> Array:
    """Value of `policy` under `reward`, with the value of terminal states held fixed."""
    live = ~mdp.terminal
    p_pi = np.einsum("sa,sat->st", policy, mdp.transition)
    r_pi = np.einsum("sa,sat,sat->s", policy, mdp.transition, reward)
    value = np.full(mdp.n_states, terminal_value)
    rhs = r_pi[live] + mdp.gamma * p_pi[np.ix_(live, mdp.terminal)].sum(axis=1) * terminal_value
    lhs = np.eye(int(live.sum())) - mdp.gamma * p_pi[np.ix_(live, live)]
    value[live] = np.linalg.solve(lhs, rhs)
    return value


def q_from_value(mdp: MDP, reward: Array, value: Array) -> Array:
    return np.einsum("sat,sat->sa", mdp.transition, reward + mdp.gamma * value[None, None, :])


def logsumexp(x: Array, weights: Array | None = None) -> Array:
    top = x.max(axis=1, keepdims=True)
    w = 1.0 if weights is None else weights
    return (top + np.log((w * np.exp(x - top)).sum(axis=1, keepdims=True)))[:, 0]


def soft_optimal(
    mdp: MDP, alpha: float, prior: Array | None = None, sweeps: int = 20000, start: Array | None = None
) -> tuple[Array, Array]:
    """Entropy-regularised optimum: pi ~ prior * exp(Q/alpha). Returns (policy, soft value).

    With `prior=None` the regulariser is Shannon entropy (the paper's model). With a prior
    it is -alpha * KL(pi || prior), the form used when fine-tuning from a reference policy.
    """
    value = np.zeros(mdp.n_states) if start is None else start.copy()
    for _ in range(sweeps):
        q = q_from_value(mdp, mdp.reward, value)
        new = alpha * logsumexp(q / alpha, prior)
        new[mdp.terminal] = 0.0
        if np.abs(new - value).max() < 1e-13:
            value = new
            break
        value = new
    q = q_from_value(mdp, mdp.reward, value)
    logits = q / alpha + (0.0 if prior is None else np.log(prior))
    policy = np.exp(logits - logsumexp(logits)[:, None])
    return policy, value


def boltzmann_fixed_point(mdp: MDP, beta: float, sweeps: int = 20000) -> Array:
    """Policy with pi = softmax(Q^pi / beta): the stationary point of policy gradient with a
    per-state entropy bonus (entropy is not added to the return)."""
    policy = uniform_policy(mdp)
    for _ in range(sweeps):
        q = q_from_value(mdp, mdp.reward, policy_value(mdp, policy, mdp.reward))
        target = np.exp(q / beta - logsumexp(q / beta)[:, None])
        new = 0.5 * policy + 0.5 * target
        if np.abs(new - policy).max() < 1e-14:
            return new
        policy = new
    raise RuntimeError("Boltzmann fixed point did not converge")


def occupancy(mdp: MDP, policy: Array, start: Array | None = None, discount: float | None = None) -> Array:
    """Visitation weights over (s, a, s'). Defaults: uniform start over live states, discount gamma."""
    live = ~mdp.terminal
    start = live / live.sum() if start is None else start
    discount = mdp.gamma if discount is None else discount
    p_pi = np.einsum("sa,sat->st", policy, mdp.transition)
    p_pi[mdp.terminal] = 0.0
    visits = np.linalg.solve(np.eye(mdp.n_states) - discount * p_pi.T, start) * live
    weights = visits[:, None, None] * policy[:, :, None] * mdp.transition
    return weights / weights.sum()


def canonicalise(mdp: MDP, reward: Array, value: Array, expected: bool) -> Array:
    """R + gamma V(s') - V(s). `expected=True` also averages over s' (STARC); False is what
    the goal-misgen code computes on sampled transitions."""
    shaped = reward + mdp.gamma * value[None, None, :] - value[:, None, None]
    if expected:
        return as_reward(np.einsum("sat,sat->sa", mdp.transition, shaped), mdp)
    return shaped


def unit_distance(x: Array, y: Array, weights: Array) -> float:
    """RMS distance between the two signals after each is scaled to unit RMS."""
    xn = x / np.sqrt((weights * x**2).sum())
    yn = y / np.sqrt((weights * y**2).sum())
    return float(np.sqrt((weights * (xn - yn) ** 2).sum()))


def pirc(
    mdp: MDP,
    implied: Array,
    trusted: Array,
    expected: bool = True,
    implied_terminal_value: float = 0.0,
) -> float:
    """PIRC distance between an implied reward on (s, a) and the MDP's true reward."""
    implied_reward = as_reward(implied, mdp)
    v_true = policy_value(mdp, trusted, mdp.reward)
    v_implied = policy_value(mdp, trusted, implied_reward, implied_terminal_value)
    return unit_distance(
        canonicalise(mdp, implied_reward, v_implied, expected),
        canonicalise(mdp, mdp.reward, v_true, expected),
        occupancy(mdp, trusted),
    )


def entropy(policy: Array) -> Array:
    return -(policy * np.log(policy)).sum(axis=1)


def implied_rewards(policy: Array) -> dict[str, Array]:
    """Four readings of a policy's log-probabilities. They differ only in what is subtracted
    in each state: nothing, the policy-weighted mean, the plain mean, or the maximum."""
    logp = np.log(policy)
    return {
        "log pi": logp,
        "log pi + H": logp + entropy(policy)[:, None],
        "log pi - mean": logp - logp.mean(axis=1, keepdims=True),
        "log pi - max": logp - logp.max(axis=1, keepdims=True),
    }


def optimal_q(mdp: MDP, sweeps: int = 20000) -> Array:
    """Q-values of the ordinary (no entropy) optimal policy, by value iteration."""
    value = np.zeros(mdp.n_states)
    for _ in range(sweeps):
        new = q_from_value(mdp, mdp.reward, value).max(axis=1)
        new[mdp.terminal] = 0.0
        if np.abs(new - value).max() < 1e-13:
            break
        value = new
    return q_from_value(mdp, mdp.reward, value)


def random_mdp(rng: np.random.Generator, n_states: int, n_actions: int, gamma: float, deterministic: bool) -> MDP:
    if deterministic:
        transition = np.zeros((n_states, n_actions, n_states))
        nxt = rng.integers(0, n_states, size=(n_states, n_actions))
        transition[np.arange(n_states)[:, None], np.arange(n_actions)[None, :], nxt] = 1.0
    else:
        transition = rng.dirichlet(np.full(n_states, 0.3), size=(n_states, n_actions))
    reward = rng.normal(size=(n_states, n_actions, n_states))
    return MDP(transition, reward, gamma, np.zeros(n_states, dtype=bool))


def random_policy(rng: np.random.Generator, mdp: MDP, sharpness: float = 1.5) -> Array:
    logits = sharpness * rng.normal(size=(mdp.n_states, mdp.n_actions))
    return np.exp(logits - logsumexp(logits)[:, None])


def corridor(length: int, gamma: float) -> MDP:
    """Episodic corridor: 'forward' walks to a +10 goal, 'stay' waits, 'quit' ends the episode with 0."""
    n_states, forward, stay, quit_ = length + 1, 0, 1, 2
    end = length
    transition = np.zeros((n_states, 3, n_states))
    reward = np.zeros((n_states, 3, n_states))
    for s in range(length):
        transition[s, forward, s + 1] = 1.0
        transition[s, stay, s] = 1.0
        transition[s, quit_, end] = 1.0
    reward[length - 1, forward, end] = 10.0
    transition[end, :, end] = 1.0
    terminal = np.zeros(n_states, dtype=bool)
    terminal[end] = True
    return MDP(transition, reward, gamma, terminal)


def mean_return(mdp: MDP, policy: Array) -> float:
    live = ~mdp.terminal
    return float(policy_value(mdp, policy, mdp.reward)[live].mean())


def floor_logprobs(policy: Array, floor: float) -> Array:
    """Stop any log-probability falling below `floor` (then renormalise): a policy that acts
    almost identically but does not emit astronomically small probabilities."""
    clipped = np.maximum(policy, np.exp(floor))
    return clipped / clipped.sum(axis=1, keepdims=True)


def check_soft_theorem(rng: np.random.Generator) -> None:
    print("\n[1] Soft-optimal policy => log pi is the true reward up to shaping (paper Theorem 3.1)")
    for deterministic in (False, True):
        mdp = random_mdp(rng, 12, 4, 0.9, deterministic)
        policy, _ = soft_optimal(mdp, alpha=0.7)
        for name, trusted in (("uniform", uniform_policy(mdp)), ("random", random_policy(rng, mdp))):
            exact = pirc(mdp, np.log(policy), trusted, expected=True)
            sampled = pirc(mdp, np.log(policy), trusted, expected=False)
            kind = "deterministic" if deterministic else "stochastic"
            print(f"    {kind:13s} dynamics, trusted={name:7s}: averaged over s' {exact:.2e} | on sampled s' {sampled:.3f}")


def check_every_policy_is_soft_optimal(rng: np.random.Generator) -> None:
    print("\n[2] Any policy is soft-optimal for its own log-probabilities, with soft value 0, in any dynamics")
    worst_policy, worst_value = 0.0, 0.0
    for _ in range(5):
        mdp = random_mdp(rng, 10, 3, 0.95, deterministic=False)
        policy = random_policy(rng, mdp)
        own = MDP(mdp.transition, as_reward(np.log(policy), mdp), mdp.gamma, mdp.terminal)
        recovered, value = soft_optimal(own, alpha=1.0, start=rng.normal(scale=5.0, size=mdp.n_states))
        worst_policy = max(worst_policy, float(np.abs(recovered - policy).max()))
        worst_value = max(worst_value, float(np.abs(value).max()))
    print(f"    max |recovered policy - policy| = {worst_policy:.1e}, max |soft value| = {worst_value:.1e}")


def check_which_reading_is_exact(rng: np.random.Generator) -> None:
    print("\n[3] Which reading of log pi is exact depends on how the policy was regularised")
    mdp = random_mdp(rng, 12, 4, 0.9, deterministic=False)
    uniform = uniform_policy(mdp)
    q_uniform = q_from_value(mdp, mdp.reward, policy_value(mdp, uniform, mdp.reward))
    q_star = optimal_q(mdp)
    subjects = {
        "entropy in the return (MaxEnt)": soft_optimal(mdp, alpha=0.7)[0],
        "per-state entropy bonus (PPO-like)": boltzmann_fixed_point(mdp, beta=0.7),
        "softmax of the uniform policy's Q": np.exp(q_uniform / 0.7 - logsumexp(q_uniform / 0.7)[:, None]),
        "softmax of the optimal Q (Boltzmann)": np.exp(q_star / 0.7 - logsumexp(q_star / 0.7)[:, None]),
    }
    readings = list(implied_rewards(uniform))
    print("    subject policy".ljust(42) + "".join(r.ljust(16) for r in readings))
    for name, policy in subjects.items():
        row = [pirc(mdp, implied, uniform) for implied in implied_rewards(policy).values()]
        print(f"    {name:38s}" + "".join(f"{d:<16.3f}" for d in row))
    hard = boltzmann_fixed_point(mdp, beta=0.7)
    adv = q_from_value(mdp, mdp.reward, policy_value(mdp, hard, mdp.reward)) - policy_value(mdp, hard, mdp.reward)[:, None]
    print(f"    PPO-like policy: max |A/beta - (log pi + H)| = {np.abs(adv / 0.7 - implied_rewards(hard)['log pi + H']).max():.1e}"
          f", max |sum_a A| = {np.abs(adv.sum(axis=1)).max():.2f}, max |sum_a pi A| = {np.abs((hard * adv).sum(axis=1)).max():.1e}")


def check_distance_is_not_performance() -> None:
    print("\n[4] Distance and performance come apart (episodic corridor, uniform trusted policy)")
    print("    length  policy                                   distance   mean return")
    for length in (5, 20, 80):
        mdp = corridor(length, gamma=0.99)
        uniform = uniform_policy(mdp)
        sharp, _ = soft_optimal(mdp, alpha=0.1)
        floored = floor_logprobs(sharp, floor=-8.0)
        hot, _ = soft_optimal(mdp, alpha=100.0)
        rows = (
            ("soft-optimal, temperature 0.1", sharp),
            ("same, log-probs floored at -8", floored),
            ("soft-optimal, temperature 100", hot),
        )
        for name, policy in rows:
            print(f"    {length:<7d} {name:40s} {pirc(mdp, np.log(policy), uniform):<10.3f} {mean_return(mdp, policy):.2f}")
        print(f"    {length:<7d} {'(uniform policy, for reference)':40s} {'undefined':10s} {mean_return(mdp, uniform):.2f}")


def check_offsets_and_terminal_value() -> None:
    print("\n[5] The right end-of-episode value depends on the regulariser (corridor, exactly optimal subjects, alpha=0.1)")
    print("    length  subject regularised by            end value 0   -log(n)/(1-g)   +log(n)/(1-g)   log pi + log n, end value 0")
    for length in (5, 20, 80):
        mdp = corridor(length, gamma=0.99)
        uniform = uniform_policy(mdp)
        absorbing = np.log(mdp.n_actions) / (1.0 - mdp.gamma)
        subjects = (
            ("entropy bonus in the return", soft_optimal(mdp, alpha=0.1)[0]),
            ("penalty for leaving uniform", soft_optimal(mdp, alpha=0.1, prior=uniform)[0]),
        )
        for name, policy in subjects:
            logp = np.log(policy)
            cells = (
                pirc(mdp, logp, uniform),
                pirc(mdp, logp, uniform, implied_terminal_value=-absorbing),
                pirc(mdp, logp, uniform, implied_terminal_value=absorbing),
                pirc(mdp, logp + np.log(mdp.n_actions), uniform),
            )
            print(f"    {length:<7d} {name:33s} {cells[0]:<13.3f} {cells[1]:<15.3f} {cells[2]:<15.3f} {cells[3]:.3f}")


@dataclass(frozen=True)
class Tree:
    """A complete token tree: node 0 is the prompt, each internal node has `branching` children."""

    branching: int
    depth: int

    @property
    def n_internal(self) -> int:
        return (self.branching**self.depth - 1) // (self.branching - 1)

    @property
    def n_nodes(self) -> int:
        return (self.branching ** (self.depth + 1) - 1) // (self.branching - 1)

    def child(self, node: int, action: int) -> int:
        return node * self.branching + 1 + action

    def path_to(self, leaf: int) -> list[tuple[int, int]]:
        steps: list[tuple[int, int]] = []
        node = leaf
        while node > 0:
            parent = (node - 1) // self.branching
            steps.append((parent, node - parent * self.branching - 1))
            node = parent
        return steps[::-1]


def tree_policy(tree: Tree, utility: Array, reference: Array, beta: float) -> tuple[Array, float]:
    """KL-regularised optimum on a tree: pi(x) ~ reference(x) * exp(utility(x)/beta). Returns
    (policy over internal nodes, soft value of the root)."""
    value = np.zeros(tree.n_nodes)
    value[tree.n_internal :] = utility
    policy = np.zeros_like(reference)
    for node in range(tree.n_internal - 1, -1, -1):
        kids = np.array([value[tree.child(node, a)] for a in range(tree.branching)])
        value[node] = beta * (np.log((reference[node] * np.exp((kids - kids.max()) / beta)).sum())) + kids.max()
        policy[node] = reference[node] * np.exp((kids - value[node]) / beta)
    return policy, float(value[0])


def leaf_logprob(tree: Tree, policy: Array) -> Array:
    return np.array([sum(np.log(policy[s, a]) for s, a in tree.path_to(leaf)) for leaf in range(tree.n_internal, tree.n_nodes)])


def tree_as_mdp(tree: Tree, utility: Array) -> MDP:
    transition = np.zeros((tree.n_nodes, tree.branching, tree.n_nodes))
    reward = np.zeros_like(transition)
    for node in range(tree.n_internal):
        for a in range(tree.branching):
            kid = tree.child(node, a)
            transition[node, a, kid] = 1.0
            if kid >= tree.n_internal:
                reward[node, a, kid] = utility[kid - tree.n_internal]
    terminal = np.arange(tree.n_nodes) >= tree.n_internal
    transition[terminal, :, :] = 0.0
    transition[terminal, :, terminal] = 1.0
    return MDP(transition, reward, 1.0, terminal)


def weighted_corr(x: Array, y: Array, weights: Array) -> float:
    xc, yc = x - (weights * x).sum(), y - (weights * y).sum()
    return float((weights * xc * yc).sum() / np.sqrt((weights * xc**2).sum() * (weights * yc**2).sum()))


def tree_occupancy(tree: Tree, mdp: MDP, policy: Array) -> Array:
    """Visitation weights over tree transitions when starting at the prompt (node 0)."""
    reach = np.zeros(tree.n_nodes)
    reach[0] = 1.0
    weights = np.zeros_like(mdp.transition)
    for node in range(tree.n_internal):
        for a in range(tree.branching):
            kid = tree.child(node, a)
            reach[kid] = reach[node] * policy[node, a]
            weights[node, a, kid] = reach[kid]
    return weights / weights.sum()


def check_token_tree(rng: np.random.Generator) -> None:
    print("\n[6] Deterministic token tree (the LLM case): branching 3, depth 5, 243 complete sequences")
    tree = Tree(branching=3, depth=5)
    n_leaves = tree.n_nodes - tree.n_internal
    utility = rng.normal(size=n_leaves)
    other = rng.normal(size=n_leaves)
    reference = rng.dirichlet(np.ones(tree.branching), size=tree.n_internal)
    beta = 0.5
    policy, root_value = tree_policy(tree, utility, reference, beta)
    ratio = leaf_logprob(tree, policy) - leaf_logprob(tree, reference)
    print(f"    (a) sum of token log-ratios = (U - V(prompt))/beta on every sequence: max error {np.abs(ratio - (utility - root_value) / beta).max():.1e}")

    mdp = tree_as_mdp(tree, utility)
    full_policy = np.full((tree.n_nodes, tree.branching), 1.0 / tree.branching)
    full_reference = full_policy.copy()
    full_policy[: tree.n_internal], full_reference[: tree.n_internal] = policy, reference
    token_ratio = np.log(full_policy) - np.log(full_reference)
    trusted = full_reference

    def token_level(implied: Array) -> float:
        implied_reward = as_reward(implied, mdp)
        v_i = policy_value(mdp, trusted, implied_reward)
        v_t = policy_value(mdp, trusted, mdp.reward)
        return unit_distance(
            canonicalise(mdp, implied_reward, v_i, True),
            canonicalise(mdp, mdp.reward, v_t, True),
            tree_occupancy(tree, mdp, trusted),
        )

    print(f"    (b) token-level PIRC, implied reward log(pi/pi_ref): {token_level(token_ratio):.1e}"
          f" | implied reward log pi alone: {token_level(np.log(full_policy)):.3f}")

    p_ref = np.exp(leaf_logprob(tree, reference))
    print("    (c) policy optimises a mixture w*U + (1-w)*U_other; distance to U measured two ways")
    print("        w      corr(U, mixture)   sequence-level sqrt(2(1-corr))   token-level PIRC   E_pi[U] from 200k reference samples, importance-weighted")
    for w in (1.0, 0.8, 0.5, 0.0):
        mixed = w * utility + (1.0 - w) * other
        pol, _ = tree_policy(tree, mixed, reference, beta)
        full = full_reference.copy()
        full[: tree.n_internal] = pol
        seq_ratio = leaf_logprob(tree, pol) - leaf_logprob(tree, reference)
        rho = weighted_corr(seq_ratio, utility, p_ref)
        truth = float((np.exp(leaf_logprob(tree, pol)) * utility).sum())
        drawn = rng.choice(n_leaves, size=200_000, p=p_ref)
        is_estimate = float((np.exp(seq_ratio[drawn]) * utility[drawn]).mean())
        print(f"        {w:<6.1f} {weighted_corr(mixed, utility, p_ref):<18.3f} {np.sqrt(max(0.0, 2 * (1 - rho))):<32.3f}"
              f" {token_level(np.log(full) - np.log(full_reference)):<18.3f} {is_estimate:.3f} (exact {truth:.3f})")

    shaped = mdp.reward.copy()
    potential = rng.normal(size=tree.n_nodes)
    potential[tree.n_internal :] = 0.0
    shaped += potential[None, None, :] - potential[:, None, None]
    shaped *= mdp.transition > 0
    v_a = policy_value(mdp, trusted, mdp.reward)
    v_b = policy_value(mdp, trusted, shaped)
    same = unit_distance(canonicalise(mdp, shaped, v_b, True), canonicalise(mdp, mdp.reward, v_a, True), tree_occupancy(tree, mdp, trusted))
    leaf_gap = [sum(shaped[s, a, tree.child(s, a)] for s, a in tree.path_to(leaf)) - utility[leaf - tree.n_internal] for leaf in range(tree.n_internal, tree.n_nodes)]
    print(f"    (d) two token-level rewards with the same per-sequence totals up to a constant: distance {same:.1e}"
          f" (spread of the totals' difference {np.ptp(leaf_gap):.1e})")


def random_episodic_mdp(rng: np.random.Generator, n_states: int, n_actions: int, gamma: float, stop: float) -> MDP:
    """Stochastic MDP in which every step ends the episode with probability about `stop`."""
    live_part = rng.dirichlet(np.full(n_states, 0.3), size=(n_states + 1, n_actions))
    stop_prob = rng.uniform(0.5 * stop, 1.5 * stop, size=(n_states + 1, n_actions, 1))
    transition = np.concatenate([live_part * (1.0 - stop_prob), stop_prob], axis=2)
    transition[n_states] = 0.0
    transition[n_states, :, n_states] = 1.0
    reward = rng.normal(size=(n_states + 1, n_actions, n_states + 1))
    reward[n_states] = 0.0
    terminal = np.zeros(n_states + 1, dtype=bool)
    terminal[n_states] = True
    return MDP(transition, reward, gamma, terminal)


def sampled_episode_sums(
    rng: np.random.Generator, mdp: MDP, trusted: Array, start: Array, episodes: int, signals: list[Array], discount: float
) -> tuple[Array, NDArray[np.int64]]:
    """Sample trusted-policy episodes. Returns the discounted sum of each (S, A, S) signal along
    each episode, shape (len(signals), episodes), and each episode's start state."""
    first = rng.choice(mdp.n_states, size=episodes, p=start)
    state = first.copy()
    alive = np.ones(episodes, dtype=bool)
    sums, scale = np.zeros((len(signals), episodes)), 1.0
    cum_policy = trusted.cumsum(axis=1)
    cum_transition = mdp.transition.cumsum(axis=2)
    while alive.any():
        idx = np.flatnonzero(alive)
        s = state[idx]
        a = (rng.random(len(idx))[:, None] > cum_policy[s]).sum(axis=1).clip(max=mdp.n_actions - 1)
        nxt = (rng.random(len(idx))[:, None] > cum_transition[s, a]).sum(axis=1).clip(max=mdp.n_states - 1)
        for k, signal in enumerate(signals):
            sums[k, idx] += scale * signal[s, a, nxt]
        state[idx] = nxt
        alive[idx] = ~mdp.terminal[nxt]
        scale *= discount
    return sums, first


def correlation_distance(x: Array, y: Array) -> float:
    return float(np.sqrt(2.0 - 2.0 * np.corrcoef(x, y)[0, 1]))


def centre_per_start(sums: Array, first: NDArray[np.int64]) -> Array:
    """Subtract, from each episode's sums, the sample mean over episodes with the same start state."""
    centred = sums.copy()
    for s0 in np.unique(first):
        centred[:, first == s0] -= centred[:, first == s0].mean(axis=1, keepdims=True)
    return centred


def check_distance_is_a_return_correlation(rng: np.random.Generator) -> None:
    print("\n[7] PIRC distance as a correlation over 400k sampled trusted episodes (three start states)")
    print("    gamma   returns, centred per start state   exact, weights gamma^2t   |   sums of one-step surprises   exact, weight 1 (the code)")
    for gamma in (1.0, 0.99, 0.9):
        mdp = random_episodic_mdp(rng, n_states=8, n_actions=3, gamma=gamma, stop=0.08)
        trusted = uniform_policy(mdp)
        implied_reward = as_reward(np.log(random_policy(rng, mdp)), mdp)
        start = np.zeros(mdp.n_states)
        start[:3] = (0.5, 0.3, 0.2)
        c_implied = canonicalise(mdp, implied_reward, policy_value(mdp, trusted, implied_reward), expected=False)
        c_true = canonicalise(mdp, mdp.reward, policy_value(mdp, trusted, mdp.reward), expected=False)
        exact_sq, exact_one = (unit_distance(c_implied, c_true, occupancy(mdp, trusted, start, d)) for d in (gamma**2, 1.0))
        returns = centre_per_start(*sampled_episode_sums(rng, mdp, trusted, start, 400_000, [mdp.reward, implied_reward], gamma))
        surprises, _ = sampled_episode_sums(rng, mdp, trusted, start, 400_000, [c_true, c_implied], 1.0)
        print(f"    {gamma:<7.2f} {correlation_distance(returns[0], returns[1]):<34.3f} {exact_sq:<25.3f} |   "
              f"{correlation_distance(surprises[0], surprises[1]):<28.3f} {exact_one:.3f}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    np.set_printoptions(precision=3, suppress=True)
    check_soft_theorem(rng)
    check_every_policy_is_soft_optimal(rng)
    check_which_reading_is_exact(rng)
    check_distance_is_not_performance()
    check_offsets_and_terminal_value()
    check_token_tree(rng)
    check_distance_is_a_return_correlation(rng)


if __name__ == "__main__":
    main()
