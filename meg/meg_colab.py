import numpy as np
import einops
from scipy.special import logsumexp, softmax


def state_occupancy(pi: np.ndarray, T: np.ndarray, gamma: float, mu: np.ndarray, max_iterations=1000, d=None):
    """
    Computes the discounted state occupancy under policy pi

    Args:
      pi: Policy, with shape (num_states, num_actions).
      T: Transition probability table, with shape (num_states, num_actions, num_states).
      mu: Initial state distribution, with shape (num_states,).

    Returns:
      d: state occupancy (num_states,)
    """

    num_states, num_actions = T.shape[:2]

    if d is None:
        d = np.random.rand(num_states)

    for iter in range(max_iterations):
        old_d = d.copy()
        try:
            d = mu + gamma * einops.einsum(pi, T, d,
                                           'prev_states actions, prev_states actions states, prev_states -> states')
        except Exception as e:
            raise e
        if np.allclose(d, old_d):
            return d

    print(f"occupancy measure failed to converge after {iter} iterations")
    return None


def state_action_occupancy(pi: np.ndarray, T: np.ndarray, gamma: float, mu: np.ndarray = None, max_iterations=1000,
                           d=None):
    """
    Computes the discounted state-action occupancy under policy pi

    Args:
      pi: Policy, with shape (num_states, num_actions).
      T: Transition probability table, with shape (num_states, num_actions, num_states).
      mu: Initial state distribution, with shape (num_states,).

    Returns:
      d: state action occupancy (num_states, num_actions)
    """

    d = state_occupancy(pi, T, gamma, mu, max_iterations, d=d)
    return einops.einsum(d, pi, 'states, states actions -> states actions')


def policy_evaluation(pi: np.ndarray, U: np.ndarray, T: np.ndarray, gamma: float, mu: np.ndarray = None,
                      max_iterations=1000, d=None):
    """
    Computes the expected return of policy pi.

    Args:
      pi: Policy, with shape (num_states, num_actions).
      U: Utility function, with shape (num_states,)
      T: Transition probability table, with shape (num_states, num_actions, num_states).
      gamma: Discount factor.
      mu: Initial state distribution, with shape (num_states,).

    Returns:
      expected return of policy

    """

    d = state_occupancy(pi, T, gamma, mu, max_iterations, d=d)
    return d @ U


def value_iteration_step(U: np.ndarray, T: np.ndarray, gamma, V=None):
    """
    Performs a single step of value iteration.

    Args:
      U: Utility function, with shape (num_states,).
      T: Transition probability table, with shape (num_states, num_actions, num_states).
      gamma: Discount factor.
      V: Current value function, with shape (num_states,).

      Returns:
      V: Updated value function, with shape (num_states,).
      pi: Optimal policy for the updated value function, with shape (num_states, num_actions).
    """

    num_states, num_actions = T.shape[:2]

    if V is None:
        V = np.zeros(num_states)

    V = U + np.max(gamma * einops.einsum(T, V, 'states actions next_states, next_states -> states actions'), axis=-1)
    pi = np.zeros((num_states, num_actions))
    pi[np.arange(num_states), np.argmax(
        gamma * einops.einsum(T, V, 'states actions next_states, next_states -> states actions'), axis=-1)] = 1
    return V, pi


def soft_value_iteration(U: np.ndarray, T: np.ndarray, beta=1.0, gamma: float = 0.9, Q=None, max_iterations=10000, ):
    """
    Finds the soft Q-function and soft optimal policy.

    Args:
      U: Utility function, with shape (num_states,).
      T: Transition probability table, with shape (num_states, num_actions, num_states).
      beta: Inverse temperature parameter.
      gamma: Discount factor.
      max_iterations: Maximum number of iterations.
      Q: Initial soft Q matrix.

    Returns:
      Q: Soft Q-function, with shape (num_states, num_actions).
      pi: Soft optimal policy, with shape (num_states, num_actions).
    """
    num_states, num_actions = T.shape[:2]

    if Q is None:
        Q = np.random.rand(num_states, num_actions)

    for iter in range(max_iterations):
        old_Q = Q.copy()
        Q = Q - Q.max()  # for numerical stability
        Q = np.expand_dims(U, -1) + gamma * einops.einsum(T, (1 / beta) * logsumexp(beta * Q, axis=-1),
                                                          'states actions next_states, next_states -> states actions')

        if np.allclose(Q, old_Q, atol=0.01):
            pi = softmax(beta * Q, axis=-1)
            return Q, pi

    print(f"soft value iteration failed to converge after {iter} iterations")
    pi = softmax(beta * Q, axis=-1)
    return None, None


def unknown_utility_meg(pi: np.ndarray, T: np.ndarray, gamma: float = 0.9, mu: np.ndarray = None, max_iterations=10000,
                        lr=0.1):
    """
    Computes MEG of a policy with respect to the set of reward functions linear in states.

    Args:
        pi (np.ndarray): Policy, with shape (num_states, num_actions).
        T (np.ndarray): Transition probability table, with shape (num_states, num_actions, num_states).
        mu (np.ndarray, optional): Initial state distribution with shape (num_states,). Defaults to uniform.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Returns:
        float: MEG_U(pi), the goal-directedness of pi.

    """
    num_states, num_actions = pi.shape
    assert T.shape == (num_states, num_actions,
                       num_states), "Transition probability table must have shape (num_states, num_actions, num_states)"
    if mu is None:
        mu = np.ones(num_states) / num_states
    else:
        assert mu.shape == (num_states,), "Initial state distribution must have shape (num_states,)"
    assert 0 <= gamma < 1, "Discount factor must be >=0 and <1"
    assert np.allclose(pi.sum(axis=-1), 1), "Policy must sum to 1 across actions"
    assert np.isclose(mu.sum(), 1), "Initial state distribution must sum to 1"

    Q = np.random.rand(num_states, num_actions)
    beta = 1.0
    U = np.random.rand(num_states)
    d_pi = np.random.rand(num_states)

    eps = 1e-9

    for iter in range(max_iterations):

        old_U = U.copy()
        Q, pi_soft = soft_value_iteration(U, T, beta, gamma, Q)

        d_pi = state_occupancy(pi, T, gamma, mu, d=d_pi)
        d_pi_soft = state_occupancy(pi_soft, T, gamma, mu, d=d_pi)
        grad = einops.einsum(d_pi - d_pi_soft, U, 'states, states -> states')

        U += lr * grad

        if np.allclose(U, old_U, atol=0.01):
            da = state_action_occupancy(pi, T, gamma, mu)
            meg = einops.einsum(da, np.log(pi_soft + eps) - np.log(1 / num_actions),
                                'states actions, states actions ->')
            return meg

    print(f"MEG failed to converge after {iter} iterations")


def unknown_utility_meg2(pi: np.ndarray, T: np.ndarray, gamma: float = 0.9, mu: np.ndarray = None, max_iterations=10000,
                         lr=0.1):
    """
    Computes MEG of a policy with respect to the set of reward functions linear in states.

    Args:
        pi (np.ndarray): Policy, with shape (num_states, num_actions).
        T (np.ndarray): Transition probability table, with shape (num_states, num_actions, num_states).
        mu (np.ndarray, optional): Initial state distribution with shape (num_states,). Defaults to uniform.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Returns:
        float: MEG_U(pi), the goal-directedness of pi.

    """
    num_states, num_actions = pi.shape
    assert T.shape == (num_states, num_actions,
                       num_states), "Transition probability table must have shape (num_states, num_actions, num_states)"
    if mu is None:
        mu = np.ones(num_states) / num_states
    else:
        assert mu.shape == (num_states,), "Initial state distribution must have shape (num_states,)"
    assert 0 <= gamma < 1, "Discount factor must be >=0 and <1"
    assert np.allclose(pi.sum(axis=-1), 1), "Policy must sum to 1 across actions"
    assert np.isclose(mu.sum(), 1), "Initial state distribution must sum to 1"

    Q = np.random.rand(num_states, num_actions)
    beta = 1.0
    U = np.random.rand(num_states)
    EU_pi = policy_evaluation(pi, U, T, gamma, mu)
    pi_soft = np.ones((num_states, num_actions)) / num_actions
    d_pi = np.random.rand(num_states)
    current_meg = 0

    eps = 1e-9

    for iter in range(max_iterations):

        old_U = U.copy()
        old_meg = current_meg
        Q, pi_soft = soft_value_iteration(U, T, beta, gamma, Q)

        d_pi = state_occupancy(pi, T, gamma, mu, d=d_pi)
        d_pi_soft = state_occupancy(pi_soft, T, gamma, mu, d=d_pi)
        grad = einops.einsum(d_pi - d_pi_soft, U, 'states, states -> states')

        U += lr * grad

        if np.allclose(U, old_U, atol=0.01):
            da = state_action_occupancy(pi, T, gamma, mu)
            current_meg = einops.einsum(da, np.log(pi_soft + eps) - np.log(1 / num_actions),
                                        'states actions, states actions ->')
        return current_meg

    print(f"MEG failed to converge after {iter} iterations")


def matt_colab_env():
    N = 5
    num_states = N * N

    actions = ['up', 'down', 'left', 'right']
    NUM_ACTIONS = len(actions)

    T = np.zeros((num_states, NUM_ACTIONS, num_states))

    def state_index(x, y):
        return x * N + y

    def state_coords(s):
        return divmod(s, N)

    for x in range(N):
        for y in range(N):
            s = state_index(x, y)
            for a_idx, action in enumerate(actions):
                if action == 'up':
                    nx, ny = x - 1, y
                elif action == 'down':
                    nx, ny = x + 1, y
                elif action == 'left':
                    nx, ny = x, y - 1
                elif action == 'right':
                    nx, ny = x, y + 1

                if action == 'up':
                    ox, oy = x + 1, y
                elif action == 'down':
                    ox, oy = x - 1, y
                elif action == 'left':
                    ox, oy = x, y + 1
                elif action == 'right':
                    ox, oy = x, y - 1

                if 0 <= nx < N and 0 <= ny < N:
                    ns_intended = state_index(nx, ny)
                else:
                    ns_intended = s

                if 0 <= ox < N and 0 <= oy < N:
                    ns_opposite = state_index(ox, oy)
                else:
                    ns_opposite = s

                T[s, a_idx, ns_intended] += 0.9
                T[s, a_idx, ns_opposite] += 0.1

    ### Setting the discount rate gamma, utility function U, and initial state distribution mu

    gamma = 0.9
    U = np.random.rand(num_states)
    mu = np.zeros(num_states)
    mu[0] = 1
