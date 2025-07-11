import numpy as np
import einops
import torch



def state_occupancy(pi: torch.tensor, T: torch.tensor, gamma: float, mu: torch.tensor, max_iterations=1000, d=None,
                    device="cpu"):
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
        d = torch.rand(num_states, device=device)

    for iter in range(max_iterations):
        old_d = d.detach().clone()
        try:
            d = mu + gamma * einops.einsum(pi, T, d,
                                           'prev_states actions, prev_states actions states, prev_states -> states')
        except Exception as e:
            raise e
        if torch.allclose(d, old_d):
            return d

    print(f"occupancy measure failed to converge after {iter} iterations")
    return None


def state_action_occupancy(pi: torch.tensor, T: torch.tensor, gamma: float, mu: torch.tensor = None,
                           max_iterations=1000,
                           d=None, device="cpu"):
    """
    Computes the discounted state-action occupancy under policy pi

    Args:
      pi: Policy, with shape (num_states, num_actions).
      T: Transition probability table, with shape (num_states, num_actions, num_states).
      mu: Initial state distribution, with shape (num_states,).

    Returns:
      d: state action occupancy (num_states, num_actions)
    """

    d = state_occupancy(pi, T, gamma, mu, max_iterations, d=d, device=device)
    try:
        return einops.einsum(d, pi, 'states, states actions -> states actions')
    except RuntimeError as e:
        raise e

def policy_evaluation(pi: torch.tensor, U: torch.tensor, T: torch.tensor, gamma: float, mu: torch.tensor = None,
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


def soft_value_iteration(U: torch.tensor, T: torch.tensor, beta=1.0, gamma: float = 0.9, Q=None,
                         max_iterations=10000, device="cpu", atol=0.01):
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
        Q = torch.rand(num_states, num_actions, device=device)

    for iter in range(max_iterations):
        old_Q = Q.detach().clone()
        Q = Q - Q.max()  # for numerical stability
        Q = U.unsqueeze(dim=-1) + gamma * einops.einsum(T, (1 / beta) * (beta * Q).logsumexp(dim=-1),
                                                        'states actions next_states, next_states -> states actions')

        if torch.allclose(Q, old_Q, atol=atol):
            pi = (beta * Q).softmax(dim=-1)
            return Q, pi

    print(f"soft value iteration failed to converge after {iter} iterations")
    pi = (beta * Q).softmax(dim=-1)
    return Q, pi

def soft_value_iteration_sa_rew(U: torch.tensor, T: torch.tensor, beta=1.0, gamma: float = 0.9, Q=None,
                         max_iterations=10000, device="cpu", atol=0.01):
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
        Q = torch.zeros(num_states, num_actions, device=device)

    for iter in range(max_iterations):
        old_Q = Q.detach().clone()
        Q = Q - Q.max()  # for numerical stability
        Q = U + gamma * einops.einsum(T, (1 / beta) * (beta * Q).logsumexp(dim=-1),
                                                        'states actions next_states, next_states -> states actions')

        if torch.allclose(Q, old_Q, atol=atol):
            pi = (beta * Q).softmax(dim=-1)
            return Q, pi

    print(f"soft value iteration failed to converge after {iter} iterations")
    pi = (beta * Q).softmax(dim=-1)
    return Q, pi

def unknown_utility_meg(pi: torch.tensor, T: torch.tensor, gamma: float = 0.9, mu: torch.tensor = None,
                        max_iterations=10000,
                        lr=0.1, device="cpu", atol=0.01):
    """
    Computes MEG of a policy with respect to the set of reward functions linear in states.

    Args:
        pi (torch.tensor): Policy, with shape (num_states, num_actions).
        T (torch.tensor): Transition probability table, with shape (num_states, num_actions, num_states).
        mu (torch.tensor, optional): Initial state distribution with shape (num_states,). Defaults to uniform.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Returns:
        float: MEG_U(pi), the goal-directedness of pi.

    """
    pi = pi.to(device=device)
    num_states, num_actions = pi.shape
    assert T.shape == (num_states, num_actions,
                       num_states), "Transition probability table must have shape (num_states, num_actions, num_states)"
    if mu is None:
        mu = torch.ones(num_states, device=device) / num_states
    else:
        assert mu.shape == (num_states,), "Initial state distribution must have shape (num_states,)"
    assert 0 <= gamma < 1, "Discount factor must be >=0 and <1"
    assert torch.allclose(pi.sum(dim=-1), torch.tensor(1.)), "Policy must sum to 1 across actions"
    assert torch.isclose(mu.sum(), torch.tensor(1.)), "Initial state distribution must sum to 1"

    Q = torch.rand(num_states, num_actions, device=device)
    beta = 1.0
    U = torch.rand(num_states, device=device)
    d_pi = torch.rand(num_states, device=device)

    eps = 1e-9

    for iter in range(max_iterations):

        old_U = U.detach().clone()
        Q, pi_soft = soft_value_iteration(U, T, beta, gamma, Q, device=device, atol=atol)

        d_pi = state_occupancy(pi, T, gamma, mu, d=d_pi, device=device)
        d_pi_soft = state_occupancy(pi_soft, T, gamma, mu, d=d_pi, device=device)
        grad = einops.einsum(d_pi - d_pi_soft, U, 'states, states -> states')

        U += lr * grad

        if torch.allclose(U, old_U, atol=atol):
            da = state_action_occupancy(pi, T, gamma, mu, device=device)
            meg = einops.einsum(da, torch.log(pi_soft + eps) - np.log(1 / num_actions),
                                'states actions, states actions ->')
            return meg.detach().cpu()

    print(f"MEG failed to converge after {iter} iterations")
    da = state_action_occupancy(pi, T, gamma, mu, device=device)
    meg = einops.einsum(da, torch.log(pi_soft + eps) - np.log(1 / num_actions),
                        'states actions, states actions ->')
    return meg.detach().cpu()




# if __name__ == "__main__":
#     GAMMA = 0.9
#     env = MattGridworld()
#     megs = {}
#     for name, policy in env.policies.items():
#         meg = unknown_utility_meg(policy.pi, env.T, GAMMA, env.mu, device=env.device)
#         print(f"{name}\tMeg: {meg:.4f}")
#         megs[name] = meg
#
#     df = pd.DataFrame(megs)
#     print(df)
