import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import einops


class TabularPolicy:
    def __init__(self, name, pi, Q=None, V=None):
        self.name = name
        self.Q = Q
        self.V = V
        self.pi = pi
        # This is in case we have a full zero, we adjust policy.
        flt = (self.pi == 0).any(dim=-1)
        self.pi[flt] = (self.pi[flt] * 10).softmax(dim=-1)
        assert (self.pi.sum(dim=-1).round(decimals=3) == 1).all(), "pi is not a probability distribution along final dim"
        self.log_pi = self.pi.log()


# Define a tabular MDP
class TabularMDP:
    def __init__(self, n_states, n_actions, transition_prob, reward_vector, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = transition_prob  # Shape: (n_states, n_actions, n_states)
        self.reward_vector = reward_vector  # Shape: (n_states,)
        self.gamma = gamma
        self.soft_opt = self.soft_q_value_iteration()
        self.hard_opt = self.q_value_iteration()

        q_uni = torch.zeros((n_states, n_actions))
        unipi = q_uni.softmax(dim=-1)
        self.uniform = TabularPolicy("Uniform", unipi, q_uni, q_uni.logsumexp(dim=-1))

    def q_value_iteration(self, n_iterations=1000):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions
        Q = torch.zeros(n_states, n_actions)
        V = torch.zeros(n_states)

        for i in range(n_iterations):
            old_Q = Q
            V = Q.max(dim=1).values
            Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(
                1)

            if (Q - old_Q).abs().max() < 1e-5:
                print(f'Q-value iteration converged in {i} iterations')
                pi = torch.nn.functional.one_hot(Q.argmax(dim=1),num_classes=n_actions).float()
                return TabularPolicy("Hard", pi, Q, V)
        print(f"Q-value iteration did not converge in {i} iterations")
        return None

    def soft_q_value_iteration(self, n_iterations=1000, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        Q = torch.zeros(n_states, n_actions)

        for i in range(n_iterations):
            old_Q = Q
            V = Q.logsumexp(dim=-1)
            Q = einops.einsum(T, gamma * V, 'states actions next_states, next_states -> states actions') + R.unsqueeze(
                1)

            if (Q - old_Q).abs().max() < 1e-4:
                if print_message:
                    print(f'soft value iteration converged in {i} iterations')
                pi = Q.softmax(dim=1)
                return TabularPolicy("Soft", pi, Q, V)
        print('soft value iteration did not converge after', n_iterations, 'iterations')
        return None

    # def compute_q_values(self, policy):
    #     """
    #     Computes Q-values for a given policy.
    #     policy: Shape (n_states, n_actions), probability distribution over actions.
    #     """
    #     P_pi = torch.einsum("sas,sa->ss", self.transition_prob, policy)  # Transition matrix under policy
    #     r_pi = torch.einsum("sa,s->s", policy, self.reward_vector)  # Expected reward under policy
    #
    #     # Solve for Q-values (Bellman equation)
    #     I = torch.eye(self.n_states)
    #     Q = torch.linalg.solve(I - self.gamma * P_pi, r_pi)
    #     return Q
    #
    # def soft_optimal_policy(self, beta):
    #     """
    #     Computes the soft-optimal policy given a temperature parameter beta.
    #     """
    #     Q = self.compute_q_values(torch.ones((self.n_states, self.n_actions)) / self.n_actions)
    #     soft_policy = F.softmax(beta * Q.unsqueeze(-1), dim=1)  # Softmax over actions
    #     return soft_policy


# Gradient descent on reward vector
def gradient_descent_on_rewards(mdp, policy, lr=0.01, steps=100):
    reward_vector = mdp.reward_vector.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([reward_vector], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        q_values = mdp.compute_q_values(policy)
        loss = -torch.sum(q_values)  # Example loss (maximize Q-values)
        loss.backward()
        optimizer.step()

    return reward_vector


# Define an epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon):
    """Generates an epsilon-greedy policy given Q-values."""
    n_states, n_actions = q_values.shape
    greedy_policy = torch.zeros((n_states, n_actions))
    greedy_actions = q_values.argmax(dim=1)
    greedy_policy[torch.arange(n_states), greedy_actions] = 1 - epsilon
    greedy_policy += epsilon / n_actions
    return greedy_policy


class AscenderLong(TabularMDP):
    def __init__(self, n_states, gamma=0.99):
        assert n_states % 2 == 0, (
            "Ascender requires a central starting state with an equal number of states to the left"
            " and right, plus an infinite terminal state. Therefore n_states must be even.")
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        for i in range(n_states - 3):
            T[i + 1, 1, i + 2] = 1
            T[i + 1, 0, i] = 1
        T[(0, -1, -2), :, -1] = 1

        R = torch.zeros(n_states)
        R[-2] = 10
        R[0] = -10

        go_left = torch.zeros((n_states, n_actions))
        go_left[:,0] = 1
        self.go_left = TabularPolicy("Go Left", go_left)

        super().__init__(n_states, n_actions, T, R, gamma)


class OneStep(TabularMDP):
    def __init__(self, gamma=0.99):
        n_states = 4
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[:2, 0, 2] = 1
        T[:2, 1, 1] = 1
        R = torch.zeros(n_states)
        R[2] = 1
        consistent_pi = torch.zeros(n_states, n_actions)
        consistent_pi[:2] = torch.FloatTensor([0.2, 0.8])
        consistent_pi[2:] = 0.5

        inconsistent_pi = consistent_pi.clone()
        inconsistent_pi[1] = torch.FloatTensor([0.8, 0.2])
        self.consistent = TabularPolicy("Consistent", consistent_pi)
        self.inconsistent = TabularPolicy("Inconsistent", inconsistent_pi)

        super().__init__(n_states, n_actions, T, R, gamma)


def titus_meg(pi, T, n_iterations=10000, lr=1e-1, print_losses=False):
    n_states, n_actions, _ = T.shape
    g = torch.randn(n_states, requires_grad=True)
    log_pi = pi.log()
    log_pi.requires_grad = False
    T.requires_grad = False

    optimizer = torch.optim.Adam([g], lr=lr)
    for i in range(n_iterations):
        old_g = g.detach().clone()
        q_est = einops.einsum(T, g, "s a ns, ns -> s a")
        v = q_est.logsumexp(dim=-1).unsqueeze(-1)
        meg_proxy = ((log_pi + v - q_est) ** 2).mean()

        # Actually calculating meg here. can use either as loss, but meg_proxy converges faster.
        log_pi_soft_star = q_est.log_softmax(dim=-1)

        meg = ((pi * log_pi_soft_star).sum(dim=-1) - np.log(1/n_actions)).sum()

        loss = meg_proxy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Tmeg:{meg_proxy:.4f}\tMeg:{meg:.4f}")
        if (g - old_g).abs().max() < 1e-5:
            print(f'Titus Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg
    print(f'Titus Meg did not converge in {i} iterations')
    return meg


def main():
    env = OneStep()
    titus_meg(env.soft_opt.pi, env.T)
    titus_meg(env.hard_opt.pi, env.T)
    titus_meg(env.uniform.pi, env.T)
    titus_meg(env.consistent.pi, env.T)
    titus_meg(env.inconsistent.pi, env.T)


    mdp = AscenderLong(n_states=6)

    titus_meg(mdp.soft_opt.pi, mdp.T)
    titus_meg(mdp.hard_opt.pi, mdp.T)
    titus_meg(mdp.uniform.pi, mdp.T)
    titus_meg(mdp.go_left.pi, mdp.T)





    # nq = einops.einsum(mdp.soft_opt.Q, mdp.transition_prob, 'states actions, states actions next_states -> next_states')
    # plt.scatter(nq.detach().cpu().numpy(), g.detach().cpu().numpy())
    # plt.show()
    print("done")


if __name__ == "__main__":
    main()
