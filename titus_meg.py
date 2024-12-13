import torch
import torch.nn.functional as F
import numpy as np
import einops


class TabularPolicy:
    def __init__(self, Q, V, pi):
        self.Q = Q
        self.V = V
        self.pi = pi
        self.log_pi = pi.log()


# Define a tabular MDP
class TabularMDP:
    def __init__(self, n_states, n_actions, transition_prob, reward_vector, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_prob = transition_prob  # Shape: (n_states, n_actions, n_states)
        self.reward_vector = reward_vector  # Shape: (n_states,)
        self.gamma = gamma
        self.soft_opt = self.soft_q_value_iteration()
        self.hard_opt = self.q_value_iteration()

    def q_value_iteration(self, n_iterations=1000):
        T = self.transition_prob
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
                pi = torch.nn.functional.one_hot(Q.argmax(dim=1)).float()
                return TabularPolicy(Q, V, pi)
        print(f"Q-value iteration did not converge in {i} iterations")
        return None

    def soft_q_value_iteration(self, n_iterations=1000, print_message=True):
        T = self.transition_prob
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

            if (Q - old_Q).abs().max() < 1e-5:
                if print_message:
                    print(f'soft value iteration converged in {i} iterations')
                pi = Q.softmax(dim=1)
                return TabularPolicy(Q, V, pi)
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
        assert n_states % 2 == 0, ("Ascender requires a central starting state with an equal number of states to the left"
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
        super().__init__(n_states, n_actions, T, R, gamma)

class OneStep(TabularMDP):
    def __init__(self, gamma=0.99):
        n_states = 4
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[:2,0,2] = 1
        T[:2,1,1] = 1
        R = torch.zeros(n_states)
        R[2] = 1
        super().__init__(n_states, n_actions, T, R, gamma)

def titus_meg(pi, T, n_iterations=1000, print_losses=False):
    n_states = T.shape[0]
    h = torch.randn((n_states,1), requires_grad=True)
    g = torch.randn((n_states,1), requires_grad=True)
    log_pi = pi.log()
    log_pi.requires_grad = False
    T.requires_grad = False

    optimizer = torch.optim.Adam([h, g], lr=1e-3)
    for i in range(n_iterations):
        old_g = g.detach().clone()
        loss = ((log_pi + h - g)**2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"Loss:{loss:.4f}")
        if (g - old_g).abs().max() < 1e-5:
            print(f'implicit policy learning converged in {i} iterations')
            return h, g
    return h, g
def main():
    m = AscenderLong(n_states=6)

    h, g = titus_meg(m.soft,m.T, print_losses=True)

    print("done")




if __name__ == "__main__":
    main()
