import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import einops

from common.meg.meg_colab import unknown_utility_meg
GAMMA = 0.9

class TabularPolicy:
    def __init__(self, name, pi, Q=None, V=None):
        self.name = name
        self.Q = Q
        self.V = V
        self.pi = pi
        # This is in case we have a full zero, we adjust policy.
        flt = (self.pi == 0).any(dim=-1)
        self.pi[flt] = (self.pi[flt] * 10).softmax(dim=-1)
        assert (self.pi.sum(dim=-1).round(
            decimals=3) == 1).all(), "pi is not a probability distribution along final dim"
        self.log_pi = self.pi.log()


# Define a tabular MDP
class TabularMDP:
    custom_policies = []
    def __init__(self, n_states, n_actions, transition_prob, reward_vector, mu=None, gamma=GAMMA, name="Unnamed"):
        self.mu = torch.ones(n_states) / n_states if mu is None else mu
        self.megs = {}
        self.name = name
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = transition_prob  # Shape: (n_states, n_actions, n_states)
        self.reward_vector = reward_vector  # Shape: (n_states,)
        self.gamma = gamma
        self.soft_opt = self.soft_q_value_iteration(print_message=False, n_iterations=10000)
        self.hard_opt = self.q_value_iteration(print_message=False, n_iterations=10000)

        q_uni = torch.zeros((n_states, n_actions))
        unipi = q_uni.softmax(dim=-1)
        self.uniform = TabularPolicy("Uniform", unipi, q_uni, q_uni.logsumexp(dim=-1))
        self.policies = [self.soft_opt, self.hard_opt, self.uniform] + self.custom_policies

    def q_value_iteration(self, n_iterations=1000, print_message=True):
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
                if print_message:
                    print(f'Q-value iteration converged in {i} iterations')
                pi = torch.nn.functional.one_hot(Q.argmax(dim=1), num_classes=n_actions).float()
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

    def meg(self, method="matt_meg"):
        meg_func = meg_funcs[method]
        print(f"\n{self.name} Environment\t{method}:")
        megs = {}
        for policy in self.policies:
            meg = meg_func(policy.pi, self.T, self.mu, suppress=True)
            print(f"{policy.name}\tMeg: {meg:.4f}")
            megs[policy.name] = meg
        self.megs = megs

def matt_meg(pi, T, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, suppress=False):
    #TODO: do Matt's new version
    n_states, n_actions, _ = T.shape
    q = torch.randn(n_states, n_actions, requires_grad=True)
    log_pi = pi.log()
    log_pi.requires_grad = False
    T.requires_grad = False

    optimizer = torch.optim.Adam([q], lr=lr)
    for i in range(n_iterations):
        old_q = q.detach().clone()
        g = einops.einsum(q, T, "s a, s a ns -> ns")
        # g(s') -> logsumexpQ(S,.) + log pi
        v = q.logsumexp(dim=-1).unsqueeze(-1)

        target = v + log_pi
        meg_proxy = ((target - g.unsqueeze(-1)) ** 2).mean()

        # Actually calculating meg here. can use either as loss, but meg_proxy converges faster.
        log_pi_soft_star = q.log_softmax(dim=-1)

        meg = ((pi * log_pi_soft_star).sum(dim=-1) - np.log(1 / n_actions)).sum()

        loss = meg_proxy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Tmeg:{meg_proxy:.4f}\tMeg:{meg:.4f}")
        if (q - old_q).abs().max() < 1e-5:
            if not suppress:
                print(f'Titus Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg
    print(f'Titus Meg did not converge in {i} iterations')
    return meg


def titus_meg(pi, T, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, suppress=False):
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

        meg = ((pi * log_pi_soft_star).sum(dim=-1) - np.log(1 / n_actions)).sum()

        loss = meg_proxy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Tmeg:{meg_proxy:.4f}\tMeg:{meg:.4f}")
        if (g - old_g).abs().max() < 1e-5:
            if not suppress:
                print(f'Titus Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg
    print(f'Titus Meg did not converge in {i} iterations')
    return meg

def non_tabular_titus_meg(pi, T, n_iterations=10000, lr=1e-1, print_losses=False, suppress=False):
    n_states, n_actions, _ = T.shape
    g = torch.randn((n_states, 1), requires_grad=True)
    h = torch.randn((n_states, 1), requires_grad=True)

    log_pi = pi.log()
    log_pi.requires_grad = False
    T.requires_grad = False

    optimizer = torch.optim.Adam([g, h], lr=lr)
    for i in range(n_iterations):
        old_g = g.detach().clone()
        # v = q_est.logsumexp(dim=-1).unsqueeze(-1)
        meg_proxy = ((log_pi + h - g) ** 2).mean()

        # Actually calculating meg here. can use either as loss, but meg_proxy converges faster.
        q_est = einops.einsum(T, g.squeeze(), "s a ns, ns -> s a")
        log_pi_soft_star = q_est.log_softmax(dim=-1)

        meg = ((pi * log_pi_soft_star).sum(dim=-1) - np.log(1 / n_actions)).sum()

        loss = meg_proxy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Tmeg:{meg_proxy:.4f}\tMeg:{meg:.4f}")
        if (g - old_g).abs().max() < 1e-5:
            if not suppress:
                print(f'Titus Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg
    print(f'Titus Meg did not converge in {i} iterations')
    return meg


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
    def __init__(self, n_states, gamma=GAMMA):
        assert n_states % 2 == 0, (
            "Ascender requires a central starting state with an equal number of states to the left"
            " and right, plus an infinite terminal state. Therefore n_states must be even.")
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        for i in range(n_states - 3):
            T[i + 1, 1, i + 2] = 1
            T[i + 1, 0, i] = 1
        T[(0, -1, -2), :, -1] = 1/n_actions

        R = torch.zeros(n_states)
        R[-2] = 10
        R[0] = -10

        mu = torch.zeros(n_states)
        mu[(n_states-1)//3] = 1.

        go_left = torch.zeros((n_states, n_actions))
        go_left[:, 0] = 1
        self.go_left = TabularPolicy("Go Left", go_left)
        self.custom_policies = [self.go_left]
        super().__init__(n_states, n_actions, T, R, gamma, f"Ascender: {int(n_states-2/2)} Pos States")


class OneStep(TabularMDP):
    def __init__(self, gamma=GAMMA):
        n_states = 5
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[:2, 0, 2] = 1
        T[:2, 1, 3] = 1
        T[2:, :, -1] = 1/n_actions
        R = torch.zeros(n_states)
        R[2] = 1
        mu = torch.zeros(n_states)
        mu[:2] = 0.5

        consistent_pi = torch.zeros(n_states, n_actions)
        consistent_pi[:2] = torch.FloatTensor([0.2, 0.8])
        consistent_pi[2:] = 0.5

        inconsistent_pi = consistent_pi.clone()
        inconsistent_pi[1] = torch.FloatTensor([0.8, 0.2])
        self.consistent = TabularPolicy("Consistent", consistent_pi)
        self.inconsistent = TabularPolicy("Inconsistent", inconsistent_pi)
        self.custom_policies = [self.consistent, self.inconsistent]
        super().__init__(n_states, n_actions, T, R, mu, gamma, "One Step")

class MattGridworld(TabularMDP):
    def __init__(self, gamma=GAMMA, N=5):
        n_states = N * N

        actions = ['up', 'down', 'left', 'right']
        n_actions = len(actions)

        T = torch.zeros((n_states, n_actions, n_states))

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
        U = torch.rand(n_states)
        mu = torch.zeros(n_states)
        mu[0] = 1
        super().__init__(n_states, n_actions, T, U, mu, gamma, "Matt Gridworld")


meg_funcs = {
    "titus_meg": titus_meg,
    # "non_tabular_titus_meg": non_tabular_titus_meg,
    "matt_meg": matt_meg,
    "meg": lambda pi, T, mu, suppress: unknown_utility_meg(pi.cpu().numpy(), T.cpu().numpy(), gamma=GAMMA, mu=mu.cpu().numpy()),
}

def main():
    env = OneStep()
    for mf in meg_funcs.keys():
        env.meg(mf)

    env = MattGridworld()
    for mf in meg_funcs.keys():
        env.meg(mf)

    env.meg()
    env.meg("titus_meg")

    mdp = AscenderLong(n_states=6)
    for mf in meg_funcs.keys():
        print(mf)
        mdp.meg(mf)

    mdp = AscenderLong(n_states=20)
    mdp.meg()
    mdp.meg()

    print("done")


if __name__ == "__main__":
    main()
