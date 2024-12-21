import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import einops

from common.meg.meg_colab import unknown_utility_meg, state_action_occupancy
from helper_local import norm_funcs, dist_funcs

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
        self.R = None


# Define a tabular MDP


class RewardFunc:
    def __init__(self, R, v, next_v, adjustment):
        self.R = R
        self.adjustment = adjustment
        self.v = v
        self.next_v = next_v
        self.C = R + adjustment
        self.n_actions = self.R.shape[1]
        self.n_states = self.R.shape[0]
        self.state = torch.arange(self.n_states).unsqueeze(-1).tile(self.n_actions)
        self.action = torch.arange(self.n_actions).unsqueeze(0).tile(self.n_states, 1)
        self.data = {
            "State": self.state.reshape(-1),
            "Action": self.action.reshape(-1),
            "Reward": self.R.reshape(-1),
            "Value": self.v.tile(self.n_actions).reshape(-1),
            "Next Value": self.next_v.reshape(-1),
            "Adjustment": self.adjustment.reshape(-1),
            "Canonicalised Reward": self.C.reshape(-1),
        }
        self.df = pd.DataFrame(self.data).round(decimals=2)
    def print(self):
        print(self.df.round(decimals=2))


class TabularMDP:
    custom_policies = []

    def __init__(self, n_states, n_actions, transition_prob, reward_vector, mu=None, gamma=GAMMA, name="Unnamed"):
        assert (transition_prob.sum(dim=-1) - 1).abs().max() < 1e-5, "Transition Probabilities do not sum to 1"
        self.mu = torch.ones(n_states) / n_states if mu is None else mu
        self.megs = {}
        self.pircs = {}
        self.normalize = norm_funcs["l2_norm"]
        self.distance = dist_funcs["l2_dist"]
        self.name = name
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = transition_prob  # Shape: (n_states, n_actions, n_states)
        self.reward_vector = reward_vector  # Shape: (n_states,)
        self.gamma = gamma
        self.soft_opt = self.soft_q_value_iteration(print_message=False, n_iterations=10000)
        self.hard_opt = self.q_value_iteration(print_message=False, n_iterations=10000, argmax=True)
        self.hard_smax = self.q_value_iteration(print_message=False, n_iterations=10000, argmax=False)

        q_uni = torch.zeros((n_states, n_actions))
        unipi = q_uni.softmax(dim=-1)
        self.uniform = TabularPolicy("Uniform", unipi, q_uni, q_uni.logsumexp(dim=-1))
        policies = [self.soft_opt, self.hard_opt, self.hard_smax, self.uniform] + self.custom_policies
        self.policies = {p.name: p for p in policies}

    def q_value_iteration(self, n_iterations=1000, print_message=True, argmax=True):
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
                if argmax:
                    pi = torch.nn.functional.one_hot(Q.argmax(dim=1), num_classes=n_actions).float()
                    policy_name = "Hard Argmax"
                else:
                    policy_name = "Hard Smax"
                    pi = (Q*1000).softmax(dim=-1)
                return TabularPolicy(policy_name, pi, Q, V)

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

    def calc_pirc(self, policy, pirc_type):
        if pirc_type == "Hard":
            adv = policy.log_pi - policy.log_pi.mean(dim=-1).unsqueeze(-1)
            policy.hard_canon = self.canonicalise(adv)
            ca = policy.hard_canon.C
        elif pirc_type == "Soft":
            adv = policy.log_pi
            policy.soft_canon = self.canonicalise(adv)
            ca = policy.soft_canon.C
        else:
            raise NotImplementedError(f"pirc_type must be one of 'Hard','Soft'. Not {pirc_type}.")

        nca = self.normalize(ca)
        ncr = self.normalize(self.canon.C)

        # plt.scatter(nca.cpu().numpy(), ncr.cpu().numpy())
        # plt.show()

        return self.distance(nca, ncr).item()

    def meg(self, method="matt_meg", verbose=False):
        meg_func = meg_funcs[method]
        if verbose:
            print(f"\n{self.name} Environment\t{method}:")
        megs = {}
        for name, policy in self.policies.items():
            meg = meg_func(policy.pi, self.T, self.mu, suppress=True)
            if verbose:
                print(f"{name}\tMeg: {meg:.4f}")
            megs[name] = meg
        self.megs[method] = megs

    def pirc(self, pirc_type):
        pircs = {}

        for name, policy in self.policies.items():
            pirc = self.calc_pirc(policy, pirc_type)
            pircs[name] = pirc
        self.pircs[pirc_type] = pircs

    def calc_megs(self, verbose=False):
        for mf in meg_funcs.keys():
            self.meg(mf)
        df = pd.DataFrame(self.megs).round(decimals=2)
        if verbose:
            print(f"{self.name} Environment")
            print(df)
        return df

    def calc_pircs(self, verbose=False):
        self.canon = self.canonicalise(self.reward_vector)
        for pirc_type in ["Hard", "Soft"]:
            self.pirc(pirc_type)
        df = pd.DataFrame(self.pircs).round(decimals=2)
        if verbose:
            print(f"\n{self.name} Environment")
            print(df)
        return df

    def meg_pirc(self):
        if self.pircs == {}:
            self.calc_pircs()
        if self.megs == {}:
            self.calc_megs()
        d = {"Meg": self.megs, "PIRC": self.pircs}
        df = pd.concat({k: pd.DataFrame.from_dict(v, orient='index').T for k, v in d.items()}, axis=1)
        return df.round(decimals=2)

    def canonicalise(self, R, trusted_pi=None):
        if trusted_pi is None:
            trusted_pi = self.uniform.pi

        R3 = torch.zeros_like(self.T)

        if R.ndim == 1:
            # State reward becomes state, action, next state
            R3[:, :, :] = R
        elif R.ndim == 2:
            # State action reward becomes state, action, next state
            R3 = R.unsqueeze(-1).tile(self.T.shape[-1])
        elif R.ndim == 3:
            R3 = R
        else:
            raise Exception(f"R.ndim must be 1, 2, or 3, not {R.ndim}.")

        v = self.value_iteration(trusted_pi, R3)

        R2 = (R3 * self.T).sum(dim=-1)

        next_v = (self.T * v.view(1, 1, -1)).sum(dim=-1)
        adjustment = self.gamma * next_v - v.view(-1, 1)

        return RewardFunc(R2, v.view(-1, 1), next_v, adjustment)

    def value_iteration(self, pi, R, n_iterations: int = 10000):
        T = self.T
        n_states = self.n_states
        V = torch.zeros((n_states))

        for _ in range(n_iterations):
            old_V = V
            Q = einops.einsum(T, (R + self.gamma * V.view(1, 1, -1)), "s a ns, s a ns -> s a")
            V = einops.einsum(pi, Q, "s a, s a -> s")
            if (V - old_V).abs().max() < 1e-5:
                return V
        print(f"Value Iteration did not converge after {n_iterations} iterations.")
        return None


def matt_meg(pi, T, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, suppress=False):
    # TODO: do Matt's new version
    n_states, n_actions, _ = T.shape
    q = torch.randn(n_states, n_actions, requires_grad=True)
    log_pi = pi.log()
    log_pi.requires_grad = False
    T.requires_grad = False

    optimizer = torch.optim.Adam([q], lr=lr)
    for i in range(n_iterations):
        old_q = q.detach().clone()
        g = einops.einsum(q, T, "s a, s a ns -> ns")
        v = q.logsumexp(dim=-1).unsqueeze(-1)

        target = einops.einsum(v + log_pi, T, 's a, s a ns -> ns')
        loss = ((target - g) ** 2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Loss:{loss:.4f}")
        if (q - old_q).abs().max() < 1e-5:
            meg = calculate_meg(pi, q, T, GAMMA, mu)
            if not suppress:
                print(f'Matt Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg
    print(f'Matt Meg did not converge in {i} iterations')
    return None


def calculate_meg(pi, q, T, gamma, mu):
    eps = 1e-9
    n_actions = pi.shape[1]
    pi_soft = q.softmax(dim=-1).detach().cpu().numpy()
    da = state_action_occupancy(pi.cpu().numpy(), T.cpu().numpy(), gamma, mu.cpu().numpy())
    meg = einops.einsum(da, np.log(pi_soft + eps) - np.log(1 / n_actions),
                        'states actions, states actions ->')
    return meg
    # Titus incorrect version:
    log_pi_soft_star = q.log_softmax(dim=-1)
    meg = ((pi * log_pi_soft_star).sum(dim=-1) - np.log(1 / n_actions)).sum()
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
        q = einops.einsum(T, g, "s a ns, ns -> s a")
        v = q.logsumexp(dim=-1).unsqueeze(-1)
        loss = ((log_pi + v - q) ** 2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Loss:{loss:.4f}")
        if (g - old_g).abs().max() < 1e-5:
            meg = calculate_meg(pi, q, T, GAMMA, mu)
            if not suppress:
                print(f'Titus Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg
    print(f'Titus Meg did not converge in {i} iterations')
    return None


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
        T[(0, -1, -2), :, -1] = 1  # /n_actions

        R = torch.zeros(n_states)
        R[-2] = 10
        R[0] = -10

        mu = torch.zeros(n_states)
        mu[(n_states - 1) // 3] = 1.

        go_left = torch.zeros((n_states, n_actions))
        go_left[:, 0] = 1
        self.go_left = TabularPolicy("Go Left", go_left)
        self.custom_policies = [self.go_left]
        super().__init__(n_states, n_actions, T, R, mu, gamma, f"Ascender: {int((n_states - 2) // 2)} Pos States")


class OneStep(TabularMDP):
    def __init__(self, gamma=GAMMA):
        n_states = 5
        n_actions = 2
        T = torch.zeros(n_states, n_actions, n_states)
        T[:2, 0, 2] = 1
        T[:2, 1, 3] = 1
        T[2:, :, -1] = 1  # /n_actions
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
    "Titus Meg": titus_meg,
    # "non_tabular_titus_meg": non_tabular_titus_meg,
    # "matt_meg": matt_meg,
    "Real Meg": lambda pi, T, mu, suppress: unknown_utility_meg(pi.cpu().numpy(), T.cpu().numpy(), gamma=GAMMA,
                                                           mu=mu.cpu().numpy()),
}


def main():
    envs = [MattGridworld(), OneStep(), AscenderLong(n_states=6), ]
    envs = {e.name: e for e in envs}

    for name, env in envs.items():
        print(f"\n{name}:\n{env.meg_pirc()}")

    return

    name = "Matt Gridworld"

    envs[name].canon.print()
    envs[name].policies["Hard Smax"].soft_canon.print()
    envs[name].policies["Hard Smax"].hard_canon.print()

    for _, env in envs.items():
        for _, policy in env.policies.items():
            print(((policy.hard_canon.v).abs()<1e-5).all())

    for name, env in envs.items():
        env.calc_megs(verbose=True)

    print("done")


if __name__ == "__main__":
    main()
