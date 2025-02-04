import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import einops

from meg.meg_torch import unknown_utility_meg, state_action_occupancy
from matplotlib import pyplot as plt

def hard_adv_from_belmann(log_pi):
    return log_pi - (log_pi.exp() * log_pi).sum(dim=-1).unsqueeze(-1)

def cosine_similarity_loss(vec1, vec2):
    return 1 - F.cosine_similarity(vec1.reshape(-1), vec2.reshape(-1), dim=-1).mean()


norm_funcs = {
    "l1_norm": lambda x: x if (x == 0).all() else x / x.abs().mean(),
    "l2_norm": lambda x: x if (x == 0).all() else x / x.pow(2).mean().sqrt(),
    "linf_norm": lambda x: x if (x == 0).all() else x / x.abs().max(),
}

dist_funcs = {
    "l1_dist": lambda x, y: (x - y).abs().mean(),
    "l2_dist": lambda x, y: (x - y).pow(2).mean().sqrt(),
}

GAMMA = 0.9


def plot_canonicalised_rewards(canon, hard_canon):
    # 1. Compute the L2 norm of the 'Canonicalised Reward' column for each DataFrame
    canon_l2_norm = np.linalg.norm(canon['Canonicalised Reward'])
    hard_canon_l2_norm = np.linalg.norm(hard_canon['Canonicalised Reward'])

    # 2. Create normalized columns (or Series)
    canon['Normalized Canonicalised Reward'] = canon['Canonicalised Reward'] / canon_l2_norm
    hard_canon['Normalized Canonicalised Reward'] = hard_canon['Canonicalised Reward'] / hard_canon_l2_norm

    # 3. Create a figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Reward
    axes[0].scatter(hard_canon['Reward'], canon['Reward'], alpha=0.6)
    axes[0].set_xlabel('hard_canon: Reward')
    axes[0].set_ylabel('canon: Reward')
    axes[0].set_title('Reward Comparison')

    # Subplot 2: Canonicalised Reward
    axes[1].scatter(hard_canon['Canonicalised Reward'], canon['Canonicalised Reward'], alpha=0.6, color='orange')
    axes[1].set_xlabel('hard_canon: Canonicalised Reward')
    axes[1].set_ylabel('canon: Canonicalised Reward')
    axes[1].set_title('Canonicalised Reward Comparison')

    # Subplot 3: Normalized Canonicalised Reward
    axes[2].scatter(hard_canon['Normalized Canonicalised Reward'],
                    canon['Normalized Canonicalised Reward'],
                    alpha=0.6, color='green')
    axes[2].set_xlabel('hard_canon: Normalized Canonicalised Reward')
    axes[2].set_ylabel('canon: Normalized Canonicalised Reward')
    axes[2].set_title('Normalized Canonicalised Reward Comparison')

    plt.tight_layout()
    plt.show()


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
        self.R = R.detach().cpu()
        self.adjustment = adjustment.detach().cpu()
        self.v = v.detach().cpu()
        self.next_v = next_v.detach().cpu()
        self.C = self.R + self.adjustment
        self.n_actions = self.R.shape[-1]
        self.n_states = self.R.shape[-2]
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
        if len(np.unique([v.shape for k, v in self.data.items()])) > 1:
            raise Exception
        self.df = pd.DataFrame(self.data).round(decimals=2)

    def print(self):
        print(self.df.round(decimals=2))


class TabularMDPs:
    def __init__(self, n_mdps):
        self.pircs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = n_mdps
        self.n_states = 15
        self.n_actions = 5
        self.T = (torch.rand((self.N, self.n_states, self.n_actions, self.n_states), device=self.device) * 100).softmax(
            dim=-1)
        self.reward_vector = (torch.rand((self.N, self.n_states), device=self.device) * 1000).softmax(dim=-1) * 10
        self.alt_reward_vector = (torch.rand((self.N, self.n_states), device=self.device) * 1000).softmax(dim=-1) * 10

        self.gamma = torch.rand((self.N,), device=self.device) / 10 + 0.9
        belmann_hightemp, belmann_lowtemp = self.q_value_iteration(100000, inv_temp=[1, 1000])
        anti_high, anti_low = self.q_value_iteration(100000, inv_temp=[1, 1000], override_reward=self.alt_reward_vector)
        self.policies = {
            "BelmannHighTemp": belmann_hightemp,
            "BelmannLowTemp": belmann_lowtemp,
            "AlternaHighTemp": anti_high,
            "AlternaLowTemp": anti_low,
        }
        self.norm_funcs = {
            "L2": lambda x: x if (x == 0).all() else x / x.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt(),
            "L1": lambda x: x if (x == 0).all() else x / x.abs().amax(dim=(-1, -2), keepdim=True),
        }
        self.distance = lambda x, y: (x - y).pow(2).mean(dim=(-1, -2)).sqrt()
        self.uniform = torch.ones((self.N, self.n_states, self.n_actions), device=self.device).softmax(dim=-1)
        self.calc_pircs()

    def calc_pircs(self, verbose=True):
        self.canon = self.canonicalise(self.reward_vector)
        hard_style = ["Centred no C", "Centred"]
        hard_style = []
        for pirc_type in hard_style + ["Soft", "Hard"]:
            self.pirc(pirc_type)
        p = self.pircs

        # all_data = [{f"{k}:{policy}": d[policy] for k, d in p.items()} for policy in self.policies.keys()]
        # new_dict = {}
        # [new_dict.update(x) for x in all_data]

        df = pd.DataFrame(p).round(decimals=2)
        df = df.reindex(sorted(df.columns), axis=1)

        df.mean()
        lt = self.policies['BelmannLowTemp']
        ht = self.policies['BelmannHighTemp']
        ladv = lt.log_pi - (lt.log_pi.exp() * lt.log_pi).sum(dim=-1).unsqueeze(-1)
        hadv = ht.log_pi - (ht.log_pi.exp() * ht.log_pi).sum(dim=-1).unsqueeze(-1)
        # These should have a ratio of 'temp' but they don't.

        # df.drop(df.filter(regex='Centred').columns, axis=1).mean()
        if verbose:
            print(df)
        return df

    def q_value_iteration(self, n_iterations=10000, print_message=True, argmax=False, inv_temp=1000, invert_reward=False,
                          override_reward=None):
        T = self.T
        R = self.reward_vector.unsqueeze(-1)
        if invert_reward:
            R *= -1
        if override_reward is not None:
            R = override_reward.unsqueeze(-1)
        gamma = self.gamma.unsqueeze(-1)
        n_states = self.n_states
        n_actions = self.n_actions
        Q = torch.zeros((self.N, n_states, n_actions), device=self.device)

        for i in range(n_iterations):
            old_Q = Q
            V = Q.max(dim=-1).values
            Q = einops.einsum(T, gamma * V, 'N states actions next_states, N next_states -> N states actions') + R

            if (Q - old_Q).abs().max() < 1e-5:
                if print_message:
                    print(f'Q-value iteration converged in {i} iterations')
                if argmax:
                    # NOT SURE THIS WOULD WORK:
                    pi = torch.nn.functional.one_hot(Q.argmax(dim=-2), num_classes=n_actions).float()
                    policy_name = "Hard Argmax"
                else:
                    policy_name = "Hard Smax"
                    if type(inv_temp) == list:
                        # hadv = hard_adv_from_belmann((Q[0] * 1).log_softmax(dim=-1))
                        # ladv = hard_adv_from_belmann((Q[0] * 10).log_softmax(dim=-1))
                        # ratio = ladv/hadv
                        # print(f"mean:{ratio.mean():.2f}\tstd:{ratio.std():.2f}")
                        return [TabularPolicy(policy_name, (Q * t).softmax(dim=-1), Q, V) for t in inv_temp]
                    pi = (Q * inv_temp).softmax(dim=-1)
                return TabularPolicy(policy_name, pi, Q, V)

        print(f"Q-value iteration did not converge in {i} iterations")
        return None

    def pirc(self, pirc_type, own_policy=False):
        # assert len(self.policies)==1, "Need to change below to pircs, rather than pirc"
        for name, policy in self.policies.items():
            pirc = self.calc_pirc(policy, pirc_type, name)
            self.pircs.update(pirc)

    def calc_pirc(self, policy, pirc_type, name):
        trusted_pi = None
        if pirc_type in ["Centred", "Centred no C"]:
            adv = policy.log_pi - policy.log_pi.mean(dim=-1).unsqueeze(-1)
            if pirc_type == "Centred":
                ca = self.canonicalise(adv, trusted_pi)
            elif pirc_type == "Centred no C":
                ca = adv
            else:
                raise NotImplementedError
        elif pirc_type == "Soft":
            adv = policy.log_pi
            ca = self.canonicalise(adv, trusted_pi)
        elif pirc_type == "Hard":
            adv = policy.log_pi - (policy.pi * policy.log_pi).sum(dim=-1).unsqueeze(dim=-1)
            ca = self.canonicalise(adv, trusted_pi)
            policy.hard_ca = ca
        else:
            raise NotImplementedError(f"pirc_type must be one of 'Hard','Hard no C','Soft'. Not {pirc_type}.")

        comp_canon = self.canon

        outputs = {}
        for norm, normalize in self.norm_funcs.items():
            nca = normalize(ca)
            ncr = normalize(comp_canon)
            outputs[f"{pirc_type}:{norm}:{name}"] = self.distance(nca, ncr).tolist()
        return outputs

    def canonicalise(self, R, trusted_pi=None):
        if trusted_pi is None:
            trusted_pi = self.uniform

        R3 = torch.zeros_like(self.T, device=self.device)

        if R.ndim == 2:
            # N, State reward becomes N, state, action, next state
            R3 = R.unsqueeze(-2).unsqueeze(-2).tile([1] + list(self.T.shape[-3:-1]) + [1])
        elif R.ndim == 3:
            # N, State, action reward becomes N, state, action, next state
            R3 = R.unsqueeze(-1).tile(self.T.shape[-1])
        elif R.ndim == 4:
            R3 = R
        else:
            raise Exception(f"R.ndim must be 2, 3, or 4, not {R.ndim}.")

        v = self.value_iteration(trusted_pi, R3, n_iterations=100000)

        R2 = (R3 * self.T).sum(dim=-1)

        next_v = (self.T * v.unsqueeze(-2).unsqueeze(-2)).sum(dim=-1)
        adjustment = self.gamma.unsqueeze(-1).unsqueeze(-1) * next_v - v.unsqueeze(-1)

        return R2 + adjustment

    def value_iteration(self, pi, R, n_iterations: int = 10000):
        T = self.T
        n_states = self.n_states
        V = torch.zeros((self.N, n_states), device=self.device)
        gamma = self.gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        for _ in range(n_iterations):
            old_V = V
            Q = einops.einsum(T, (R + gamma * V.unsqueeze(-2).unsqueeze(-2)), "N s a ns, N s a ns -> N s a")
            V = einops.einsum(pi, Q, "N s a, N s a -> N s")
            if (V - old_V).abs().max() < 1e-5:
                return V
        print(f"Value Iteration did not converge after {n_iterations} iterations.")
        return None


class TabularMDP:
    custom_policies = []

    def __init__(self, n_states, n_actions, transition_prob, reward_vector, mu=None, gamma=GAMMA, name="Unnamed", device=None):
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.own_pircs = {}
        self.new_pircs = {}
        self.canon = None
        assert (transition_prob.sum(dim=-1) - 1).abs().max() < 1e-5, "Transition Probabilities do not sum to 1"
        self.mu = torch.ones(n_states, device=self.device) / n_states if mu is None else mu.to(device=self.device)
        self.megs = {}
        self.pircs = {}
        self.returns = {}
        self.normalize = norm_funcs["l2_norm"]
        self.distance = dist_funcs["l2_dist"]
        self.name = name
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = transition_prob.to(device=self.device)  # Shape: (n_states, n_actions, n_states)
        self.reward_vector = reward_vector.to(device=self.device)  # Shape: (n_states,)
        self.gamma = gamma

        self.soft_opt = self.soft_q_value_iteration(print_message=False, n_iterations=10000)
        self.hard_opt = self.q_value_iteration(print_message=False, n_iterations=10000, argmax=True)
        self.hard_smax = self.q_value_iteration(print_message=False, n_iterations=10000, argmax=False)

        q_uni = torch.zeros((n_states, n_actions), device=self.device)
        unipi = q_uni.softmax(dim=-1)
        self.uniform = TabularPolicy("Uniform", unipi, q_uni, q_uni.logsumexp(dim=-1))
        # self.hard_adv_cosine = self.hard_adv_learner(print_message=False,
        #                                              n_iterations=10000,
        #                                              criterion=cosine_similarity_loss,
        #                                              name="Hard Adv Cosine")
        # self.hard_adv = self.hard_adv_learner(print_message=False, n_iterations=10000)
        # self.hard_adv_stepped = self.hard_adv_learner_stepped(print_message=False, n_iterations=10000)
        # self.hard_adv_cont = self.hard_adv_learner_continual(print_message=False, n_iterations=10000)
        policies = [self.soft_opt,
                    self.hard_opt,
                    self.hard_smax,
                    # self.hard_adv,
                    # self.hard_adv_cosine,
                    # self.hard_adv_stepped,
                    # self.hard_adv_cont,
                    self.uniform] + self.custom_policies
        self.policies = {p.name: p for p in policies if p is not None}

    def evaluate_policy(self, policy: TabularPolicy):
        R3 = torch.zeros_like(self.T)
        # State reward becomes state, action, next state
        R3[:, :, :] = self.reward_vector
        v = self.value_iteration(policy.pi, R3)

        ret = einops.einsum(self.mu, v, "s, s ->")
        return ret.item()

    def q_value_iteration(self, n_iterations=1000, print_message=True, argmax=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions
        Q = torch.zeros(n_states, n_actions, device=self.device)

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
                    pi = (Q * 1000).softmax(dim=-1)
                return TabularPolicy(policy_name, pi, Q, V)

        print(f"Q-value iteration did not converge in {i} iterations")
        return None

    def hard_adv_learner(self, n_iterations=1000, lr=1e-1, print_message=True, criterion=nn.MSELoss(), name="Hard Adv"):
        T = self.T
        R = self.reward_vector
        n_states = self.n_states
        n_actions = self.n_actions

        logits = torch.rand((n_states, n_actions), requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([logits], lr=lr)
        with torch.no_grad():
            cr = self.canonicalise(R).C.to(device=self.device)

        for i in range(n_iterations):
            old_logits = logits.detach().clone()
            log_pi = logits.log_softmax(dim=-1)
            g = log_pi - (log_pi.exp() * log_pi).sum(dim=-1).unsqueeze(-1)
            loss = criterion(g, cr)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if torch.allclose(logits, old_logits, atol=1e-5):
                if print_message:
                    print(f'hard adv learning converged in {i} iterations')
                pi = logits.softmax(dim=-1).detach()
                return TabularPolicy(name, pi)
        print('hard adv learning did not converge after', n_iterations, 'iterations')
        return None

    def hard_adv_learner_stepped(self, n_iterations=1000, lr=1e-1, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        logits = (torch.ones((n_states, n_actions)) / n_actions).log()
        logits.requires_grad = True

        optimizer = torch.optim.Adam([logits], lr=lr)
        for epoch in range(n_iterations):
            start_logits = logits.detach().clone()
            with torch.no_grad():
                cr = self.canonicalise(R, logits.softmax(dim=-1)).C

            for i in range(n_iterations):
                old_logits = logits.detach().clone()
                log_pi = logits.log_softmax(dim=-1)
                g = log_pi - log_pi.mean(dim=-1).unsqueeze(-1)
                loss = ((g - cr) ** 2).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if torch.allclose(logits, old_logits, atol=1e-5):
                    break
            if torch.allclose(logits, start_logits, atol=1e-5):
                if print_message:
                    print(f'hard adv stepped converged in {epoch} epochs')
                pi = logits.softmax(dim=-1).detach()
                return TabularPolicy("Hard Adv Stepped", pi)
        print('hard adv stepped did not converge after', n_iterations, 'epochs')
        return None

    def hard_adv_learner_continual(self, n_iterations=1000, lr=1e-1, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        logits = (torch.ones((n_states, n_actions)) / n_actions).log()
        logits.requires_grad = True
        optimizer = torch.optim.Adam([logits], lr=lr)
        for i in range(n_iterations):
            start_logits = logits.detach().clone()
            with torch.no_grad():
                cr = self.canonicalise(R, logits.softmax(dim=-1)).C
            log_pi = logits.log_softmax(dim=-1)
            g = log_pi - log_pi.mean(dim=-1).unsqueeze(-1)
            loss = ((g - cr) ** 2).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if torch.allclose(logits, start_logits, atol=1e-5):
                if print_message:
                    print(f'hard adv continual converged in {i} iterations')
                pi = logits.softmax(dim=-1).detach()
                return TabularPolicy("Hard Adv Cont", pi)
        print('hard adv continual did not converge after', n_iterations, 'iterations')
        return None

    def soft_q_value_iteration(self, n_iterations=1000, print_message=True):
        T = self.T
        R = self.reward_vector
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions

        Q = torch.zeros(n_states, n_actions, device=self.device)

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

    def calc_pirc(self, policy, pirc_type, own_policy=False):
        trusted_pi = policy.pi if own_policy else None
        if pirc_type in ["Centred", "Centred no C"]:
            adv = policy.log_pi - policy.log_pi.mean(dim=-1).unsqueeze(-1)
            if pirc_type == "Centred":
                policy.centred_canon = self.canonicalise(adv, trusted_pi)
                ca = policy.centred_canon.C
            elif pirc_type == "Centred no C":
                v = torch.zeros((adv.shape[0], 1))
                z = torch.zeros_like(adv)
                policy.centred_no_canon = RewardFunc(adv, v, z, z)
                ca = adv
            else:
                raise NotImplementedError
        elif pirc_type == "Soft":
            adv = policy.log_pi
            policy.soft_canon = self.canonicalise(adv, trusted_pi)
            ca = policy.soft_canon.C
        elif pirc_type == "Hard":
            adv = policy.log_pi - (policy.pi * policy.log_pi).sum(dim=-1).unsqueeze(dim=-1)
            policy.hard_canon = self.canonicalise(adv, trusted_pi)
            ca = policy.hard_canon.C
        else:
            raise NotImplementedError(f"pirc_type must be one of 'Hard','Hard no C','Soft'. Not {pirc_type}.")

        comp_canon = self.canon
        if own_policy:
            comp_canon = self.canonicalise(self.reward_vector, trusted_pi)

        nca = self.normalize(ca)
        ncr = self.normalize(comp_canon.C)

        return self.distance(nca, ncr).item()

    def meg(self, method="matt_meg", time_it=False, verbose=False, atol=1e-5):
        meg_func = meg_funcs[method]
        if verbose:
            print(f"\n{self.name} Environment\t{method}:")
        megs = {}
        for name, policy in self.policies.items():
            start = time.time()
            meg = meg_func(policy.pi, self.T, self.mu, device=self.device, suppress=True, atol=atol)
            elapsed = time.time() - start
            if verbose:
                print(f"{name}\tMeg: {meg:.4f}\tElapsed: {elapsed:.4f}")
            megs[name] = {"Meg": meg.item(),"Time": elapsed} if time_it else meg.item()
        self.megs[method] = megs

    def pirc(self, pirc_type, own_policy=False):
        pircs = {}

        for name, policy in self.policies.items():
            pirc = self.calc_pirc(policy, pirc_type, own_policy)
            pircs[name] = pirc
        if own_policy:
            self.own_pircs[pirc_type] = pircs
            return
        self.pircs[pirc_type] = pircs

    def new_pirc(self, is_hard):
        pircs = {}

        for name, policy in self.policies.items():
            pirc = self.calc_new_pirc(policy, is_hard)
            pircs[name] = pirc
        type_name = "Centred" if is_hard else "Soft"
        self.new_pircs[type_name] = pircs

    def calc_megs(self, verbose=False, time_it=False, atol=1e-5):
        for mf in meg_funcs.keys():
            self.meg(mf, time_it=time_it, atol=atol)
        if time_it:
            df = pd.concat(
                {person: pd.DataFrame(methods).T for person, methods in self.megs.items()},
                axis=1
            ).round(decimals=2)
        else:
            df = pd.DataFrame(self.megs).round(decimals=2)
        if verbose:
            print(f"{self.name} Environment")
            print(df)
        return df

    def calc_pircs(self, verbose=False, own_policy=False):
        self.canon = self.canonicalise(self.reward_vector)
        hard_style = ["Centred no C"]
        if own_policy:
            hard_style = ["Centred"]
        for pirc_type in hard_style + ["Soft", "Hard"]:
            self.pirc(pirc_type, own_policy)
        p = self.own_pircs if own_policy else self.pircs
        df = pd.DataFrame(p).round(decimals=2)
        if verbose:
            print(f"\n{self.name} Environment")
            print(df)
        return df

    def calc_new_pircs(self, verbose=False):
        if self.canon is None:
            self.canon = self.canonicalise(self.reward_vector)
        for is_hard in [True, False]:
            self.new_pirc(is_hard)
        df = pd.DataFrame(self.new_pircs).round(decimals=2)
        if verbose:
            print(f"\n{self.name} Environment")
            print(df)
        return df

    def calc_returns(self, verbose=False):
        returns = {}
        for name, policy in self.policies.items():
            returns[name] = self.evaluate_policy(policy)
        self.returns["Centred"] = returns
        df = pd.DataFrame(self.returns).round(decimals=2)
        if verbose:
            print(f"{self.name} Environment")
            print(df)
        return df

    def meg_pirc(self):
        if self.pircs == {}:
            self.calc_pircs()
        if self.megs == {}:
            self.calc_megs()
        if self.returns == {}:
            self.calc_returns()
        d = {"Meg": self.megs, "PIRC": self.pircs, "Expected Return": self.returns}
        df = pd.concat({k: pd.DataFrame.from_dict(v, orient='index').T for k, v in d.items()}, axis=1)
        return df.round(decimals=2)

    def all_pirc(self):
        if self.pircs == {}:
            self.calc_pircs()
        if self.new_pircs == {}:
            self.calc_new_pircs()
        if self.own_pircs == {}:
            self.calc_pircs(own_policy=True)
        d = {"PIRC": self.pircs, "New PIRC": self.new_pircs, "Own PIRC": self.own_pircs}
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
        V = torch.zeros((n_states), device=self.device)

        for _ in range(n_iterations):
            old_V = V
            Q = einops.einsum(T, (R + self.gamma * V.view(1, 1, -1)), "s a ns, s a ns -> s a")
            V = einops.einsum(pi, Q, "s a, s a -> s")
            if (V - old_V).abs().max() < 1e-5:
                return V
        print(f"Value Iteration did not converge after {n_iterations} iterations.")
        return None

    def reward_from_policy(self, policy: TabularPolicy, n_iterations=10000, lr=1e-1, hard=False):
        T = self.T
        gamma = self.gamma
        n_states = self.n_states
        n_actions = self.n_actions
        A = policy.log_pi
        if hard:
            A = A - A.mean(dim=-1).unsqueeze(dim=-1)
        Q = torch.randn((n_states, n_actions), requires_grad=True)
        optimizer = torch.optim.Adam([Q], lr=lr)

        for i in range(n_iterations):
            old_Q = Q.detach().clone()
            if hard:
                V = einops.einsum(policy.pi, Q, "s a, s a -> s").unsqueeze(-1)
            else:
                V = Q.logsumexp(dim=-1).unsqueeze(-1)
            A_hat = Q - V
            loss = ((A - A_hat) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if torch.allclose(Q, old_Q, atol=1e-5):
                if hard:
                    V = einops.einsum(policy.pi, Q, "s a, s a -> s")
                else:
                    V = Q.logsumexp(dim=-1)
                # Vn = einops.einsum(T,V, "s a ns, ns -> s a")
                next_v = (T * V.view(1, 1, -1)).sum(dim=-1)

                return (A - gamma * next_v + V.unsqueeze(-1)).detach().cpu()
        print(f"Reward from policy did not converge after {n_iterations} iterations.")
        return None

    def calc_new_pirc(self, policy, is_hard):
        R = self.reward_from_policy(policy, hard=is_hard)
        if R is None:
            return np.nan
        canon = self.canonicalise(R)
        if is_hard:
            policy.new_hard_canon = canon
        else:
            policy.new_soft_canon = canon

        cr = canon.C

        nca = self.normalize(cr)
        ncr = self.normalize(self.canon.C)

        return self.distance(nca, ncr).item()


# def matt_meg(pi, T, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, suppress=False):
#     # TODO: do Matt's new version
#     n_states, n_actions, _ = T.shape
#     q = torch.randn(n_states, n_actions, requires_grad=True)
#     log_pi = pi.log()
#     log_pi.requires_grad = False
#     T.requires_grad = False
#
#     optimizer = torch.optim.Adam([q], lr=lr)
#     for i in range(n_iterations):
#         old_q = q.detach().clone()
#         g = einops.einsum(q, T, "s a, s a ns -> ns")
#         v = q.logsumexp(dim=-1).unsqueeze(-1)
#
#         target = einops.einsum(v + log_pi, T, 's a, s a ns -> ns')
#         loss = ((target - g) ** 2).mean()
#
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         if print_losses and i % 10 == 0:
#             print(f"Loss:{loss:.4f}")
#         if (q - old_q).abs().max() < 1e-5:
#             meg = calculate_meg(pi, q, T, GAMMA, mu)
#             if not suppress:
#                 print(f'Matt Meg converged in {i} iterations. Meg:{meg:.4f}')
#             return meg
#     print(f'Matt Meg did not converge in {i} iterations')
#     return None


def calculate_meg(pi, q, T, gamma, mu, device):
    eps = 1e-9
    n_actions = pi.shape[1]
    pi_soft = q.softmax(dim=-1)
    da = state_action_occupancy(pi, T, gamma, mu, device=device)
    meg = einops.einsum(da, torch.log(pi_soft + eps) - np.log(1 / n_actions),
                        'states actions, states actions ->')
    return meg
    # Titus incorrect version:
    log_pi_soft_star = q.log_softmax(dim=-1)
    meg = ((pi * log_pi_soft_star).sum(dim=-1) - np.log(1 / n_actions)).sum()
    return meg


def titus_meg(pi, T, mu=None, n_iterations=10000, lr=1e-1, print_losses=False, device="cpu", suppress=False, atol=1e-5,
              state_based=True, soft=True):
    n_states, n_actions, _ = T.shape
    if state_based:
        g_shape = (n_states,)
    else:
        g_shape = (n_states, n_actions)
    g = torch.randn(g_shape, requires_grad=True, device=device)
    pi = pi.to(device)
    log_pi = pi.log()
    log_pi.requires_grad = False
    T.requires_grad = False

    optimizer = torch.optim.Adam([g], lr=lr)
    for i in range(n_iterations):
        old_g = g.detach().clone()
        if state_based:
            q = einops.einsum(T, g, "s a ns, ns -> s a")
        else:
            q = g
        if soft:
            a = log_pi
            v = q.logsumexp(dim=-1).unsqueeze(-1)
        else:
            a = log_pi - (pi*log_pi).sum(dim=-1).unsqueeze(-1)
            v = (pi*q).sum(dim=-1).unsqueeze(-1)
        loss = ((a + v - q) ** 2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if print_losses and i % 10 == 0:
            print(f"Loss:{loss:.4f}")
        if (g - old_g).abs().max() < atol:
            if state_based:
                q = einops.einsum(T, g, "s a ns, ns -> s a")
            else:
                q = g
            meg = calculate_meg(pi, q, T, GAMMA, mu, device)
            if not suppress:
                print(f'Titus Meg converged in {i} iterations. Meg:{meg:.4f}')
            return meg.detach().cpu()
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
        mu[(n_states - 1) // 2] = 1.

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
    "Titus Meg Soft S": titus_meg,
    "Titus Meg Soft SxA": lambda pi, T, mu, device, suppress, atol: titus_meg(pi, T, mu=mu, device=device, atol=atol, soft=True, state_based=False, suppress=suppress),
    "Titus Meg Hard S": lambda pi, T, mu, device, suppress, atol: titus_meg(pi, T, mu=mu, device=device, atol=atol, soft=False, state_based=True, suppress=suppress),
    "Titus Meg Hard SxA": lambda pi, T, mu, device, suppress, atol: titus_meg(pi, T, mu=mu, device=device, atol=atol, soft=False, state_based=False, suppress=suppress),
    # "Real Meg": lambda pi, T, mu, device, suppress, atol: unknown_utility_meg(pi, T, gamma=GAMMA, mu=mu, device=device, atol=0.01),
}


def main():
    envs = [AscenderLong(n_states=6), MattGridworld(), OneStep(), ]
    # envs = [MattGridworld()]
    envs = {e.name: e for e in envs}

    for name, env in envs.items():
        df = env.calc_megs(verbose=False, time_it=False, atol=1e-5)
        print(df)
        # print(f"\n{name}:\n{env.meg_pirc()}")

    return

    name = "Ascender: 2 Pos States"
    name = "Matt Gridworld"
    policy = "Hard Smax"
    envs[name].canon.print()
    envs[name].policies[policy].soft_canon.print()
    envs[name].policies[policy].hard_canon.print()

    p = envs[name].policies[policy]
    plot_canonicalised_rewards(envs[name].canon.df, p.hard_canon.df)

    for _, env in envs.items():
        print(env.policies[policy].hard_canon.adjustment.unique())

    for _, env in envs.items():
        for _, policy in env.policies.items():
            print(((policy.hard_canon.v).abs() < 1e-5).all())

    for name, env in envs.items():
        env.calc_megs(verbose=True)

    print("done")


def try_hard_adv_train():
    envs = [AscenderLong(n_states=6), MattGridworld(), OneStep(), ]
    envs = {e.name: e for e in envs}
    # envs[0].evaluate_policy(envs[0].uniform)
    [print(env.all_pirc()) for name, env in envs.items()]

    return
    name = "Ascender: 2 Pos States"
    policy = "Hard Smax"
    envs[name].canon.print()
    envs[name].policies[policy].new_hard_canon.print()
    envs[name].policies[policy].hard_canon.print()
    envs[name].policies[policy].new_soft_canon.print()
    envs[name].policies[policy].soft_canon.print()


def vMDP():
    envs = TabularMDPs(50)


if __name__ == "__main__":
    # vMDP()
    main()
    # try_hard_adv_train()
