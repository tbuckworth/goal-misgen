import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from collections import deque


class Storage():

    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device, act_shape):
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.hidden_states_batch = torch.zeros(self.num_steps + 1, self.num_envs, self.hidden_state_size)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob_eval_policy = torch.zeros(self.num_steps, self.num_envs)
        self.subject_probs = torch.zeros(self.num_steps, self.num_envs, *self.act_shape)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done, info, log_prob_act, value, logp_eval_policy=None, probs=None):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)
        if logp_eval_policy is not None:
            self.log_prob_eval_policy[self.step] = torch.from_numpy(logp_eval_policy.copy())
        if probs is not None:
            self.subject_probs[self.step] = torch.from_numpy(probs.copy())
        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_hidden_state, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.hidden_states_batch[-1] = torch.from_numpy(last_hidden_state.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def translate_logp_mean_to_reward_mean(self):
        mu_logp = self.log_prob_eval_policy.mean()
        mu_r = self.rew_batch.mean()
        self.log_prob_eval_policy += mu_r - mu_logp

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i + 1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.return_batch = self.adv_batch + self.value_batch[:-1]
        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_unique_generator(self, mini_batch_size=None):
        # Logic for getting indices of unique tensors
        # TODO: if stochastic, then we need the triples (obs, acts, next_obs)
        all_obs = self.obs_batch[:-1].reshape(-1, *self.obs_shape)
        all_acts = self.act_batch.reshape(-1, 1)
        all_pairs = torch.concat((all_obs, all_acts), dim=-1)
        unique_pairs, rev_index = all_pairs.unique(dim=0, return_inverse=True)
        if len(unique_pairs) == len(all_pairs):
            return self.fetch_train_generator(mini_batch_size)
        indices = [(rev_index == i).argwhere()[0].item() for i in range(len(unique_pairs))]
        assert (all_pairs[indices] == unique_pairs).all(), "Check logic, but this should be correct"
        yield from self.collect_and_yield(indices)

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False, valid_envs=0, valid=False):
        if valid_envs >= self.num_envs:
            raise IndexError(f"valid_envs: {valid_envs} must be less than num_envs: {self.num_envs}")
        b_envs = valid_envs
        if not valid:
            b_envs = self.num_envs - valid_envs

        batch_size = self.num_steps * b_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If agent's policy is not recurrent, data could be sampled without considering the time-horizon
        if not recurrent:
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                                   mini_batch_size,
                                   drop_last=True)
            for indices in sampler:
                yield from self.collect_and_yield(indices, valid_envs, valid)
        # If agent's policy is recurrent, data should be sampled along the time-horizon
        else:
            num_mini_batch_per_epoch = batch_size // mini_batch_size
            num_envs_per_batch = self.num_envs // num_mini_batch_per_epoch
            perm = torch.randperm(self.num_envs)
            for start_ind in range(0, self.num_envs, num_envs_per_batch):
                idxes = perm[start_ind:start_ind + num_envs_per_batch]
                obs_batch = torch.FloatTensor(self.obs_batch[:-1, idxes]).reshape(-1, *self.obs_shape).to(self.device)
                # [0:1] instead of [0] to keep two-dimensional array
                hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[0:1, idxes]).reshape(-1,
                                                                                                     self.hidden_state_size).to(
                    self.device)
                act_batch = torch.FloatTensor(self.act_batch[:, idxes]).reshape(-1).to(self.device)
                done_batch = torch.FloatTensor(self.done_batch[:, idxes]).reshape(-1).to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch[:, idxes]).reshape(-1).to(self.device)
                value_batch = torch.FloatTensor(self.value_batch[:-1, idxes]).reshape(-1).to(self.device)
                return_batch = torch.FloatTensor(self.return_batch[:, idxes]).reshape(-1).to(self.device)
                adv_batch = torch.FloatTensor(self.adv_batch[:, idxes]).reshape(-1).to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def collect_and_yield(self, indices, valid_envs=0, valid=False):
        if valid:
            raise NotImplementedError("Storage has not implemented valid. see LirlStorage for implementation")
        obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
        hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[:-1]).reshape(-1,
                                                                                      self.hidden_state_size).to(
            self.device)
        act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
        done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
        log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
        value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
        return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
        adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
        yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def fetch_log_data(self):
        return self.rew_batch.numpy(), self.done_batch.numpy()
        # Don't know why we were bothering with this?
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()

        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        return rew_batch, done_batch

    def compute_importance_sampling_estimate(self, gamma):
        """
        Computes the off-policy evaluation estimate using importance sampling.

        Assumes:
          - self.rew_batch: Tensor of rewards (shape: [num_steps, num_envs])
          - self.log_prob_act_batch: Log probabilities under the behaviour policy (shape: [num_steps, num_envs])
          - self.log_prob_eval_policy: Log probabilities under the target policy (shape: [num_steps, num_envs])
          - self.step: Number of steps recorded (<= self.num_steps)
        """
        T = self.step  # actual number of steps recorded in the batch
        device = self.rew_batch.device

        # Create a discount vector: [1, gamma, gamma^2, ..., gamma^(T-1)]
        discounts = gamma ** torch.arange(T, dtype=torch.float32, device=device)

        # Compute per-environment discounted returns over T steps
        returns = (self.rew_batch[:T] * discounts.unsqueeze(1)).sum(dim=0)

        # Compute log importance weights for each environment: sum_t [log π(a_t|s_t) - log β(a_t|s_t)]
        log_imp_weights = (self.log_prob_eval_policy[:T] - self.log_prob_act_batch[:T]).sum(dim=0)
        imp_weights = torch.exp(log_imp_weights)

        # The final IS estimate is the average of (importance weight * return) over all environments
        is_estimate = (imp_weights * returns).mean()

        return is_estimate

    def compute_pdwis_estimate(self, gamma):
        """
        Computes the off-policy evaluation estimate using Per-Decision Weighted Importance Sampling (PDWIS).

        Assumes:
          - self.rew_batch: Tensor of rewards (shape: [num_steps, num_envs])
          - self.log_prob_act_batch: Log probabilities under the behaviour policy (shape: [num_steps, num_envs])
          - self.log_prob_eval_policy: Log probabilities under the target policy (shape: [num_steps, num_envs])
          - self.step: Number of steps recorded (<= self.num_steps)
          - self.gamma: Discount factor
        """

        T = self.step  # Actual number of steps recorded in the batch
        device = self.rew_batch.device

        # Create a discount vector: [1, gamma, gamma^2, ..., gamma^(T-1)]
        discounts = gamma ** torch.arange(T, dtype=torch.float32, device=device)  # Shape: [T]

        # Compute cumulative log importance weights for each time step:
        # diff_log has shape [T, num_envs] representing log π_eval - log β at each step.
        diff_log = self.log_prob_eval_policy[:T] - self.log_prob_act_batch[:T]
        # Cumulative log weights: each entry is sum_{j=0}^t (log π_eval - log β)
        cum_log_weights = torch.cumsum(diff_log, dim=0)  # Shape: [T, num_envs]
        # Convert log weights to weights
        weights = torch.exp(cum_log_weights)  # Shape: [T, num_envs]

        # For each time step, compute the per-decision estimate:
        #   ratio[t] = (sum_i [w[t,i] * rew_batch[t,i] * discounts[t]]) / (sum_i w[t,i])
        per_decision_estimates = []
        for t in range(T):
            numerator = (weights[t] * self.rew_batch[t] * discounts[t]).sum()
            denominator = weights[t].sum()
            # Guard against division by zero
            if denominator > 0:
                per_decision_estimate_t = numerator / denominator
            else:
                per_decision_estimate_t = 0.0
            per_decision_estimates.append(per_decision_estimate_t)

        # The final PDWIS estimate is the sum over time steps
        pdwis_estimate = sum(per_decision_estimates)
        return pdwis_estimate

    def get_returns(self, gamma=0.99):
        """
        Computes the discounted return for each complete episode (i.e. an episode that
        both starts and ends within the stored trajectory).

        Args:
            gamma (float): Discount factor.

        Returns:
            np.ndarray: An array of discounted returns, one per complete episode.
        """
        returns = []
        # Convert stored rewards and done flags to numpy arrays.
        # Assuming self.rew_batch and self.done_batch have shape (num_steps, num_envs)
        rewards = self.rew_batch.cpu().numpy()
        dones = self.done_batch.cpu().numpy()
        num_steps, num_envs = rewards.shape

        # Loop over each environment.
        for env in range(num_envs):
            episode_return = 0.0
            discount = 1.0
            started = False
            # Iterate over each time step in this environment.
            for t in range(num_steps):
                if started:
                    episode_return += discount * rewards[t, env]
                    discount *= gamma
                if dones[t, env]:
                    started = True
                    # Episode finished; store the return.
                    returns.append(episode_return)
                    # Reset for the next episode.
                    episode_return = 0.0
                    discount = 1.0
        return np.array(returns)


class LirlStorage(Storage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elementwise_meg = torch.zeros(self.num_steps, self.num_envs)

    def store_values(self, value_model, device, mini_batch_size):
        batch_size = self.num_steps * self.num_envs
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
        n = self.num_envs
        for indices in sampler:
            obs_batch = torch.FloatTensor(self.obs_batch).reshape(-1, *self.obs_shape)[indices].to(
                device)
            step = [i // n for i in indices]
            env = [i % n for i in indices]
            with torch.no_grad():
                self.value_batch[step, env] = value_model(obs_batch).to(self.value_batch.device).squeeze()

    def collect_and_yield(self, indices, valid_envs=0, valid=False):
        if not valid:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1, valid_envs:]).reshape(-1, *self.obs_shape)[indices].to(
                self.device)
            nobs_batch = torch.FloatTensor(self.obs_batch[1:, valid_envs:]).reshape(-1, *self.obs_shape)[indices].to(
                self.device)
            act_batch = torch.FloatTensor(self.act_batch[:, valid_envs:]).reshape(-1)[indices].to(self.device)
            done_batch = torch.FloatTensor(self.done_batch[:, valid_envs:]).reshape(-1)[indices].to(self.device)

            log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch[:, valid_envs:]).reshape(-1)[indices].to(
                self.device)
            value_batch = torch.FloatTensor(self.value_batch[:-1, valid_envs:]).reshape(-1)[indices].to(self.device)
            return_batch = torch.FloatTensor(self.return_batch[:, valid_envs:]).reshape(-1)[indices].to(self.device)
            adv_batch = torch.FloatTensor(self.adv_batch[:, valid_envs:]).reshape(-1)[indices].to(self.device)
            rew_batch = torch.FloatTensor(self.rew_batch[:, valid_envs:]).reshape(-1)[indices].to(self.device)
            log_prob_eval_policy = torch.FloatTensor(self.log_prob_eval_policy[:, valid_envs:]).reshape(-1)[indices].to(
                self.device)
            probs = torch.FloatTensor(self.subject_probs[:, valid_envs:]).reshape(-1, *self.act_shape)[indices].to(
                self.device)

        else:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1, :valid_envs]).reshape(-1, *self.obs_shape)[indices].to(
                self.device)
            nobs_batch = torch.FloatTensor(self.obs_batch[1:, :valid_envs]).reshape(-1, *self.obs_shape)[indices].to(
                self.device)
            act_batch = torch.FloatTensor(self.act_batch[:, :valid_envs]).reshape(-1)[indices].to(self.device)
            done_batch = torch.FloatTensor(self.done_batch[:, :valid_envs]).reshape(-1)[indices].to(self.device)
            log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch[:, :valid_envs]).reshape(-1)[indices].to(
                self.device)
            value_batch = torch.FloatTensor(self.value_batch[:-1, :valid_envs]).reshape(-1)[indices].to(self.device)
            return_batch = torch.FloatTensor(self.return_batch[:, :valid_envs]).reshape(-1)[indices].to(self.device)
            adv_batch = torch.FloatTensor(self.adv_batch[:, :valid_envs]).reshape(-1)[indices].to(self.device)
            rew_batch = torch.FloatTensor(self.rew_batch[:, :valid_envs]).reshape(-1)[indices].to(self.device)
            log_prob_eval_policy = torch.FloatTensor(self.log_prob_eval_policy[:, :valid_envs]).reshape(-1)[indices].to(
                self.device)
            probs = torch.FloatTensor(self.subject_probs[:, :valid_envs]).reshape(-1, *self.act_shape)[indices].to(
                self.device)
        # todo make this a dict:
        yield obs_batch, nobs_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch, rew_batch, log_prob_eval_policy, probs, indices

    def store_meg(self, elementwise_meg, indices, valid_envs=0):
        n = self.num_envs - valid_envs
        idx_pairs = [(i // n, i % n) for i in indices]
        row_idx, col_idx = zip(*idx_pairs)

        self.elementwise_meg[:, valid_envs:][row_idx, col_idx] = elementwise_meg.detach().cpu().clone().float()

    def full_meg(self, gamma=0.99, valid_envs=0):
        # TODO: double check this is legit
        d = self.done_batch.clone().bool()

        n_full_episodes = d.sum(dim=0) - 1

        n = torch.zeros_like(d).int()

        for col in range(d.shape[1]):
            start = False
            for row in range(d.shape[0]):
                done = d[row, col]
                if done:
                    start = True
                    count = 1
                if not d[row, col:].any():
                    start = False
                if start and row + 1 < d.shape[0]:
                    n[row + 1, col] = count
                    count += 1

        discounts = (gamma ** n) * (1 - (n == 0).int())
        full_meg = (self.elementwise_meg * discounts).sum() / n_full_episodes.sum()
        return full_meg
