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
        self.disc_batch = torch.ones(self.num_envs)
        self.cum_returns = torch.zeros(self.num_envs)
        self.cum_log_imp_weights = torch.zeros(self.num_envs)
        self.is_estimate = torch.zeros(self.num_envs)
        self.pdwis_estimate = torch.zeros(self.num_envs)
        self.t = torch.zeros(self.num_envs)
        self.cliw_hist = [{} for _ in range(self.num_envs)]
        self.episode_returns = []
        self.episode_is_ests = []
        self.episode_pdwis_ests = []
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0
        # self.disc_batch[self.step] = 1.

    def store(self, obs, hidden_state, act, rew, done, info, log_prob_act, value, logp_eval_policy=None, probs=None, gamma=None):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)
        if gamma is not None and logp_eval_policy is not None:
            disc_rew = self.disc_batch * rew
            self.cum_returns += disc_rew
            self.cum_log_imp_weights += (logp_eval_policy - log_prob_act)
            self.is_estimate += self.cum_log_imp_weights.exp() * disc_rew
            # PDWIS:
            for i, (t,cum_liw) in enumerate(zip(self.t, self.cum_log_imp_weights)):
                self.cliw_hist[i].update({t:cum_liw.item()})
            norm_weights = []
            for i,t in enumerate(self.t):
                norm = np.exp([d[t] for d in self.cliw_hist if t in d.keys()]).mean() + 1e-8
                norm_weights.append(self.cum_log_imp_weights[i].exp()/norm)
            # normalized_weights = self.cum_log_imp_weights.exp() / (sum(self.cum_log_imp_weights.exp()) + 1e-8)
            normalized_weights = torch.tensor(norm_weights)
            self.pdwis_estimate += normalized_weights * disc_rew
            if done.any():
                self.episode_returns.append(self.cum_returns[done].tolist())
                self.episode_is_ests.append(self.is_estimate[done].tolist())
                self.episode_pdwis_ests.append(self.pdwis_estimate[done].tolist())
            self.disc_batch *= gamma
            self.t += 1
            self.disc_batch[done] = 1.
            self.cum_returns[done] = 0.
            self.cum_log_imp_weights[done] = 0.
            self.is_estimate[done] = 0.
            self.t[done] = 0

        if logp_eval_policy is not None:
            self.log_prob_eval_policy[self.step] = torch.from_numpy(logp_eval_policy.copy())
            self.accumulate_importance_samling_estimates(gamma)
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
        T = self.step or self.num_steps  # actual number of steps recorded in the batch

        device = self.rew_batch.device

        # Create a discount vector: [1, gamma, gamma^2, ..., gamma^(T-1)]
        discounts = gamma ** torch.arange(T, dtype=torch.float32, device=device)

        # Compute per-environment discounted returns over T steps
        returns = (self.rew_batch[:T] * discounts.unsqueeze(1)).sum(dim=0)

        # Compute log importance weights for each environment: sum_t [log π(a_t|s_t) - log β(a_t|s_t)]
        log_imp_weights = (self.log_prob_eval_policy[:T] - self.log_prob_act_batch[:T]).sum(dim=0)
        imp_weights = log_imp_weights.exp()

        # The final IS estimate is the average of (importance weight * return) over all environments
        is_estimate = (imp_weights * returns).mean()

        return is_estimate

    def compute_pdwis_estimate_old(self, gamma):
        """
        Computes the off-policy evaluation estimate using Per-Decision Weighted Importance Sampling (PDWIS).

        Assumes:
          - self.rew_batch: Tensor of rewards (shape: [num_steps, num_envs])
          - self.log_prob_act_batch: Log probabilities under the behaviour policy (shape: [num_steps, num_envs])
          - self.log_prob_eval_policy: Log probabilities under the target policy (shape: [num_steps, num_envs])
          - self.step: Number of steps recorded (<= self.num_steps)
          - self.gamma: Discount factor
        """

        T = self.step or self.num_steps  # Actual number of steps recorded in the batch
        device = self.rew_batch.device

        # Create a discount vector: [1, gamma, gamma^2, ..., gamma^(T-1)]
        discounts = gamma ** torch.arange(T, dtype=torch.float32, device=device)  # Shape: [T]

        # Compute cumulative log importance weights for each time step:
        # diff_log has shape [T, num_envs] representing log π_eval - log β at each step.
        diff_log = self.log_prob_eval_policy[:T] - self.log_prob_act_batch[:T]
        # Cumulative log weights: each entry is sum_{j=0}^t (log π_eval - log β)
        cum_log_weights = torch.cumsum(diff_log, dim=0)  # Shape: [T, num_envs]
        # Convert log weights to weights
        weights = cum_log_weights.exp()  # Shape: [T, num_envs]

        # For each time step, compute the per-decision estimate:
        #   ratio[t] = (sum_i [w[t,i] * rew_batch[t,i] * discounts[t]]) / (sum_i w[t,i])
        per_decision_estimates = []
        for t in range(T):
            # Think that summing may be wrong here - why over env dimension? also are dones taken into account?
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

    def compute_pdwis_estimate(self, gamma):
        """
        Computes the off-policy evaluation estimate using Per-Decision Weighted Importance Sampling (PDWIS),
        accounting for episodes that may not be aligned across environments.

        Assumes:
          - self.rew_batch: Tensor of rewards (shape: [num_steps, num_envs])
          - self.log_prob_act_batch: Log probabilities under the behaviour policy (shape: [num_steps, num_envs])
          - self.log_prob_eval_policy: Log probabilities under the target policy (shape: [num_steps, num_envs])
          - self.done_batch: Boolean tensor indicating terminal steps (shape: [num_steps, num_envs])
          - self.step: Number of steps recorded (<= self.num_steps)
          - self.gamma: Discount factor
        """
        T = self.step or self.num_envs
        # Only consider the recorded steps.
        num_steps, num_envs = self.rew_batch[:T].shape
        device = self.rew_batch.device

        # Dictionaries to accumulate numerator and denominator for each relative time step (across episodes).
        acc = {}  # key: relative time index t_episode, value: sum_i (w_t * reward * gamma^t)
        denom = {}  # key: relative time index t_episode, value: sum_i w_t

        # Loop over each environment (each column in the batch).
        for env in range(num_envs):
            t_episode = 0  # relative time index within the current episode.
            cumulative_log_weight = 0.0  # will be reset at the start of each episode.

            for t in range(num_steps):
                # If this is the start of the column or a new episode (previous step ended episode), reset counters.
                if t == 0 or self.done_batch[t - 1, env]:
                    t_episode = 0
                    cumulative_log_weight = 0.0

                # Update cumulative importance weight (in log-space) for the current step.
                log_diff = (self.log_prob_eval_policy[t, env] - self.log_prob_act_batch[t, env]).item()
                cumulative_log_weight += log_diff
                weight = torch.exp(torch.tensor(cumulative_log_weight, dtype=torch.float32, device=device))

                # Get the reward and discount factor for this relative step.
                reward = self.rew_batch[t, env]
                discount = gamma ** t_episode

                # Accumulate numerator and denominator for this relative time index.
                if t_episode not in acc:
                    acc[t_episode] = 0.0
                    denom[t_episode] = 0.0
                acc[t_episode] += (weight * reward * discount).item()
                denom[t_episode] += weight.item()

                t_episode += 1  # Increment relative time index.

        # Compute per-decision estimates by averaging across all episodes that contributed at each relative time index.
        pdwis_estimates = []
        max_t = max(acc.keys()) if acc else 0
        for t_episode in range(max_t + 1):
            if denom.get(t_episode, 0.0) > 0:
                per_decision = acc[t_episode] / denom[t_episode]
            else:
                per_decision = 0.0
            pdwis_estimates.append(per_decision)

        # The final PDWIS estimate is the sum over the per-decision estimates.
        final_estimate = sum(pdwis_estimates)
        return final_estimate

    def compute_off_policy_estimates(self, gamma):
        return np.mean(self.episode_is_ests), np.mean(self.episode_pdwis_ests), np.mean(self.episode_returns)
        """
        Computes the off-policy evaluation estimates using:
          - Importance Sampling (IS)
          - Per-Decision Weighted Importance Sampling (PDWIS)
          - Behavioural policy return

        Accounts for misaligned episodes stored in deques.

        Assumes:
          - self.rew_batch: Tensor of rewards (shape: [num_steps, num_envs])
          - self.log_prob_act_batch: Log probabilities under the behaviour policy (shape: [num_steps, num_envs])
          - self.log_prob_eval_policy: Log probabilities under the target policy (shape: [num_steps, num_envs])
          - self.done_batch: Boolean tensor indicating terminal steps (shape: [num_steps, num_envs])
          - self.step: Number of steps recorded (<= self.num_steps)
          - gamma: Discount factor (can be provided as an argument)

        Returns:
          A tuple (IS_estimate, PDWIS_estimate, Behaviour_policy_return)
        """
        # Use self.step if available, otherwise fallback to self.num_steps.
        T = self.step or self.num_steps
        num_steps, num_envs = self.rew_batch[:T].shape

        # For IS: store importance-weighted discounted return per complete episode.
        is_episode_estimates = []
        # For behavioural policy: store discounted return per complete episode.
        behavior_episode_returns = []

        # For PDWIS: accumulate numerator and denominator per relative time step (across all complete episodes)
        pdwis_numerators = {}  # key: relative time step, value: sum_i (w_t * r_t * gamma^t)
        pdwis_denoms = {}  # key: relative time step, value: sum_i (w_t)

        # Loop over each environment (each column)
        for env in range(num_envs):
            t_episode = 0  # Relative time index within the episode.
            cumulative_log_weight = 0.0  # Cumulative log importance weight for the episode.
            discounted_return = 0.0  # Discounted sum of rewards for the episode (for IS & behavioural return).
            gamma_power = 1.0  # Tracks gamma^t for the current step.

            for t in range(num_steps):
                # If this is the first step or a new episode starts, reset counters.
                if t == 0 or self.done_batch[t - 1, env]:
                    t_episode = 0
                    cumulative_log_weight = 0.0
                    discounted_return = 0.0
                    gamma_power = 1.0

                # Update cumulative importance weight (in log space)
                log_diff = self.log_prob_eval_policy[t, env] - self.log_prob_act_batch[t, env]
                cumulative_log_weight += log_diff
                weight = cumulative_log_weight.exp()  # Convert log weight to weight.

                # Get the reward at this step and current discount.
                reward = self.rew_batch[t, env].item()
                discount = gamma_power

                # --- PDWIS accumulation ---
                pdwis_numerators[t_episode] = pdwis_numerators.get(t_episode, 0.0) + weight * reward * discount
                pdwis_denoms[t_episode] = pdwis_denoms.get(t_episode, 0.0) + weight

                # --- IS and behavioural return accumulation ---
                discounted_return += discount * reward

                # Update for the next step.
                gamma_power *= gamma
                t_episode += 1

                # If this step ends an episode, finalize the episode's contributions.
                if self.done_batch[t, env]:
                    episode_weight = cumulative_log_weight.exp()
                    is_episode_estimates.append(episode_weight * discounted_return)
                    behavior_episode_returns.append(discounted_return)
                    # Counters will be reset automatically at the next step if a new episode begins.

        # Compute final IS estimate: average over complete episodes.
        if len(is_episode_estimates) > 0:
            is_estimate = sum(is_episode_estimates) / len(is_episode_estimates)
        else:
            is_estimate = 0.0

        # Compute PDWIS estimate: sum over relative time steps of (numerator / denominator).
        pdwis_estimate = 0.0
        if pdwis_numerators:
            max_t = max(pdwis_numerators.keys())
            for t_episode in range(max_t + 1):
                denom = pdwis_denoms.get(t_episode, 0.0)
                if denom > 0:
                    pdwis_estimate += pdwis_numerators[t_episode] / denom
        else:
            pdwis_estimate = 0.0

        # Compute the average behavioural policy return (over complete episodes).
        if len(behavior_episode_returns) > 0:
            behaviour_return = sum(behavior_episode_returns) / len(behavior_episode_returns)
        else:
            behaviour_return = 0.0

        # Convert to scalar floats if they are torch.Tensors.
        if isinstance(is_estimate, torch.Tensor):
            is_estimate = is_estimate.item()
        if isinstance(pdwis_estimate, torch.Tensor):
            pdwis_estimate = pdwis_estimate.item()

        return is_estimate, pdwis_estimate, behaviour_return

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

    def accumulate_importance_samling_estimates(self, gamma):
        assert gamma, "gamma required for importance sampling"




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
