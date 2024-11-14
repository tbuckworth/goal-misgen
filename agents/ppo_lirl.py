from common.policy import UniformPolicy
from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
import numpy as np


class PPO_Lirl(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 env_valid=None,
                 storage_valid=None,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 l1_coef=0.,
                 anneal_lr=True,
                 num_rew_updates=10,
                 rew_model=None,
                 next_rew_model=None,
                 inv_temp_rew_model=1.,
                 next_rew_loss_coef=1.,
                 storage_trusted=None,
                 **kwargs):

        super(PPO_Lirl, self).__init__(env, policy, logger, storage, device,
                                       n_checkpoints, env_valid, storage_valid)
        self.trusted_policy = UniformPolicy(policy.action_size, device)
        self.next_rew_loss_coef = next_rew_loss_coef
        self.inv_temp = inv_temp_rew_model
        self.rew_model = rew_model
        self.next_rew_model = next_rew_model
        self.anneal_lr = anneal_lr
        self.l1_coef = l1_coef
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.num_rew_updates = num_rew_updates
        self.storage_trusted = storage_trusted

    def predict(self, obs, hidden_state, done, policy=None):
        if policy is None:
            policy = self.policy
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1 - done).to(device=self.device)
        dist, value, hidden_state = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), hidden_state.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def predict_for_logit_saliency(self, obs, act):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        dist, value, hidden_state = self.policy(obs, None, None)
        log_prob_act = dist.log_prob(torch.tensor(act).to(device=self.device))
        log_prob_act.backward(retain_graph=True)

        return obs.grad.data.detach().cpu().numpy()

    def predict_for_rew_saliency(self, obs, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        dist, value, hidden_state = self.policy(obs, None, None)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act, value, obs

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list, l1_reg_list, total_loss_list = [], [], [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1 - done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip,
                                                                                              self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                l1_reg = sum([param.abs().sum() for param in self.policy.parameters()])

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss + l1_reg * self.l1_coef
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                l1_reg_list.append(l1_reg.item())
                total_loss_list.append(loss.item())

        summary = {
            'Loss/total': np.mean(total_loss_list),
            'Loss/pi': np.mean(pi_loss_list),
            'Loss/v': np.mean(value_loss_list),
            'Loss/entropy': np.mean(entropy_loss_list),
            'Loss/l1_reg': np.mean(l1_reg_list)
        }
        return summary

    def train(self, num_timesteps):
        learn_rew_every = num_timesteps // self.num_rew_updates
        rew_checkpoint_cnt = 0
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)
            self.collect_rollouts(done_v, hidden_state_v, obs_v, self.storage_trusted, self.env_valid,
                                  self.trusted_policy)
            # Need to re-do this, so it's fresh for the valid env data collection:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            self.collect_rollouts(done, hidden_state, obs, self.storage, self.env)

            # valid
            if self.env_valid is not None:
                self.collect_rollouts(done_v, hidden_state_v, obs_v, self.storage_valid, self.env_valid)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)
            self.logger.dump(summary)
            if self.anneal_lr:
                self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)

            if self.t > ((rew_checkpoint_cnt + 1) * learn_rew_every):
                self.optimize_reward()
                with torch.no_grad():
                    self.evaluate_correlation()

            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every):
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()

    def collect_rollouts(self, done, hidden_state, obs, storage, env, policy=None):
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, policy)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state
        value_batch = storage.value_batch[:self.n_steps]
        _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, policy)
        storage.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

    def optimize_reward(self):
        rew_loss_list, next_rew_loss_list, total_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.rew_model.train()
        self.policy.eval()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                # TODO: get nobs_batch going also return rew_batch
                obs_batch, nobs_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, rew_batch = sample
                flt = done_batch.bool()

                with torch.no_grad():
                    dist_batch, value_batch, h_batch = self.policy.forward_with_embedding(obs_batch)
                    _, _, next_h_batch = self.policy.forward_with_embedding(nobs_batch)

                log_prob_act_batch = dist_batch.log_prob(act_batch)
                log_prob_act_batch *= self.inv_temp

                rew, value = self.rew_model(h_batch)
                next_rew, next_val = self.rew_model(next_h_batch)
                next_rew_est = self.next_rew_model(h_batch, act_batch)

                # flt is true for penultimate obs - we cannot calculate
                # (next_val for penultimate obs) = (val of terminal obs) (because there is no terminal obs)
                adv = rew[~flt] - value[~flt] + self.gamma * next_val[~flt]
                # instead we use the next rew est of the penultimate obs,
                # because V(sT) = R(sT) = NR(sT-1,aT-1), so we can express without reference to T:
                adv_dones = rew[flt] = value[flt] + self.gamma * next_rew_est[flt]

                loss = torch.nn.MSELoss()(adv.squeeze(), log_prob_act_batch[~flt])
                loss2 = torch.nn.MSELoss()(adv_dones.squeeze(), log_prob_act_batch[flt])

                # just scaling the losses according to number of observations:
                coef = flt.mean()
                rew_loss = (1 - coef) * loss + coef * loss2

                # detach grads for next_rew?
                next_rew_loss = torch.nn.MSELoss()(next_rew_est, next_rew)

                total_loss = rew_loss + next_rew_loss * self.next_rew_loss_coef

                total_loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                rew_loss_list.append(rew_loss.item())
                next_rew_loss_list.append(next_rew_loss.item())
                total_loss_list.append(total_loss.item())

        # rew bit is misguided - use the logic from evaluate_correlation.
        # rew, value = self.rew_model(h_batch)
        #
        # rew_corr = torch.corrcoef(torch.stack((rew.squeeze(), rew_batch.squeeze())))[0, 1].item()
        # val_corr = torch.corrcoef(torch.stack((value.squeeze(), value_batch.squeeze())))[0, 1].item()
        # TODO: just directly upload to wandb?
        summary = {
            'Loss/total_rew_loss': np.mean(total_loss_list),
            'Loss/rew_loss': np.mean(rew_loss_list),
            'Loss/next_rew_loss': np.mean(next_rew_loss_list),
            # 'Corr/rew_corr': rew_corr,
            # 'Corr/value_corr': val_corr,
        }
        return summary

    def evaluate_correlation(self):
        rew_loss_list, next_rew_loss_list, total_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size

        self.policy.eval()
        self.rew_model.eval()
        self.next_rew_model.eval()

        recurrent = self.policy.is_recurrent()
        generator = self.storage_trusted.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                               recurrent=recurrent)
        rew_tuples = [self.sample_next_rews(sample) for sample in generator]
        rew_est, rew = zip(*rew_tuples)

        rew_hat = torch.concat(list(rew_est))
        rew_batch = torch.concat(list(rew))
        rew_corr = torch.corrcoef(torch.stack((rew_hat.squeeze(), rew_batch.squeeze())))[0, 1].item()

        summary = {
            'Corr/rew_corr_val_env': rew_corr,
        }
        return summary

    def sample_next_rews(self, sample):
        obs_batch, _, act_batch, done_batch, _, _, _, _, rew_batch = sample
        _, _, h_batch = self.policy.forward_with_embedding(obs_batch)
        next_rew_est = self.next_rew_model(h_batch, act_batch)
        return next_rew_est, rew_batch
