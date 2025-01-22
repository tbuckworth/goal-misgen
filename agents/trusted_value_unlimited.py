import os

import wandb
from torch import nn

from helper_local import plot_values_ascender
from .base_agent import BaseAgent
import torch
import torch.optim as optim
import numpy as np


class TrustedValue(BaseAgent):
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
                 mini_batch_per_epoch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 value_model=None,
                 value_model_val=None,
                 val_epoch=100,
                 trusted_policy=None,
                 n_val_envs=0,
                 save_pics_ascender=False,
                 td_lmbda=True,
                 **kwargs):

        super(TrustedValue, self).__init__(env, policy, logger, storage, device,
                                           n_checkpoints, env_valid, storage_valid)

        self.td_lmbda = td_lmbda
        self.save_pics_ascender = save_pics_ascender
        if n_val_envs >= n_envs:
            raise IndexError(f"n_val_envs:{n_val_envs} must be less than n_envs:{n_envs}")
        self.n_val_envs = n_val_envs
        self.val_epoch = val_epoch
        self.trusted_policy = trusted_policy
        self.value_model = value_model
        self.value_model_val = value_model_val
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate, eps=1e-5)
        self.value_optimizer_val = optim.Adam(self.value_model_val.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict_temp(self, obs, act, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            act = torch.FloatTensor(act).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            logp_eval_policy = dist.log_prob(act).cpu().numpy()
        return logp_eval_policy

    def predict(self, obs, hidden_state, done, policy=None, value_model=None):
        if policy is None:
            policy = self.policy
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = policy(obs, hidden_state, mask)
            if value_model is not None and self.td_lmbda:
                value = value_model(obs)
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

    def train(self, num_timesteps):
        min_loss = min_loss_val = np.inf
        e = e_val = t = 0
        while t < num_timesteps:
            # Collect supervised data for unshifted env
            obs = self.env.reset()
            hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done = np.zeros(self.n_envs)
            t += self.collect_rollouts(done, hidden_state, obs, self.storage, self.env, self.trusted_policy, self.value_model)

            # Collect supervised data for shifted env
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)
            self.collect_rollouts(done_v, hidden_state_v, obs_v, self.storage_valid, self.env_valid, self.trusted_policy, self.value_model_val)

            min_loss, e = self.optimize_value(self.storage, self.value_model, self.value_optimizer, "Training", min_loss, e)
            min_loss_val, e_val = self.optimize_value(self.storage_valid, self.value_model_val, self.value_optimizer_val, "Validation", min_loss_val, e_val)

        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()

    def collect_rollouts(self, done, hidden_state, obs, storage, env, policy=None, value_model=None, save_extra=False):
        logp_eval_policy = None
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, policy, value_model)
            if save_extra:
                logp_eval_policy = self.predict_temp(obs, act, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value, logp_eval_policy)
            obs = next_obs
            hidden_state = next_hidden_state
        _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, policy, value_model)
        storage.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
        return self.n_steps * env.num_envs

    def optimize_value(self, storage, value_model, value_optimizer, env_type, min_val_loss=np.inf, epochs=0):
        filepath = f'{self.logger.logdir}/{env_type}'
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_cnt = 1
        # if self.td_lmbda:
        #     storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
        value_model.train()
        for e in range(self.val_epoch):

            generator = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                      recurrent=False,
                                                      valid_envs=self.n_val_envs,
                                                      valid=False,
                                                      )
            generator_valid = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                            recurrent=False,
                                                            valid_envs=self.n_val_envs,
                                                            valid=True,
                                                            )
            val_losses = []
            val_losses_valid = []

            for sample in generator:
                obs_batch, nobs_batch, _, done_batch, _, _, return_batch, _, rew_batch, _ = sample
                value_batch = value_model(obs_batch).squeeze()
                if self.td_lmbda:
                    target = return_batch
                else:
                    next_value_batch = value_model(nobs_batch).squeeze()
                    target = rew_batch + self.gamma * next_value_batch * (1 - done_batch)
                value_loss = nn.MSELoss()(target, value_batch)
                value_loss.backward()
                val_losses.append(value_loss.item())

            for sample in generator_valid:
                obs_batch_val, nobs_batch_val, _, done_batch_val, _, _, return_batch_val, _, rew_batch_val, _ = sample
                with torch.no_grad():
                    value_batch_val = value_model(obs_batch_val).squeeze()
                    if self.td_lmbda:
                        target = return_batch_val
                    else:
                        next_value_batch_val = value_model(nobs_batch_val).squeeze()
                        target = rew_batch_val + self.gamma * next_value_batch_val * (1 - done_batch_val)
                    value_loss_val = nn.MSELoss()(target, value_batch_val)
                val_losses_valid.append(value_loss_val.item())
            value_optimizer.step()
            value_optimizer.zero_grad()
            grad_accumulation_cnt += 1

            mean_val_loss = np.mean(val_losses_valid)
            wandb.log({
                f'Loss/value_epoch': e + epochs,
                f'Loss/value_loss_{env_type}': np.mean(val_losses),
                f'Loss/value_loss_valid_{env_type}': mean_val_loss,
            })
            if mean_val_loss < min_val_loss:
                # Save the model
                print("Saving model.")

                torch.save({'model_state_dict': value_model.state_dict(),
                            'optimizer_state_dict': value_optimizer.state_dict()},
                           f'{filepath}/model_min_val_loss.pth')
                min_val_loss = mean_val_loss
            else:
                if e >= self.val_epoch - 1:
                    return min_val_loss, e
            if self.save_pics_ascender and e % 100 == 0:
                plot_values_ascender(self.logger.logdir, obs_batch, value_batch.detach(), e)
        return min_val_loss, e