from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BPO(BaseAgent):
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
                 uniform_value=False,
                 store_hard_adv=True,
                 bpo_clip=1.0,
                 **kwargs):

        super(BPO, self).__init__(env, policy, logger, storage, device,
                                  n_checkpoints, env_valid, storage_valid)

        self.inv_temp = nn.Parameter(torch.tensor([1.],requires_grad=True,device=device))
        self.bpo_clip = bpo_clip
        self.store_hard_adv = store_hard_adv
        self.uniform_value = uniform_value
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
        self.optimizer = optim.Adam([x for x in self.policy.parameters()] + [self.inv_temp], lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
            if self.store_hard_adv:
                log_prob_act = log_prob_act + dist.entropy()

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

    def predict_for_logit_saliency(self, obs, act, all_acts=False):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        dist, value, hidden_state = self.policy(obs, None, None)
        if all_acts:
            dist.logits.mean().backward(retain_graph=True)
        else:
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

    def optimize(self, loss_func):
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
                entropy_loss, l1_reg, loss, pi_loss, value_loss = loss_func(sample)
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
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

    def ppo_loss(self, sample):
        if self.store_hard_adv:
            raise Exception("store_hard_adv=True incompatible with ppo. set num_bpo_timesteps to None or > num_timesteps")
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
        value_loss = self.clipped_bellman_error(old_value_batch, return_batch, value_batch)
        l1_reg = sum([param.abs().sum() for param in self.policy.parameters()])
        # Policy Entropy
        entropy_loss = dist_batch.entropy().mean()
        loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss + l1_reg * self.l1_coef
        return entropy_loss, l1_reg, loss, pi_loss, value_loss

    def clipped_bellman_error(self, old_value_batch, return_batch, value_batch):
        value_loss = self.clipped_error(return_batch, value_batch, old_value_batch, self.eps_clip)
        return value_loss * 0.5

    def clipped_error(self, target, estimate, old_estimate, clip):
        clipped = old_estimate + (estimate - old_estimate).clamp(-clip, clip)
        surr1 = (estimate - target).pow(2)
        surr2 = (clipped - target).pow(2)
        loss = torch.max(surr1, surr2).mean()
        return loss

    def bpo_loss(self, sample):
        obs_batch, hidden_state_batch, act_batch, done_batch, \
            old_hard_adv_batch, old_value_batch, return_batch, adv_batch = sample
        mask_batch = (1 - done_batch)
        dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)
        log_prob_act_batch = dist_batch.log_prob(act_batch)

        if self.uniform_value:
            value_loss = self.uniform_policy_value_loss(act_batch, dist_batch, log_prob_act_batch, old_value_batch,
                                                        return_batch, value_batch)
        else:
            value_loss = self.clipped_bellman_error(old_value_batch, return_batch, value_batch)

        # Hard Boltzman Loss
        hard_adv_hat = self.inv_temp * (log_prob_act_batch + dist_batch.entropy())#dist_batch.probs.log().mean(dim=-1)
        pi_loss = self.clipped_error(adv_batch, hard_adv_hat, old_hard_adv_batch, self.bpo_clip)

        # enforce that hard_adv expectation is zero?

        l1_reg = sum([param.abs().sum() for param in self.policy.parameters()])

        # Policy Entropy (unchanged)
        entropy_loss = dist_batch.entropy().mean()

        # Final Loss
        loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss + l1_reg * self.l1_coef
        return entropy_loss, l1_reg, loss, pi_loss, value_loss

    def uniform_policy_value_loss(self, act_batch, dist_batch, log_prob_act_batch, old_value_batch, return_batch,
                                  value_batch):
        # Compute log probabilities for uniform policy
        action_space_size = dist_batch.probs.shape[-1]  # Assuming categorical action space
        log_uniform_prob = -torch.log(torch.tensor(action_space_size, dtype=torch.float32, device=act_batch.device))
        # Compute importance weights
        log_importance_weights = log_uniform_prob - log_prob_act_batch
        importance_weights = torch.exp(log_importance_weights)
        # Clip importance weights
        clipped_weights = torch.clamp(importance_weights, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
        # Clipped Bellman-Error (modified for uniform policy)
        clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
        v_surr1 = clipped_weights * (value_batch - return_batch).pow(2)
        v_surr2 = clipped_weights * (clipped_value_batch - return_batch).pow(2)
        value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()
        return value_loss

    def train(self, num_timesteps, num_bpo_timesteps=None):
        if num_bpo_timesteps is None:
            num_bpo_timesteps = num_timesteps
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, *self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, *self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, log_prob_act_v, value_v, next_hidden_state_v = self.predict(obs_v, hidden_state_v, done_v)
                    next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_v)
                    self.storage_valid.store(obs_v, hidden_state_v, act_v,
                                             rew_v, done_v, info_v,
                                             log_prob_act_v, value_v)
                    obs_v = next_obs_v
                    hidden_state_v = next_hidden_state_v
                _, _, last_val_v, hidden_state_v = self.predict(obs_v, hidden_state_v, done_v)
                self.storage_valid.store_last(obs_v, hidden_state_v, last_val_v)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            loss_fn = self.bpo_loss if self.t < num_bpo_timesteps else self.ppo_loss
            summary = self.optimize(loss_fn)
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
