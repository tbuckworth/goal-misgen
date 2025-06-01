from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
import numpy as np


class LatentDiffusion(BaseAgent):
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
                 reward_termination=None,
                 alpha_max_ent=0.,
                 diffusion_policy=None,
                 **kwargs):

        super(LatentDiffusion, self).__init__(env, policy, logger, storage, device,
                                              n_checkpoints, env_valid, storage_valid)

        self.diffusion_policy = diffusion_policy
        self.ddpm = self.diffusion_policy.diffusion_model
        self.alpha_max_ent = alpha_max_ent
        self.reward_termination = reward_termination
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
        self.optimizer = optim.Adam(self.ddpm.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, policy):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            dist, value, latents = policy.forward_with_latents(obs)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), latents.cpu().numpy(), dist.entropy().cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list, l1_reg_list, total_loss_list = [], [], [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.ddpm.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample

                latents = hidden_state_batch  # TODO: figure this out and rename
                x0 = latents.to(self.device)
                t = torch.randint(0, self.ddpm.n_steps, (x0.size(0),),
                                  device=self.device, dtype=torch.long)

                loss = self.ddpm.p_losses(x0, t)

                l1_reg = sum([param.abs().sum() for param in self.ddpm.parameters()])

                # Let ddpm to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                l1_reg_list.append(l1_reg.item())
                total_loss_list.append(loss.item() * x0.size(0))

        summary = {
            'Loss/total': np.mean(total_loss_list),
            'Loss/l1_reg': np.mean(l1_reg_list)
        }
        return summary

    def train(self, num_timesteps):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            self.collect_rollouts(obs, self.env, self.storage, self.policy)

            # valid
            if self.env_valid is not None:
                self.collect_rollouts(obs_v, self.env_valid, self.storage_valid, self.diffusion_policy)

            # Optimize policy & value
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)
            mean_episode_rewards = self.logger.dump(summary)
            premature_finish = self.reward_termination is not None and self.reward_termination <= mean_episode_rewards
            if self.anneal_lr:
                self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every) or premature_finish:
                print("Saving model.")
                torch.save({
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
                if premature_finish:
                    break
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()

    def collect_rollouts(self, obs, env, storage, policy):
        for _ in range(self.n_steps):
            act, log_prob_act, value, latents, entropy = self.predict(obs, policy)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, latents, act, rew, done, info, log_prob_act, value,
                          entropy=entropy * self.alpha_max_ent)
            obs = next_obs
        _, _, last_val, latents, entropy = self.predict(obs, policy)
        storage.store_last(obs, latents, last_val)
        # Compute advantage estimates
        storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
