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
                 **kwargs):

        super(LatentDiffusion, self).__init__(env, policy, logger, storage, device,
                                              n_checkpoints, env_valid, storage_valid)

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
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
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

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy(), dist.entropy().cpu().numpy()


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

                #TODO: this is just boilerplate at this point:
                # 1.  Collect a buffer of latent vectors from your trained policy.
                latents = torch.stack(collected_latents)  # (N, latent_dim)

                # 2.  Build dataset / loader.
                dataset = LatentReplay(latents)
                loader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True)

                # 3.  Instantiate model + DDPM wrapper.
                latent_dim = latents.shape[1]
                net = LatentDiffusionModel(latent_dim)
                ddpm = DDPM(net)

                # 4.  Train.
                train_diffusion(ddpm, loader, epochs=20)

                # 5.  Plug into your DiffusionPolicy:
                policy = ...  # pre-trained RL policy
                diffusion_policy = DiffusionPolicy(policy, ddpm)




                l1_reg = sum([param.abs().sum() for param in self.policy.parameters()])

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = ...
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
            self.collect_rollouts(done, hidden_state, obs, self.env, self.storage)

            # valid
            if self.env_valid is not None:
                self.collect_rollouts(done_v, hidden_state_v, obs_v, self.env_valid, self.storage_valid)

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

    def collect_rollouts(self, done, hidden_state, obs, env, storage):
        #TODO: replace hidden state with latents
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state, entropy = self.predict(obs, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value,
                               entropy=entropy * self.alpha_max_ent)
            hidden_state = next_hidden_state
            obs = next_obs
            hidden_state = next_hidden_state
        value_batch = storage.value_batch[:self.n_steps]
        _, _, last_val, hidden_state, entropy = self.predict(obs, hidden_state, done)
        storage.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
