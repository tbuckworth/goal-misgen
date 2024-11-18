import pandas as pd
import wandb
from torch import nn

from common import orthogonal_init
from common.policy import UniformPolicy
from .base_agent import BaseAgent
from common.misc_util import adjust_lr
import torch
import torch.optim as optim
import numpy as np


class Canoncicaliser(BaseAgent):
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
                 val_model=None,
                 inv_temp_rew_model=1.,
                 next_rew_loss_coef=1.,
                 storage_trusted=None,
                 storage_trusted_val=None,
                 rew_lr=1e-5,
                 reset_rew_model_weights=False,
                 rew_learns_from_trusted_rollouts=False,

                 **kwargs):

        super(Canoncicaliser, self).__init__(env, policy, logger, storage, device,
                                             n_checkpoints, env_valid, storage_valid)

        self.rew_learns_from_trusted_rollouts = rew_learns_from_trusted_rollouts
        self.reset_rew_model_weights = reset_rew_model_weights
        self.print_ascent_rewards = True
        self.trusted_policy = UniformPolicy(policy.action_size, device)
        self.next_rew_loss_coef = next_rew_loss_coef
        self.inv_temp = inv_temp_rew_model
        self.val_model = val_model
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
        self.value_optimizer = optim.Adam(self.val_model.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.num_rew_updates = num_rew_updates
        self.storage_trusted = storage_trusted
        self.storage_trusted_val = storage_trusted_val

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

    def train(self, num_timesteps):
        learn_rew_every = num_timesteps // self.num_rew_updates
        rew_checkpoint_cnt = 0
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0

        # Collect supervised data for unshifted env
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        self.collect_rollouts(done, hidden_state, obs, self.storage_trusted, self.env,
                              self.trusted_policy)
        # Need to re-do this, so it's fresh for the env data collection:
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        # Collect supervised data for shifted env
        obs_v = self.env_valid.reset()
        hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done_v = np.zeros(self.n_envs)
        self.collect_rollouts(done_v, hidden_state_v, obs_v, self.storage_trusted_val, self.env_valid,
                              self.trusted_policy)
        # Need to re-do this, so it's fresh for the valid env data collection:
        obs_v = self.env_valid.reset()
        hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done_v = np.zeros(self.n_envs)

        # Run Policy just to measure reward:
        self.policy.eval()
        self.collect_rollouts(done, hidden_state, obs, self.storage, self.env)
        self.collect_rollouts(done_v, hidden_state_v, obs_v, self.storage_valid, self.env_valid)
        rew_batch, done_batch = self.storage.fetch_log_data()
        rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
        self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)

        self.optimize_value()

        with torch.no_grad():
            if self.print_ascent_rewards:
                print("Train Env Rew:")
            df_train = self.canonicalise_and_evaluate(self.storage_trusted)
            if self.print_ascent_rewards:
                print("Valid Env Rew:")
            df_valid = self.canonicalise_and_evaluate(self.storage_trusted_val)

            joined_df = pd.merge(df_train, df_valid, on=["ID"])

            # Convert the joined DataFrame back to a W&B Table
            joined_table = wandb.Table(dataframe=joined_df)
        # log_data = {
        #     "timesteps": self.t,
        #     "dist_train": 0,
        #     "dist_valid": dist_valid,
        # }
        # log_data.update(summary)
        # wandb.log(log_data)
        # rew_checkpoint_cnt += 1

        # Save the model
        if self.t > ((checkpoint_cnt + 1) * save_every):
            print("Saving model.")
            torch.save({'model_state_dict': self.policy.state_dict(),
                        'optimizer_state_dict': self.value_optimizer.state_dict()},
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

    def optimize_value(self):

        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        if self.reset_rew_model_weights:
            self.val_model.apply(orthogonal_init)

        self.val_model.train()
        self.policy.eval()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            storage = self.storage_trusted_val
            generator = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                      recurrent=recurrent)
            val_losses = []
            for sample in generator:
                obs_batch, nobs_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, rew_batch = sample

                value_batch = self.val_model(obs_batch)
                next_value_batch = self.val_model(nobs_batch)

                target = rew_batch + self.gamma * next_value_batch * (1 - done_batch)

                value_loss = nn.MSELoss()(target - value_batch)

                value_loss.backward()

                val_losses.append(value_loss.item())

            self.value_optimizer.step()
            self.value_optimizer.zero_grad()
            grad_accumulation_cnt += 1
            wandb.log({
                'Loss/epoch': e,
                'Loss/val_loss': np.mean(val_losses),
            })

    def canonicalise_and_evaluate(self, storage):
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size

        self.policy.eval()
        self.val_model.eval()

        recurrent = self.policy.is_recurrent()
        generator = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                  recurrent=recurrent)
        rew_tuples = [self.sample_next_data(sample) for sample in generator]
        rew, obs, act, value, next_value, log_probs, dones = zip(*rew_tuples)

        val_batch = torch.concat(list(value))
        next_val_batch = torch.concat(list(next_value))
        rew_batch = torch.concat(list(rew))
        obs_batch = torch.concat(list(obs))
        act_batch = torch.concat(list(act))
        done_batch = torch.concat(list(dones))
        logp_batch = torch.concat(list(log_probs))

        adjustment = self.gamma * next_val_batch * (1 - done_batch) - val_batch
        canon_logp = logp_batch + adjustment
        canon_true_r = rew_batch + adjustment

        norm_funcs = {
            "l1_norm": lambda x: x / x.abs().mean(),
            "l2_norm": lambda x: x / x.pow(2).mean().sqrt(),
            "linf_norm": lambda x: x / x.abs().max(),
        }

        dist_funcs = {
            "l1_dist": lambda x, y: norm_funcs["l1_norm"](x - y),
            "l2_dist": lambda x, y: norm_funcs["l1_norm"](x - y),
        }
        # dist_table = wandb.Table(columns=["Norm", "Metric", "Env", "Distance"])
        dists = {}
        for norm_name, normalize in norm_funcs.items():
            for dist_name, distance in dist_funcs.items():
                if norm_name not in dists.keys():
                    dists[norm_name] = {}
                dist = distance(normalize(canon_logp), normalize(canon_true_r))
                dists[norm_name][dist_name] = dist
                # dist_table.add_data(norm_name, dist_name, env_type, dist)

        # if self.norm_func == "l1":
        #     def normalize(arr):
        #         l1 = arr.abs().sum()
        #         return arr / l1
        # elif self.norm_func == "l2":
        #     def normalize(arr):
        #         l2 = arr.pow(2).sum().sqrt()
        #         return arr / l2
        # elif self.norm_func == "linf":
        #     def normalize(arr):
        #         linf = arr.abs().max()[0]
        #         return arr / linf
        # else:
        #     raise NotImplementedError(f"{self.norm_func} not implemented")
        #
        # norm_c_logp = normalize(canon_logp)
        # norm_c_true_r = normalize(canon_true_r)
        #
        # if self.distance_metric == "corr":
        #     def distance(arr1, arr2):
        #         return torch.corrcoef(torch.stack((arr1, arr2)))[0, 1]
        # elif self.distance_metric == "l1":
        #     def distance(arr1, arr2):
        #         return (arr1 - arr2).abs().mean()
        # elif self.distance_metric == "l2":
        #     def distance(arr1, arr2):
        #         return (arr1, arr2).pow(2).mean().sqrt()
        # else:
        #     raise NotImplementedError()
        #
        # dist = distance(norm_c_logp, norm_c_true_r)

        if self.print_ascent_rewards:
            unq_obs_rew = torch.concat(
                (
                    obs_batch[..., (0, 2, 4)],
                    act_batch.unsqueeze(-1),
                    norm_c_logp,
                    norm_c_true_r
                ),
                dim=-1).unique(dim=0)
            df = pd.DataFrame(data=unq_obs_rew.detach().cpu().numpy(),
                              columns=["Left", "Middle", "Right", "Act", "normC(log p)", "normC(Rew)"])
            int_cols = ["Left", "Middle", "Right", "Act", "normC(Rew)"]
            float_cols = ["normC(log p)"]
            df[int_cols] = df[int_cols].astype(np.int64)
            df[float_cols] = df[float_cols].round(decimals=2)
            print(df)

        return pd.DataFrame(dists, columns=["Norm", "Metric", "Distance"])

    def sample_next_data(self, sample):
        obs_batch, nobs_batch, act_batch, done_batch, _, _, _, _, rew_batch = sample
        dist, _, _ = self.policy.forward_with_embedding(obs_batch)
        value = self.val_model(obs_batch)
        next_value = self.val_model(nobs_batch)

        return rew_batch, obs_batch, act_batch, value, next_value, dist.log_prob(act_batch), done_batch
