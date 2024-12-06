import pandas as pd
import wandb
from torch import nn

from common import orthogonal_init
from common.policy import UniformPolicy, CraftedTorchPolicy
from helper_local import norm_funcs, dist_funcs
from .base_agent import BaseAgent
from common.misc_util import adjust_lr
import torch
import torch.optim as optim
import numpy as np


class Canonicaliser(BaseAgent):
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
                 value_model=None,
                 value_model_val=None,
                 value_model_logp=None,
                 value_model_logp_val=None,
                 val_epoch=100,
                 inv_temp_rew_model=1.,
                 next_rew_loss_coef=1.,
                 storage_trusted=None,
                 storage_trusted_val=None,
                 rew_lr=1e-5,
                 reset_rew_model_weights=False,
                 rew_learns_from_trusted_rollouts=False,
                 trusted_policy=None,
                 n_val_envs=0,
                 use_unique_obs=False,
                 adjust_terminal_values=True,
                 **kwargs):

        super(Canonicaliser, self).__init__(env, policy, logger, storage, device,
                                            n_checkpoints, env_valid, storage_valid)

        self.adjust_terminal_values = adjust_terminal_values
        self.use_unique_obs = use_unique_obs
        if n_val_envs >= n_envs:
            raise IndexError(f"n_val_envs:{n_val_envs} must be less than n_envs:{n_envs}")
        self.n_val_envs = n_val_envs
        self.val_epoch = val_epoch
        self.rew_learns_from_trusted_rollouts = rew_learns_from_trusted_rollouts
        self.reset_rew_model_weights = reset_rew_model_weights
        self.print_ascent_rewards = False
        self.trusted_policy = trusted_policy
        self.next_rew_loss_coef = next_rew_loss_coef
        self.inv_temp = inv_temp_rew_model
        self.value_model = value_model
        self.value_model_val = value_model_val
        self.value_model_logp = value_model_logp
        self.value_model_logp_val = value_model_logp_val
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
        if len([x for x in self.policy.parameters()]) > 0:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate, eps=1e-5)
        self.value_optimizer_val = optim.Adam(self.value_model_val.parameters(), lr=learning_rate, eps=1e-5)
        self.value_optimizer_logp = optim.Adam(self.value_model_logp.parameters(), lr=learning_rate, eps=1e-5)
        self.value_optimizer_logp_val = optim.Adam(self.value_model_logp_val.parameters(), lr=learning_rate, eps=1e-5)
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
        self.norm_funcs = norm_funcs
        self.dist_funcs = dist_funcs

    def predict_temp(self, obs, act, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            act = torch.FloatTensor(act).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            logp_eval_policy = dist.log_prob(act).cpu().numpy()
        return logp_eval_policy

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
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        # Collect supervised data for unshifted env
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        self.collect_rollouts(done, hidden_state, obs, self.storage_trusted, self.env,
                              self.trusted_policy, self.policy)
        # Need to re-do this, so it's fresh for the env data collection:
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        # Collect supervised data for shifted env
        obs_v = self.env_valid.reset()
        hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done_v = np.zeros(self.n_envs)
        self.collect_rollouts(done_v, hidden_state_v, obs_v, self.storage_trusted_val, self.env_valid,
                              self.trusted_policy, self.policy)
        # Need to re-do this, so it's fresh for the valid env data collection:
        obs_v = self.env_valid.reset()
        hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done_v = np.zeros(self.n_envs)

        # Traditional PPO:
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

            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every) and self.epoch > 0:
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1

        if False: # TODO: make this correct
            self.optimize_value(self.storage_trusted, self.value_model, self.value_optimizer, "Training")
            self.optimize_value(self.storage_trusted_val, self.value_model_val, self.value_optimizer_val, "Validation")
        self.optimize_value(self.storage_trusted, self.value_model_logp, self.value_optimizer_logp, "Training","logits")
        self.optimize_value(self.storage_trusted_val, self.value_model_logp_val, self.value_optimizer_logp_val, "Validation","logits")



        with torch.no_grad():
            if self.print_ascent_rewards:
                print("Train Env Rew:")
            df_train, dt = self.canonicalise_and_evaluate_efficient(self.storage_trusted, self.value_model, self.value_model_logp)
            if self.print_ascent_rewards:
                print("Valid Env Rew:")
            df_valid, dv = self.canonicalise_and_evaluate_efficient(self.storage_trusted_val, self.value_model_val, self.value_model_logp_val)

            df_train["Env"] = "Train"
            df_valid["Env"] = "Valid"
            comb = pd.concat([df_train, df_valid])
            pivoted_df = comb.pivot(index=["Norm", "Metric"], columns="Env", values="Distance").reset_index()
            wandb.log({
                "distances": wandb.Table(dataframe=comb),
                "distances_pivoted": wandb.Table(dataframe=pivoted_df),
                "L2_L2_Train": dt,
                "L2_L2_Valid": dv,
            })

        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()

    def collect_rollouts(self, done, hidden_state, obs, storage, env, policy=None, save_extra=False):
        logp_eval_policy = None
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, policy)
            if save_extra:
                logp_eval_policy = self.predict_temp(obs, act, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value, logp_eval_policy)
            obs = next_obs
            hidden_state = next_hidden_state
        value_batch = storage.value_batch[:self.n_steps]
        _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, policy)
        storage.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

    def optimize_value(self, storage, value_model, value_optimizer, env_type, rew_type="reward"):
        distance = self.dist_funcs["l2_dist"]
        normalize = self.norm_funcs["l2_norm"]
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        if self.reset_rew_model_weights:
            value_model.apply(orthogonal_init)

        value_model.train()
        self.policy.eval()
        for e in range(self.val_epoch):
            recurrent = self.policy.is_recurrent()
            generator = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                      recurrent=recurrent,
                                                      valid_envs=self.n_val_envs,
                                                      valid=False,
                                                      )
            generator_valid = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                            recurrent=recurrent,
                                                            valid_envs=self.n_val_envs,
                                                            valid=True,
                                                            )
            val_losses = []
            val_losses_valid = []
            # clp = []
            # ctr = []

            for sample in generator:
                (obs_batch, nobs_batch, act_batch, done_batch,
                 old_log_prob_act_batch, old_value_batch, return_batch,
                 adv_batch, rew_batch, logp_eval_policy_batch) = sample
                value_batch = value_model(obs_batch).squeeze()
                next_value_batch = value_model(nobs_batch).squeeze()
                if rew_type == "reward":
                    R = rew_batch
                elif rew_type == "logits":
                    if self.adjust_terminal_values:
                        R = logp_eval_policy_batch
                    else:
                        raise NotImplementedError("Need to get mean log pi across all the actions, so needs to be stored\n"
                                                  "Maybe just store log pi - mean(log pi) = advantage estimate")
                        R = logp_eval_policy_batch - logp_eval_policy_batch.mean(dim=-1)
                else:
                    raise NotImplementedError
                target = R + self.gamma * next_value_batch * (1 - done_batch)
                value_loss = nn.MSELoss()(target, value_batch)
                value_loss.backward()
                val_losses.append(value_loss.item())
                # with torch.no_grad():
                #     adjustment = self.gamma * next_value_batch * (1 - done_batch) - value_batch
                    # canon_logp = logp_eval_policy_batch + adjustment
                    # canon_true_r = R + adjustment
                    # clp.append(canon_logp)
                    # ctr.append(canon_true_r)

            for sample in generator_valid:
                obs_batch_val, nobs_batch_val, act_batch_val, done_batch_val, \
                    old_log_prob_act_batch_val, old_value_batch_val, return_batch_val, adv_batch_val, rew_batch_val, logp_eval_policy_batch_val = sample
                with torch.no_grad():
                    value_batch_val = value_model(obs_batch_val).squeeze()
                    next_value_batch_val = value_model(nobs_batch_val).squeeze()
                    if rew_type == "reward":
                        R = rew_batch_val
                    elif rew_type == "logits":
                        if self.adjust_terminal_values:
                            R = logp_eval_policy_batch_val
                        else:
                            raise NotImplementedError(
                                "Need to get mean log pi across all the actions, so needs to be stored\n"
                                "Maybe just store log pi - mean(log pi) = advantage estimate")
                            R = logp_eval_policy_batch_val - logp_eval_policy_batch_val.mean(dim=-1)
                    else:
                        raise NotImplementedError
                    target = R + self.gamma * next_value_batch_val * (1 - done_batch_val)

                    value_loss_val = nn.MSELoss()(target, value_batch_val)

                val_losses_valid.append(value_loss_val.item())
            value_optimizer.step()
            value_optimizer.zero_grad()
            grad_accumulation_cnt += 1

            # canon_logp = torch.concat(clp)
            # canon_true_r = torch.concat(ctr)
            # dist = distance(normalize(canon_logp), normalize(canon_true_r))

            wandb.log({
                f'Loss/value_epoch_{env_type}': e,
                f'Loss/value_loss_{env_type}': np.mean(val_losses),
                f'Loss/value_loss_valid_{env_type}': np.mean(val_losses_valid),
                # f'Loss/l2_normalized_l2_distance_{env_type}': dist.item(),
            })

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
                obs_batch, nobs_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, _, _ = sample
                mask_batch = (1 - done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, None, mask_batch)

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

                l1_reg = torch.concat([param.abs().reshape(-1) for param in self.policy.parameters()]).mean()

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

    def sample_next_data(self, sample):
        obs_batch, nobs_batch, act_batch, done_batch, _, _, _, _, rew_batch, _ = sample
        dist, _, _ = self.policy.forward_with_embedding(obs_batch)
        value = self.value_model(obs_batch)
        next_value = self.value_model(nobs_batch)

        return rew_batch, obs_batch, act_batch, value, next_value, dist.log_prob(act_batch), done_batch

    def canonicalise_and_evaluate_efficient(self, storage, value_model, value_model_logp):
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size

        self.policy.eval()
        value_model.eval()

        recurrent = self.policy.is_recurrent()
        if self.use_unique_obs:
            generator = storage.fetch_unique_generator(mini_batch_size=self.mini_batch_size)
        else:
            generator = storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                      recurrent=recurrent)

        tuples = [self.sample_and_canonise(sample, value_model, value_model_logp) for sample in generator]
        logp_batch, rew_batch, adj_batch, adj_batch_logp = zip(*tuples)
        logp = torch.concat(list(logp_batch))
        rew = torch.concat(list(rew_batch))
        adj = torch.concat(list(adj_batch))
        adj_logp = torch.concat(list(adj_batch_logp))

        exp_pi = logp.exp().exp()
        adv = exp_pi - exp_pi.mean()

        canon_logp = adv + adj_logp
        canon_true_r = rew + adj

        data = []
        d = np.nan
        for norm_name, normalize in self.norm_funcs.items():
            for dist_name, distance in self.dist_funcs.items():
                dist = distance(normalize(canon_logp), normalize(canon_true_r))
                data.append({'Norm': norm_name, 'Metric': dist_name, 'Distance': dist.item()})
                if norm_name == "l2_norm" and dist_name == "l2_dist":
                    d = dist.item()
        return pd.DataFrame(data), d

    def sample_and_canonise(self, sample, value_model, value_model_logp):
        obs_batch, nobs_batch, act_batch, done_batch, _, _, _, _, rew_batch, _ = sample
        dist, _, _ = self.policy.forward_with_embedding(obs_batch)
        val_batch = value_model(obs_batch).squeeze()
        next_val_batch = value_model(nobs_batch).squeeze()
        logp_batch = dist.log_prob(act_batch)

        val_batch_logp = value_model_logp(obs_batch).squeeze()
        next_val_batch_logp = value_model_logp(nobs_batch).squeeze()

        # N.B. This is for uniform policy, but probably makes sense for any policy.
        inf_term_value = (1 / (1 - self.gamma)) * np.log(1 / dist.logits.shape[-1])

        if self.adjust_terminal_values:
            next_val_batch[done_batch.bool()] = inf_term_value
            adjustment = self.gamma * next_val_batch - val_batch
            next_val_batch_logp[done_batch.bool()] = inf_term_value
            adjustment_logp = self.gamma * next_val_batch_logp - val_batch_logp
        else:
            # N.B. Rew is function of next states in our storage
            adjustment = self.gamma * next_val_batch * (1-done_batch) - val_batch
            # N.B. Rew is function of next states in our storage
            adjustment_logp = self.gamma * next_val_batch_logp * (1 - done_batch) - val_batch_logp

        return logp_batch, rew_batch, adjustment, adjustment_logp
