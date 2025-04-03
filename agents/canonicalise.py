import os
import time

import pandas as pd
import wandb
from torch import nn

from common import orthogonal_init
from common.env.procgen_wrappers import get_action_names
from helper_local import norm_funcs, dist_funcs, plot_values_ascender, remove_duplicate_actions
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
                 q_model=None,
                 q_model_val=None,
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
                 soft_canonicalisation=True,
                 load_value_models=False,
                 meg=False,
                 remove_duplicate_actions=True,
                 centered_logprobs=False,
                 adjust_logprob_mean=False,
                 infinite_value=True,
                 meg_version="direct",
                 pirc=True,
                 meg_ground_next=True,
                 consistency_coef=10., **kwargs):

        super(Canonicaliser, self).__init__(env, policy, logger, storage, device,
                                            n_checkpoints, env_valid, storage_valid)
        self.consistency_coef = consistency_coef
        self.meg_ground_next = meg_ground_next
        self.pirc = pirc
        self.infinite_value = infinite_value
        self.adjust_logprob_mean = adjust_logprob_mean
        self.centered_logprobs = centered_logprobs
        self.remove_duplicate_actions = remove_duplicate_actions
        self.n_actions = self.env.action_space.n
        self.meg = meg
        if self.meg:
            if meg_version == "direct":
                self.meg_version = self.meg_v2_direct
            elif meg_version == "original":
                self.meg_version = self.meg_v1
            elif meg_version == "critic":
                self.meg_version = self.meg_v3
            elif meg_version == "kldiv":
                self.meg_version = self.kl_div_meg
        self.load_value_models = load_value_models
        self.soft_adv = soft_canonicalisation
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
        self.q_model = q_model
        self.q_model_val = q_model_val
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

        if self.meg:
            self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=learning_rate, eps=1e-5)
            self.q_optimizer_val = optim.Adam(self.q_model_val.parameters(), lr=learning_rate, eps=1e-5)
        else:
            self.q_optimizer = self.q_optimizer_val = None
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
        self.logvaldir = os.path.join(self.logger.logdir, "val_models")
        self.logmegdir = os.path.join(self.logger.logdir, "meg")
        for d in [self.logvaldir, self.logmegdir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def predict_subject_adv(self, obs, act, hidden_state, done, subject_policy):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            act = torch.FloatTensor(act).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = subject_policy(obs, hidden_state, mask)
            logp_eval_policy = dist.log_prob(act)
            if not self.soft_adv:
                if self.centered_logprobs:
                    return (logp_eval_policy - dist.probs.log().mean(dim=-1)).cpu().numpy()
                # converting log pi to implied hard advantage func:
                logp_exp = (dist.probs * dist.probs.log()).sum(dim=-1)
                return (logp_eval_policy - logp_exp).cpu().numpy(), dist.probs.cpu().numpy()
        return logp_eval_policy.cpu().numpy(), dist.probs.cpu().numpy()

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
            # summary = self.optimize()
            summary = self.dummy_optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = is_return_v = pdwis_return_v = None
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

        if self.adjust_logprob_mean:
            self.storage_trusted.translate_logp_mean_to_reward_mean()
            self.storage_trusted_val.translate_logp_mean_to_reward_mean()

        if not self.load_value_models:
            self.optimize_value(self.storage_trusted, self.value_model, self.value_optimizer, "Training")
            self.optimize_value(self.storage_trusted_val, self.value_model_val, self.value_optimizer_val, "Validation")
        if self.pirc:
            self.optimize_value(self.storage_trusted, self.value_model_logp, self.value_optimizer_logp, "Training",
                                "logits")
            self.optimize_value(self.storage_trusted_val, self.value_model_logp_val, self.value_optimizer_logp_val,
                                "Validation", "logits")

        if self.meg:
            meg_train = self.optimize_meg(self.storage_trusted, self.q_model, self.q_optimizer, "Training")
            meg_valid = self.optimize_meg(self.storage_trusted_val, self.q_model_val, self.q_optimizer_val,
                                          "Validation")
        else:
            meg_train = meg_valid = np.nan

        wandb.log({
            "mean_returns": np.mean(self.storage.get_returns(self.gamma)),
            "val_mean_returns": np.mean(self.storage_valid.get_returns(self.gamma)),
        })

        # Remove:
        is_return, pdwis_return = self.storage_trusted.compute_off_policy_estimates(self.gamma)

        if self.pirc:
            with torch.no_grad():
                if self.print_ascent_rewards:
                    print("Train Env Rew:")
                df_train, dt = self.canonicalise_and_evaluate_efficient(self.storage_trusted, self.value_model,
                                                                        self.value_model_logp)
                if self.print_ascent_rewards:
                    print("Valid Env Rew:")
                df_valid, dv = self.canonicalise_and_evaluate_efficient(self.storage_trusted_val, self.value_model_val,
                                                                        self.value_model_logp_val)

                df_train["Env"] = "Train"
                df_valid["Env"] = "Valid"
                comb = pd.concat([df_train, df_valid])
                pivoted_df = comb.pivot(index=["Norm", "Metric"], columns="Env", values="Distance").reset_index()
                is_return, pdwis_return = self.storage_trusted.compute_off_policy_estimates(self.gamma)
                is_return_v, pdwis_return_v = self.storage_trusted.compute_off_policy_estimates(self.gamma)
        wandb.log({
                    "distances": wandb.Table(dataframe=comb),
                    "distances_pivoted": wandb.Table(dataframe=pivoted_df),
                    "L2_L2_Train": dt,
                    "L2_L2_Valid": dv,
                    "Meg_Train": meg_train,
                    "Meg_Valid": meg_valid,
                    "IS_Train": is_return,
                    "IS_Valid": is_return_v,
                    "PDWIS_Train": pdwis_return,
                    "PDWIS_Valid": pdwis_return_v,
                })

        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()

    def collect_rollouts(self, done, hidden_state, obs, storage, env, policy=None, subject_policy=None):
        logp_eval_policy = probs = None
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done, policy)
            if subject_policy is not None:
                logp_eval_policy, probs = self.predict_subject_adv(obs, act, hidden_state, done, subject_policy)
            next_obs, rew, done, info = env.step(act)
            storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value, logp_eval_policy, probs)
            obs = next_obs
            hidden_state = next_hidden_state
        value_batch = storage.value_batch[:self.n_steps]
        _, _, last_val, hidden_state = self.predict(obs, hidden_state, done, policy)
        storage.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

    def optimize_value(self, storage, value_model, value_optimizer, env_type, rew_type="reward"):
        logdir = os.path.join(self.logvaldir, env_type, rew_type)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        # grad_accumulation_steps = batch_size / self.mini_batch_size
        # grad_accumulation_cnt = 1

        if self.reset_rew_model_weights:
            value_model.apply(orthogonal_init)

        value_model.train()
        self.policy.eval()
        checkpoint_cnt = 0
        save_every = self.val_epoch // self.num_checkpoints
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

            for sample in generator:
                (obs_batch, nobs_batch, act_batch, done_batch,
                 old_log_prob_act_batch, old_value_batch, return_batch,
                 adv_batch, rew_batch, logp_eval_policy_batch, probs, indices) = sample
                value_batch = value_model(obs_batch).squeeze()
                next_value_batch = value_model(nobs_batch).squeeze()
                if rew_type == "reward":
                    R = rew_batch
                    term_value = 0
                elif rew_type == "logits":
                    R = logp_eval_policy_batch
                    term_value = self.inf_term_value() if self.infinite_value else 0
                else:
                    raise NotImplementedError
                target = R + self.gamma * (next_value_batch * (1 - done_batch) + term_value * done_batch)
                value_loss = nn.MSELoss()(target, value_batch)
                value_loss.backward()
                val_losses.append(value_loss.item())

            for sample in generator_valid:
                obs_batch_val, nobs_batch_val, act_batch_val, done_batch_val, \
                    old_log_prob_act_batch_val, old_value_batch_val, return_batch_val, adv_batch_val, rew_batch_val, logp_eval_policy_batch_val, probs, indices = sample
                with torch.no_grad():
                    value_batch_val = value_model(obs_batch_val).squeeze()
                    next_value_batch_val = value_model(nobs_batch_val).squeeze()
                    if rew_type == "reward":
                        R = rew_batch_val
                        term_value = 0
                    elif rew_type == "logits":
                        R = logp_eval_policy_batch_val
                        term_value = self.inf_term_value() if self.infinite_value else 0
                    else:
                        raise NotImplementedError
                    target = R + self.gamma * (
                            next_value_batch_val * (1 - done_batch_val) + term_value * done_batch_val)

                    value_loss_val = nn.MSELoss()(target, value_batch_val)

                val_losses_valid.append(value_loss_val.item())
            value_optimizer.step()
            value_optimizer.zero_grad()
            # grad_accumulation_cnt += 1
            # if e == self.val_epoch - 1:
            #     plot_values_ascender(self.logger.logdir, obs_batch, value_batch.detach(), e)
            if e > ((checkpoint_cnt + 1) * save_every) or e == self.val_epoch - 1:
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           logdir + '/model_' + str(e) + '.pth')
                checkpoint_cnt += 1
            wandb.log({
                f'Loss/value_epoch_{env_type}': e,
                f'Loss/value_loss_{rew_type}_{env_type}': np.mean(val_losses),
                f'Loss/value_loss_valid_{rew_type}_{env_type}': np.mean(val_losses_valid),
            })

    def optimize_meg(self, storage, q_model, optimizer, env_type):
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size

        if self.reset_rew_model_weights:
            q_model.apply(orthogonal_init)

        max_ent = np.log(1 / self.n_actions)
        q_model.train()
        self.policy.eval()
        checkpoint_cnt = 0
        save_every = self.val_epoch // self.num_checkpoints
        for e in range(self.val_epoch):
            # TODO: play around with this:
            self.current_consistency_coef = (1.05 ** (e + 1)) / (1.05 ** self.val_epoch) * self.consistency_coef
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
            losses = {"loss": [], "meg_loss": [], "consistency_loss": []}
            losses_valid = {"loss": [], "meg_loss": [], "consistency_loss": []}
            megs = []

            for sample in generator:
                meg, elementwise_meg = self.meg_version(losses, max_ent, q_model, sample, valid=False)
                indices = sample[-1]
                storage.store_meg(elementwise_meg, indices, valid_envs=self.n_val_envs)
                megs.append(meg.item())
            for sample in generator_valid:
                _, elementwise_meg = self.meg_version(losses_valid, max_ent, q_model, sample, valid=True)

            full_meg = storage.full_meg(self.gamma, self.n_val_envs)

            optimizer.step()
            optimizer.zero_grad()
            # grad_accumulation_cnt += 1
            if e > ((checkpoint_cnt + 1) * save_every) or e == self.val_epoch - 1:
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           self.logmegdir + '/model_' + str(e) + '.pth')
                checkpoint_cnt += 1
            wandb.log({
                f'Loss/value_epoch_{env_type}': e,
                f'Loss/full_meg_loss_{env_type}': np.mean(losses["loss"]),
                f'Loss/full_meg_loss_valid_{env_type}': np.mean(losses_valid["loss"]),
                f'Loss/meg_loss_{env_type}': np.mean(losses["meg_loss"]),
                f'Loss/meg_loss_valid_{env_type}': np.mean(losses_valid["meg_loss"]),
                f'Loss/consistency_loss_{env_type}': np.mean(losses["consistency_loss"]),
                f'Loss/consistency_loss_valid_{env_type}': np.mean(losses_valid["consistency_loss"]),
                f'Loss/mean_meg_{env_type}': np.mean(megs),
                f'Loss/full_meg_{env_type}': full_meg.item(),
            })
        return np.mean(megs)

    def meg_v1(self, losses, max_ent, q_model, sample, valid=False):
        (obs_batch, nobs_batch, act_batch, done_batch,
         old_log_prob_act_batch, old_value_batch, return_batch,
         adv_batch, rew_batch, logp_eval_policy_batch, pi_subject, indices) = sample
        if self.meg_ground_next:
            obs = nobs_batch
        else:
            obs = obs_batch
        if not valid:
            # Forcing it to be function of next state
            q_value_batch = q_model(obs)
            value_batch = q_value_batch.logsumexp(dim=-1)
            q_selected = q_value_batch[torch.arange(len(act_batch)), act_batch.to(torch.int64)]
            target = logp_eval_policy_batch + value_batch - q_selected
            loss = ((target * (1 - done_batch)) ** 2).mean()
            loss.backward()
            losses.append(loss.item())
            log_pi_soft_star = q_value_batch.log_softmax(dim=-1)
            meg = (pi_subject * (log_pi_soft_star - max_ent)).sum(dim=-1).mean()
            return meg
        with torch.no_grad():
            q_value_batch_val = q_model(obs)
            value_batch_val = q_value_batch_val.logsumexp(dim=-1)
            q_selected = q_value_batch_val[torch.arange(len(act_batch)), act_batch.to(torch.int64)]
            loss_val = ((logp_eval_policy_batch + value_batch_val - q_selected) ** 2).mean()
            losses.append(loss_val.item())
            return None

    def meg_v2_direct(self, losses, max_ent, q_model, sample, valid=False):
        (obs_batch, nobs_batch, act_batch, done_batch,
         old_log_prob_act_batch, old_value_batch, return_batch,
         adv_batch, rew_batch, logp_eval_policy_batch, pi_subject, indices) = sample
        if self.meg_ground_next:
            obs = nobs_batch
            flt = (1 - done_batch).unsqueeze(dim=-1)
        else:
            obs = obs_batch
            flt = 1
        if not valid:
            q_value_batch = q_model(obs)
            value_batch = q_value_batch.logsumexp(dim=-1).unsqueeze(dim=-1)
            meg = (pi_subject * (q_value_batch - value_batch - max_ent) * flt).sum(dim=-1).mean()
            loss = -meg
            loss.backward()
            losses.append(loss.item())
            return meg
        with torch.no_grad():
            q_value_batch = q_model(obs)
            value_batch = q_value_batch.logsumexp(dim=-1).unsqueeze(dim=-1)
            meg = (pi_subject * (q_value_batch - value_batch - max_ent) * flt).sum(dim=-1).mean()
            losses.append(-meg.item())
            return None

    def meg_v3(self, losses, max_ent, q_model, sample, valid=False):
        raise NotImplementedError
        # (obs_batch, nobs_batch, act_batch, done_batch,
        #  old_log_prob_act_batch, old_value_batch, return_batch,
        #  adv_batch, rew_batch, logp_eval_policy_batch, pi_subject, indices) = sample
        # obs = nobs_batch
        # if not valid:
        #     adv = logp_eval_policy_batch - (pi_subject * pi_subject.log()).sum(dim=-1)
        #     q_target = adv + old_value_batch
        #     q_value_batch = q_model(obs)
        #     loss = (q_value_batch - q_target).pow(2).mean()
        #     # oops, maybe not such a good idea actually.
        #     meg = (pi_subject * (q_value_batch - max_ent)).mean()
        #     loss.backward()
        #     losses.append(loss.item())
        #     return meg
        # with torch.no_grad():
        #     q_value_batch = q_model(obs)
        #     value_batch = q_value_batch.logsumexp(dim=-1).unsqueeze(dim=-1)
        #     meg = (pi_subject * (q_value_batch - value_batch - max_ent) * flt).sum(dim=-1).mean()
        #     losses.append(-meg.item())
        #     return None

    def kl_div_meg(self, losses, max_ent, q_model, sample, valid=False):
        (obs_batch, nobs_batch, act_batch, done_batch,
         old_log_prob_act_batch, old_value_batch, return_batch,
         adv_batch, rew_batch, logp_eval_policy_batch, pi_subject, indices) = sample

        def generate_loss_meg():
            q_value_batch, _ = q_model(obs_batch)
            q_taken = q_value_batch[torch.arange(len(act_batch)), act_batch.to(torch.int64)]
            _, next_q = q_model(nobs_batch)
            log_pi_star = q_value_batch.log_softmax(dim=-1)
            meg = (pi_subject * (log_pi_star - max_ent)).mean()  # TODO: sum?
            loss1 = -meg
            loss2 = ((q_taken - self.gamma * next_q) * (1 - done_batch)).pow(2).mean()
            loss = loss1 + loss2 * self.current_consistency_coef
            # self.full_meg(done_batch, pi_subject, log_pi_star, max_ent)
            elementwise_meg = (pi_subject * (log_pi_star - max_ent)).sum(dim=-1)
            losses["loss"].append(loss.item())
            losses["meg_loss"].append(loss1.item())
            losses["consistency_loss"].append(loss2.item())
            return loss, meg, elementwise_meg

        if not valid:
            loss, meg, elementwise_meg = generate_loss_meg()
            loss.backward()
            return meg, elementwise_meg
        with torch.no_grad():
            loss, meg, elementwise_meg = generate_loss_meg()
            return None, elementwise_meg

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
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, _, _, _, _ = sample
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
        obs_batch, nobs_batch, act_batch, done_batch, _, _, _, _, rew_batch, _, _, _ = sample
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

        tuples = [self.sample_and_canonicalise(sample, value_model, value_model_logp) for sample in generator]
        logp_batch, rew_batch, adj_batch, adj_batch_logp = zip(*tuples)
        logp = torch.concat(list(logp_batch))
        rew = torch.concat(list(rew_batch))
        adj = torch.concat(list(adj_batch))
        adj_logp = torch.concat(list(adj_batch_logp))
        canon_logp = logp + adj_logp

        canon_true_r = rew + adj

        # This is useful to see why there is a gap (in ascender at least).
        # torch.stack((logp, rew)).unique(dim=1).T

        data = []
        d = np.nan
        for norm_name, normalize in self.norm_funcs.items():
            for dist_name, distance in self.dist_funcs.items():
                dist = distance(normalize(canon_logp), normalize(canon_true_r))
                data.append({'Norm': norm_name, 'Metric': dist_name, 'Distance': dist.item()})
                if norm_name == "l2_norm" and dist_name == "l2_dist":
                    d = dist.item()
        return pd.DataFrame(data), d

    def sample_and_canonicalise(self, sample, value_model, value_model_logp):
        obs_batch, nobs_batch, act_batch, done_batch, _, _, _, _, rew_batch, logp_batch, _, _ = sample
        dist, _, _ = self.policy.forward_with_embedding(obs_batch)
        if self.remove_duplicate_actions:
            try:
                dist, act_batch = remove_duplicate_actions(dist, act_batch.to(torch.int32), self.env)
            except Exception as e:
                pass

        val_batch = value_model(obs_batch).squeeze()
        next_val_batch = value_model(nobs_batch).squeeze()
        # logp_batch = dist.log_prob(act_batch)

        val_batch_logp = value_model_logp(obs_batch).squeeze()
        next_val_batch_logp = value_model_logp(nobs_batch).squeeze()
        # N.B. This is for uniform policy, but probably makes sense for any policy.
        # term_value = (1 / (1 - self.gamma)) * np.log(dist.logits.shape[-1])
        term_value = self.inf_term_value() if self.infinite_value else 0
        next_val_batch[done_batch.bool()] = 0
        adjustment = self.gamma * next_val_batch - val_batch
        next_val_batch_logp[done_batch.bool()] = term_value
        adjustment_logp = self.gamma * next_val_batch_logp - val_batch_logp

        # if not self.soft_adv:
        #     if self.centered_logprobs:
        #         logp_batch -= dist.probs.log().mean(dim=-1)
        #     else:
        #         # make it hard advantage func:
        #         logp_exp = (dist.probs * dist.probs.log()).sum(dim=-1)
        #         logp_batch -= logp_exp
        # normalize = self.norm_funcs["l2_norm"]
        # distance = self.dist_funcs["l2_dist"]
        # print(distance(normalize(logp_batch+adjustment_logp), normalize(rew_batch+adjustment)).item())
        return logp_batch, rew_batch, adjustment, adjustment_logp
        # For cartpole - plots angle against val models
        import matplotlib.pyplot as plt
        plt.scatter(obs_batch[:, 2].cpu().numpy(), (val_batch).cpu().numpy())
        plt.scatter(obs_batch[:, 2].cpu().numpy(), (val_batch_logp).cpu().numpy())
        plt.show()

    def dummy_optimize(self):
        return {
            'Loss/total': np.nan,
            'Loss/pi': np.nan,
            'Loss/v': np.nan,
            'Loss/entropy': np.nan,
            'Loss/l1_reg': np.nan,
        }

    def inf_term_value(self):
        return (1 / (1 - self.gamma)) * np.log(self.n_actions)
