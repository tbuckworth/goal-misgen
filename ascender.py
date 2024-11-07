from matplotlib import pyplot as plt
import numpy as np
from scipy.special import softmax, log_softmax
import torch

from common.model import MlpModel, MlpModelNoFinalRelu

gamma = 0.99
class AscentEnv():
    def __init__(self, shifted=False, n_states=5):
        self.shifted = shifted
        self.n_states = n_states
        self.states = np.arange(n_states)
        self.state = 0

    def obs(self, state):
        if state == self.n_states - 1:
            return np.zeros((6,))
        return np.concatenate([self.features(state - 1), self.features(state), self.features(state + 1)])

    def features(self, state):
        if state < 0 or state >= self.n_states:
            return -np.ones((2,))
        if not self.shifted:
            return np.full((2,), state + 1)
        return np.array((state + 1, self.n_states - state))

    def reward(self, state):
        if state == self.n_states - 1:
            return 10
        return 0

    def done(self, state):
        if state == self.n_states - 1:
            return True
        return False

    def step(self, action):
        self.state = min(max(self.state + action, 0), self.n_states - 1)

        obs, rew, done, info = self.observe()
        if done:
            self.reset()
        return obs, rew, done, info

    def reset(self):
        self.state = 0
        return self.obs(self.state)

    def observe(self):
        return self.obs(self.state), self.reward(self.state), self.done(self.state), {}


class Policy():
    def __init__(self, misgen=False):
        self.embedder = np.zeros((6, 3))
        self.actor = np.zeros((3, 2))
        self.actor[0, 0] = 1.
        self.actor[2, 1] = 1.
        if not misgen:
            self.embedder[0, 0] = 1.
            self.embedder[2, 1] = 1.
            self.embedder[4, 2] = 1.
        else:
            self.embedder[1, 0] = 1.
            self.embedder[3, 1] = 1.
            self.embedder[5, 2] = 1.

    def embed(self, obs):
        return obs @ self.embedder

    def forward(self, obs, embed=True):
        if embed:
            h = self.embed(obs)
        else:
            h = obs
        logits = h @ self.actor
        return log_softmax(logits, axis=-1)

    def act(self, obs, embed=True):
        logits = self.forward(obs, embed=embed)
        p = np.exp(logits)
        return np.random.choice([-1, 1], p=p)


class LearnablePolicy():
    def __init__(self, device):
        self.device = device
        self.embedder = torch.nn.Parameter(torch.randn((6, 3)).to(self.device), requires_grad=True)
        self.actor = torch.nn.Parameter(torch.randn((3, 2)).to(self.device), requires_grad=True)
        self.params = [self.embedder, self.actor]

    def embed(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
            return (obs @ self.embedder).detach().cpu().numpy()
        return obs @ self.embedder

    def forward(self, obs, embed=True):
        if embed:
            h = self.embed(obs)
        else:
            h = obs
        logits = h @ self.actor
        return logits.log_softmax(dim=-1)

    def act(self, obs, embed=True):
        obs = torch.FloatTensor(obs).to(self.device)
        logits = self.forward(obs, embed=embed)
        p = logits.exp().detach().cpu().numpy()
        return np.random.choice([-1, 1], p=p)


def concat_data(Obs, obs):
    obs = np.array(obs)
    if Obs is None:
        return np.expand_dims(obs, axis=0)
    return np.concatenate((Obs, np.expand_dims(obs, axis=0)), axis=0)

def implicit_policy_learning(verbose=False, gamma=gamma, epochs=1000, sub_epochs=10, learning_rate=1e-3, inv_temp=1, l1_coef=0.):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = AscentEnv(shifted=False)
    policy = LearnablePolicy(device)

    for epoch in range(epochs):
        Actions, Done, Nactions, Nobs, Obs, Rew = collect_rollouts(env, policy, False, 1000)

        critic = MlpModelNoFinalRelu(input_dims=3, hidden_dims=[256, 256, 1])
        

        critic.to(device)

        optimizer = torch.optim.Adam(policy.params + list(critic.parameters()), lr=learning_rate)

        obs = torch.FloatTensor(Obs).to(device)
        next_obs = torch.FloatTensor(Nobs).to(device)
        acts = torch.FloatTensor(Actions).to(device)
        rew = torch.FloatTensor(Rew).to(device)
        flt = Done

        action_idx = [tuple(n for n in range(len(acts))), tuple(1 if x == 1 else 0 for x in acts)]
        

        # log_prob_acts *= inv_temp

        losses = []

        for sub_epoch in range(sub_epochs):
            val = critic(obs)
            next_val = critic(next_obs)
            log_probs = policy.forward(obs, embed=False)
            adv = log_probs[action_idx]
            
            rew_hat_dones = adv[flt] + val[flt] - gamma * next_val[flt]

            rew_hat = adv[~flt] + val[~flt] - gamma * next_val[~flt]


            loss = torch.nn.MSELoss()(rew[~flt], rew_hat)
            loss2 = torch.nn.MSELoss()(rew[flt], rew_hat_dones)
            
            coef = flt.mean()
            l1_loss = 0
            for param in policy.params:
                l1_loss += torch.sum(torch.abs(param))
                

            total_loss = (1 - coef) * loss + coef * loss2 + l1_loss * l1_coef


            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            losses.append(total_loss.item())
            if sub_epoch % 1 == 0 and verbose:
                print(f"\tEpoch:{epoch}\tSub_Epoch:{sub_epoch}\tLoss:{total_loss.item():.4f}\tL1:{l1_loss.item():.2f}")
        print(f"Reward:{Rew.mean():.2f}")
    return policy
        # if epoch % 100 == 0 and verbose:
        #     print(f"Epoch:{epoch}\tLoss:{total_loss.item():.4f}\tL1:{l1_loss.item():.2f}")



def inverse_reward_shaping(shifted=False, misgen=False, verbose=False, gamma=gamma, epochs=5000, learning_rate=1e-3, inv_temp=1):
    env = AscentEnv(shifted=shifted)
    policy = Policy(misgen=misgen)


    Actions, Done, Nactions, Nobs, Obs, Rew = collect_rollouts(env, policy, verbose)
    # Obs = concat_data(Obs, nobs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    critic = MlpModelNoFinalRelu(input_dims=3, hidden_dims=[256, 256, 1])
    current_state_reward = MlpModelNoFinalRelu(input_dims=3, hidden_dims=[256, 256, 1])
    next_reward = MlpModelNoFinalRelu(input_dims=4, hidden_dims=[256, 256, 1])

    critic.to(device)
    next_reward.to(device)
    current_state_reward.to(device)

    optimizer = torch.optim.Adam(list(current_state_reward.parameters()) + list(critic.parameters()), lr=learning_rate)
    nr_optimizer = torch.optim.Adam(next_reward.parameters(), lr=learning_rate)

    obs = torch.FloatTensor(Obs[:-1]).to(device)
    next_obs = torch.FloatTensor(Nobs[:-1]).to(device)
    next_next_obs = torch.FloatTensor(Nobs[1:]).to(device)  # add zero
    acts = torch.FloatTensor(Actions[:-1]).to(device)
    next_actions = torch.FloatTensor(Nactions[:-1]).to(device)
    flt = Done[:-1]
    # flt = np.concatenate((Done[2:],np.array([False])))

    action_idx = [tuple(n for n in range(len(next_actions))), tuple(1 if x == 1 else 0 for x in acts)]
    log_probs = torch.FloatTensor(policy.forward(obs.detach().cpu().numpy(), embed=False)).to(device)
    log_prob_acts = log_probs[action_idx]
    log_prob_acts.requires_grad = False

    log_prob_acts *= inv_temp

    obs_acts = torch.concat((obs, acts.unsqueeze(-1)), -1)

    # torch.concat((obs_acts,next_obs),dim=-1).unique(dim=0)

    losses = []
    # torch.autograd.set_detect_anomaly(True)

    # terminal_obs = torch.zeros(6).to(device)
    # tobs_left = torch.concat([terminal_obs, torch.tensor((-1,)).to(device)], dim=0)
    # tobs_right = torch.concat([terminal_obs, torch.tensor((1,)).to(device)], dim=0)

    # tobs_act = torch.stack([tobs_left, tobs_right], dim=0)

    for epoch in range(epochs):
        rew_hat = current_state_reward(obs)
        val = critic(obs)
        next_val = critic(next_obs)
        # next_val[flt] = 0
        adv_dones = rew_hat[flt] - val[flt] + np.log(2)

        adv = rew_hat[~flt] - val[~flt] + gamma * next_val[~flt]

        loss = torch.nn.MSELoss()(adv.squeeze(), log_prob_acts[~flt])

        # loss2 = torch.nn.MSELoss()(adv_dones.squeeze(), torch.zeros_like(adv_dones.squeeze()))
        loss2 = (adv_dones.squeeze() ** 2).mean()

        coef = flt.mean()

        # terminal_next_rew = next_reward(tobs_act).squeeze()
        #
        # loss3 = torch.nn.MSELoss()(terminal_next_rew, torch.zeros_like(terminal_next_rew))

        total_loss = (1 - coef) * loss + coef * loss2  # + loss3 #no coef for loss3 for now

        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(total_loss.item())
        if epoch % 100 == 0 and verbose:
            print(f"Epoch:{epoch}\tLoss:{total_loss.item():.4f}\t"
                  # f"Next State Reward Loss:{current_state_reward_loss.item():.4f}\t"
                  f"Adv dones:{adv_dones.mean():.4f}\t"
                  f"Adv:{adv.mean():.4f}\t"
                  f"Rew_hat:{rew_hat.mean():.4f}\t"
                  f"Val:{val.mean():.4f}\t"
                  f"Next_val:{next_val.mean():.4f}\t"
                  )

            # nr = next_reward(obs_acts.unique(dim=0)).squeeze()

            no = next_obs.unique(dim=0)
            r = current_state_reward(no).squeeze()

            # next_obs.unique(dim=0)
            values = critic(obs.unique(dim=0)).squeeze()
            print(no)
            print("Rewards", (r - r.min()).detach().cpu().numpy().round(2))
            print("Values", values.detach().cpu().numpy().round(2))
            # for i in range(4):
            #     print(f"\nstate {i}")
            #     print(f"value = {values[i]}")
            #     print(f"r = {r[i]}")
            #     # print(f"r(left) = {nr[2*i]}")
            #     # print(f"r(right) = {nr[2 * i + 1]}")
            #
            #
            #
            # print(f"\nstate {4}")
            # print(f"value = {critic(terminal_obs).squeeze()}")
            # print(f"r = {current_state_reward(terminal_obs).squeeze()}")
            # print(f"rn(left) = {next_reward(tobs_left).squeeze()}")
            # print(f"rn(right) = {next_reward(tobs_left).squeeze()}")

            # print(r)
            # print(f"s_0 l, s_0 r, s_1 l, s_1 r, s_2 l, s_3 r, s_4 l, s_4 r")

            # plt.hist(rew_hat.detach().cpu().numpy())
            # plt.show()

    rew_learned = current_state_reward(next_obs).detach()
    for epoch in range(epochs//5):
        rew_from_last_obs = next_reward(obs_acts)

        current_state_reward_loss = torch.nn.MSELoss()(rew_from_last_obs, rew_learned)
        current_state_reward_loss.backward()
        nr_optimizer.step()
        nr_optimizer.zero_grad()
        if epoch % 100 == 0 and verbose:
            print(f"Epoch:{epoch}\tDistill Loss:{current_state_reward_loss.item():.4f}\t")
            oa = obs_acts.unique(dim=0)
            # print(oa)
            pred_R = next_reward(oa).squeeze().detach().cpu().numpy()
            print((pred_R - pred_R.min()).round(2))
            # plt.scatter(rew_learned.detach().cpu().numpy(), rew_from_last_obs.detach().cpu().numpy())
            # plt.show()
            # print("done")
    prefix = "Mis" if misgen else ""
    prefenv = "Un" if not shifted else ""
    print(f"Env: {prefenv}shifted\tAgent: {prefix}gen")
    print(f"Loss: {total_loss.item():.4f}\tDistill Loss: {current_state_reward_loss.item():.4f}")
    torch_embedder = lambda x: x @ torch.FloatTensor(policy.embedder).to(device)
    next_reward.embedder = torch_embedder
    critic.embedder = torch_embedder
    return next_reward, critic


def collect_rollouts(env, policy, verbose, timesteps = 10000):
    done = True
    Obs = Rew = Done = Nobs = Actions = Nactions = None
    
    for i in range(timesteps):
        if done:
            obs = env.reset()
            if verbose:
                print(env.state, obs)

        action = policy.act(obs)
        nobs, rew, done, info = env.step(action)
        if verbose:
            print(env.state, nobs, rew, done)

        next_action = policy.act(nobs)

        Obs = concat_data(Obs, policy.embed(obs))
        Actions = concat_data(Actions, action)
        Nobs = concat_data(Nobs, policy.embed(nobs))
        Nactions = concat_data(Nactions, next_action)
        Rew = concat_data(Rew, rew)
        Done = concat_data(Done, done)
        obs = nobs
    return Actions, Done, Nactions, Nobs, Obs, Rew

class UniformPolicy():
    def __init__(self):
        pass

    def act(self, obs):
        return np.random.choice([-1, 1], p=[0.5,0.5])

    def embed(self, obs):
        return obs

def evaluate_rew_functions(misgen_fwd_reward, gen_fwd_reward):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = AscentEnv(shifted=True)
    uniform_policy = UniformPolicy()
    Actions, Done, Nactions, Nobs, Obs, Rew = collect_rollouts(env, uniform_policy, verbose=False)

    obs = torch.FloatTensor(Obs).to(device)
    acts = torch.FloatTensor(Actions).to(device)

    misgen_rew = misgen_fwd_reward.embed_and_forward(obs, acts).detach().cpu().numpy().squeeze()
    gen_rew = gen_fwd_reward.embed_and_forward(obs, acts).detach().cpu().numpy().squeeze()

    stacked_rewards = np.stack((Rew, misgen_rew, gen_rew))
    correls = np.corrcoef(stacked_rewards)
    print(f"reward correlations: \n{correls}")

    stacked_info = np.stack((Nobs[:, 2], Rew-Rew.min(), misgen_rew-misgen_rew.min(), gen_rew-gen_rew.min()))
    print(f"Rewards by state: \n{np.unique(stacked_info, axis=1).T.round(2)}")

    return

    true_mc_vals = np.zeros(len(Rew))
    misgen_mc_vals = np.zeros(len(Rew))
    gen_mc_vals = np.zeros(len(Rew))
    for i in range(len(obs))[::-1]:
        if i==len(obs)-1 or Done[i]:
            true_mc_vals[i] = Rew[i]
            misgen_mc_vals[i] = misgen_rew[i]
            gen_mc_vals[i] = gen_rew[i]
        else:
            true_mc_vals[i] = Rew[i] + gamma*true_mc_vals[i+1]
            misgen_mc_vals[i] = misgen_rew[i] + gamma*misgen_mc_vals[i+1]
            gen_mc_vals[i] = gen_rew[i] + gamma*gen_mc_vals[i+1]
        # print(f"i = {i}\trew = {Rew[i]}\t Done = {Done[i]}\t true_mc_val = {true_mc_vals[i]}")

    stacked_mc_vals = np.stack((true_mc_vals, misgen_mc_vals, gen_mc_vals))
    val_correls = np.corrcoef(stacked_mc_vals)
    print(f"mc value correlations: \n{val_correls}")

    stacked_info = np.stack((Nobs[:, 2], true_mc_vals, misgen_mc_vals, gen_mc_vals))
    import pandas as pd
    df = pd.DataFrame(stacked_info.T, columns=["state", "true val", "misgen val", "gen val"])
    grouped_df = df.groupby('state').mean().reset_index()

    grouped_df.corr()

    #conclusion: the mc value correlations look a bit worse than the reward correlations.
    # But maybe that's due to the uniform policy making value a silly metric



    # plt.scatter(x=Rew, y=misgen_rew, alpha=0.5, c='b', label='Misgen')
    # plt.scatter(x=Rew, y=gen_rew, alpha=0.5, c='r', label='Gen')
    # plt.show()

    print("done")





if __name__ == "__main__":
    learned_policy = implicit_policy_learning(verbose=True)

    misgen_fwd_reward, misgen_critic = inverse_reward_shaping(shifted=False, misgen=True)
    gen_fwd_reward, gen_critic = inverse_reward_shaping(shifted=False, misgen=False)

    evaluate_rew_functions(misgen_fwd_reward, gen_fwd_reward)

    print("end")