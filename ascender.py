

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import softmax, log_softmax
import torch

from common.model import MlpModel, MlpModelNoFinalRelu


class AscentEnv():
    def __init__(self, shifted=False, n_states = 5):
        self.shifted = shifted
        self.n_states = n_states
        self.states = np.arange(n_states)
        self.state = 0
        
    def obs(self, state):
        if state == self.n_states - 1:
            return np.zeros((6,))
        return np.concatenate([self.features(state-1),self.features(state),self.features(state+1)]) 
    
    def features(self, state):
        if state<0 or state >= self.n_states:
            return -np.ones((2,))
        if not self.shifted:
            return np.full((2,), state + 1 )
        return  np.array((state + 1 , self.n_states - state))

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
        self.embedder = np.zeros((6,3))
        self.actor = np.zeros((3,2))
        self.actor[0,0] = 1.
        self.actor[2,1] = 1.
        if not misgen:
            self.embedder[0,0] = 1.
            self.embedder[2,1] = 1.
            self.embedder[4,2] = 1.
        else:
            self.embedder[1,0] = 1.
            self.embedder[3,1] = 1.
            self.embedder[5,2] = 1.

    def forward(self, obs):
        logits = obs @ self.embedder @ self.actor
        return log_softmax(logits, axis=-1)
    
    def act(self, obs):
        logits = self.forward(obs)
        p = np.exp(logits)
        return np.random.choice([-1,1], p=p)
        r = np.random.random(1)
        if p[0] < r:
            return -1
        return 1

def concat_data(Obs, obs):
    obs = np.array(obs)
    if Obs is None:
        return np.expand_dims(obs,axis=0)
    return np.concatenate((Obs, np.expand_dims(obs, axis=0)),axis=0)

def main(verbose=False, gamma=0.99, epochs=10000, learning_rate=1e-3, inv_temp=1):
    env = AscentEnv(shifted=False)
    policy = Policy(misgen=False)
    done = True

    Obs = Rew = Done = Nobs = Actions = Nactions = None
    for i in range(10000):
        if done:
            obs = env.reset()
            if verbose:
                print(env.state, obs)

        action = policy.act(obs)
        nobs, rew, done, info = env.step(action)
        if verbose:
            print(env.state, nobs, rew, done)

        next_action = policy.act(nobs)

        Obs = concat_data(Obs, obs)
        Actions = concat_data(Actions, action)
        Nobs = concat_data(Nobs, nobs)
        Nactions = concat_data(Nactions, next_action)
        Rew = concat_data(Rew, rew)
        Done = concat_data(Done, done)
        obs = nobs
    # Obs = concat_data(Obs, nobs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    critic = MlpModelNoFinalRelu(input_dims=6, hidden_dims=[256, 256, 1])
    current_state_reward = MlpModelNoFinalRelu(input_dims=6, hidden_dims=[256, 256, 1])
    next_reward = MlpModelNoFinalRelu(input_dims=7, hidden_dims=[256, 256, 1])

    critic.to(device)
    next_reward.to(device)
    current_state_reward.to(device)

    optimizer = torch.optim.Adam(list(current_state_reward.parameters())+ list(critic.parameters()), lr = learning_rate)
    nr_optimizer = torch.optim.Adam(next_reward.parameters(), lr=learning_rate)

    obs = torch.FloatTensor(Obs[:-1]).to(device)
    next_obs = torch.FloatTensor(Nobs[:-1]).to(device)
    next_next_obs = torch.FloatTensor(Nobs[1:]).to(device) # add zero
    acts = torch.FloatTensor(Actions[:-1]).to(device)
    next_actions = torch.FloatTensor(Nactions[:-1]).to(device)
    flt = Done[:-1]
    # flt = np.concatenate((Done[2:],np.array([False])))

    action_idx = [tuple(n for n in range(len(next_actions))),tuple(1 if x == 1 else 0 for x in acts)]
    log_probs = torch.FloatTensor(policy.forward(obs.detach().cpu().numpy())).to(device)
    log_prob_acts = log_probs[action_idx]
    log_prob_acts.requires_grad = False

    log_prob_acts *= inv_temp

    obs_acts = torch.concat((obs,acts.unsqueeze(-1)),-1)
    losses = []
    # torch.autograd.set_detect_anomaly(True)

    terminal_obs = torch.zeros(6).to(device)
    tobs_left = torch.concat([terminal_obs, torch.tensor((-1,)).to(device)], dim=0)
    tobs_right = torch.concat([terminal_obs, torch.tensor((1,)).to(device)], dim=0)

    tobs_act = torch.stack([tobs_left, tobs_right], dim=0)


    for epoch in range(epochs):
        rew_hat = current_state_reward(obs)
        val = critic(obs)
        next_val = critic(next_obs)
        # next_val[flt] = 0
        adv_dones =  rew_hat[flt] - val[flt] + np.log(2)

        adv = rew_hat[~flt] - val[~flt] + gamma * next_val[~flt]

        loss = torch.nn.MSELoss()(adv.squeeze(), log_prob_acts[~flt])

        # loss2 = torch.nn.MSELoss()(adv_dones.squeeze(), torch.zeros_like(adv_dones.squeeze()))
        loss2 = (adv_dones.squeeze()**2).mean()

        coef = flt.mean()

        # terminal_next_rew = next_reward(tobs_act).squeeze()
        #
        # loss3 = torch.nn.MSELoss()(terminal_next_rew, torch.zeros_like(terminal_next_rew))

        total_loss = (1-coef) * loss + coef * loss2 # + loss3 #no coef for loss3 for now

        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()



        losses.append(total_loss.item())
        if epoch % 100 == 0:
            print(f"Epoch:{epoch}\tLoss:{total_loss.item():.4f}\t"
                  # f"Next State Reward Loss:{current_state_reward_loss.item():.4f}\t"
                  f"Adv dones:{adv_dones.mean():.4f}\t"
                  f"Adv:{adv.mean():.4f}\t"
                  f"Rew_hat:{rew_hat.mean():.4f}\t"
                  f"Val:{val.mean():.4f}\t"
                  f"Next_val:{next_val.mean():.4f}\t"
                  )

            # nr = next_reward(obs_acts.unique(dim=0)).squeeze()

            r = current_state_reward(next_obs.unique(dim=0)).squeeze()

            # next_obs.unique(dim=0)
            values = critic(obs.unique(dim=0)).squeeze()

            print("Rewards", (r-r.min()).detach().cpu().numpy().round(2))
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
    for epoch in range(epochs):
        rew_from_last_obs = next_reward(obs_acts)

        current_state_reward_loss = torch.nn.MSELoss()(rew_from_last_obs, rew_learned)
        current_state_reward_loss.backward()
        nr_optimizer.step()
        nr_optimizer.zero_grad()
        if epoch % 100 == 0:
            print(f"Distill Loss:{current_state_reward_loss.item():.4f}\t")
            pred_R = next_reward(obs_acts.unique(dim=0)).squeeze().detach().cpu().numpy().round(2)
            print(pred_R-pred_R.min())

if __name__ == "__main__":
    main()
