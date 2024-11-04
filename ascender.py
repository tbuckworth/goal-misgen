

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import softmax, log_softmax
import torch

from common.model import MlpModel

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

def main(verbose=False, gamma=0.99, epochs=10000, learning_rate=1e-5):
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

    next_reward = MlpModel(input_dims=7, hidden_dims=[64,64,1])
    critic = MlpModel(input_dims=6, hidden_dims=[64, 64, 1])

    next_reward.to(device)
    critic.to(device)

    optimizer = torch.optim.Adam(list(next_reward.parameters())+ list(critic.parameters()), lr = learning_rate)

    obs = torch.FloatTensor(Obs[:-1]).to(device)
    next_obs = torch.FloatTensor(Nobs[:-1]).to(device)
    next_next_obs = torch.FloatTensor(Nobs[1:]).to(device) # add zero
    acts = torch.FloatTensor(Actions[:-1]).to(device)
    next_actions = torch.FloatTensor(Nactions[:-1]).to(device)

    next_action_idx = [tuple(n for n in range(len(next_actions))),tuple(1 if x == 1 else 0 for x in next_actions)]
    log_probs = torch.FloatTensor(policy.forward(next_obs.detach().cpu().numpy())).to(device)
    log_prob_acts = log_probs[next_action_idx]
    log_prob_acts.requires_grad = False

    obs_acts = torch.concat((obs,acts.unsqueeze(-1)),-1)
    losses = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):    
        rew_hat = next_reward(obs_acts)
        next_val = critic(next_obs)
        next_next_val = critic(next_next_obs)
        flt = Done[:-1]
        # next_next_val[flt] = 0
        adv_dones = rew_hat[flt] - next_val[flt]

        adv = rew_hat[~flt] - next_val[~flt] + gamma * next_next_val[~flt]

        loss = torch.nn.MSELoss()(adv.squeeze(), log_prob_acts[~flt])

        loss2 = (adv_dones.squeeze()**2).mean()

        coef = flt.mean()
        coef = flt.mean()

        total_loss = loss + coef * loss2

        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(total_loss.item())
        if epoch % 100 == 0:
            print(f"Epoch:{epoch}\tLoss{total_loss.item():.4f}\tAdv dones:{adv_dones.mean():.4f}\tAdv:{adv.mean():.4f}\tRew_hat:{rew_hat[flt].mean():.4f}")
            
            # r = next_reward(obs_acts.unique(dim=0))
            # print(r)
            # print(f"s_0 l, s_0 r, s_1 l, s_1 r, s_2 l, s_3 r, s_4 l, s_4 r")

            # plt.hist(rew_hat.detach().cpu().numpy())
            # plt.show()

if __name__ == "__main__":
    main()
