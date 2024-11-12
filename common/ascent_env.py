import numpy as np
import gym
from gym import spaces


class AscentEnv():
    def __init__(self, num_envs=2, shifted=False, n_states=5):
        self.observation_space = spaces.Box(-1, 4, (6,))
        self.action_space = spaces.Discrete(2)
        self.num_envs = num_envs
        self.shifted = shifted
        self.n_states = n_states
        self.states = np.arange(n_states)
        self.state = np.zeros(num_envs)
        self.mirror = np.arange(self.num_envs) % 2 == 0
        self.infos = [{} for _ in range(self.num_envs)]

    def close(self):
        pass

    def obs(self, state):
        obs = np.zeros((self.num_envs, 6))
        obs[..., 2] = state
        obs[..., 3] = state
        obs[..., 4:6] = obs[..., 2:4] + 1
        obs[..., 0:2] = obs[..., 2:4] - 1
        if self.shifted:
            obs[..., 1] = self.n_states - state - 1
            obs[..., 3] = self.n_states - state - 2
            obs[..., 5] = self.n_states - state - 3

        obs[state == self.n_states - 1] = 0

        obs[self.mirror] = obs[self.mirror][..., ::-1]

        return obs

    def reward(self, state):
        rews = np.zeros(self.num_envs)
        rews[state == self.n_states - 1] = 10.
        return rews

    def done(self, state):
        dones = np.zeros(self.num_envs)
        dones[state == self.n_states - 1] = 1
        return dones.astype(bool)

    def step(self, act):
        action = act.copy()
        action[action == 0] = -1
        assert len(action) == self.num_envs, "Action batch dimension should match num envs"
        assert np.all(np.bitwise_or(action == 1, action == -1)), "All actions must be -1 or 1"
        action[self.mirror] *= -1

        self.state = self.state + action
        self.state[self.state < 0] = 0
        self.state[self.state >= self.num_envs] = self.n_states - 1

        rew = self.reward(self.state)
        done = self.done(self.state)
        self.state[done] = 0
        obs = self.obs(self.state)

        return obs, rew, done, self.infos

    def reset(self):
        self.state = np.zeros(self.num_envs)
        return self.obs(self.state)

    def observe(self):
        return self.obs(self.state), self.reward(self.state), self.done(self.state), self.infos
