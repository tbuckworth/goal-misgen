import numpy as np

class AscentEnv():
    def __init__(self, shifted=False, n_states=5):
        self.shifted = shifted
        self.n_states = n_states
        self.states = np.arange(n_states)
        self.state = 0
        self.mirror = False

    def close():
        pass

    def obs(self, state):
        if state == self.n_states - 1:
            return np.zeros((6,))
        obs = np.concatenate([self.features(state - 1), self.features(state), self.features(state + 1)])
        if self.mirror:
            return obs[::-1].copy()
        return obs

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
        if self.mirror:
            action *= -1
        self.state = min(max(self.state + action, 0), self.n_states - 1)

        obs, rew, done, info = self.observe()
        if done:
            self.reset()
        return obs, rew, done, info

    def reset(self):
        self.state = 0
        self.mirror = not self.mirror
        return self.obs(self.state)

    def observe(self):
        return self.obs(self.state), self.reward(self.state), self.done(self.state), {}
