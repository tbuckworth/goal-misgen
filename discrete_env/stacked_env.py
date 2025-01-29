
from typing import Optional

import numpy as np

from gymnasium import Env

class StackedEnv(Env):
    """
    ### Description
    """
    def __init__(self, envs):
        self.envs = envs
        self.e_num_envs = [e.num_envs for e in self.envs]
        self.num_envs = sum(self.e_num_envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, action):
        action = np.array(action)
        assert action.size == self.num_envs, f"number of actions ({action.size}) must match n_envs ({self.num_envs})"
        start = 0
        for env in self.envs:
            n = env.num_envs
            env.step(action[start:start+n])
            start += n


        if self.render_mode == "human":
            self.render()
        return self._get_ob(), self.reward(), self.terminated(), self.info()

    def reward(self):
        return np.concatenate([e.reward for e in self.envs], axis=0)

    def terminated(self):
        return np.concatenate([e.terminated for e in self.envs], axis=0)

    def info(self):
        infos = []
        for e in self.envs:
            infos += e.info
        return infos


    def get_params(self, suffix=""):
        return self.envs[0].get_params(suffix)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        return np.concatenate([e.reset(seed=seed, options=options) for e in self.envs], axis=0)

    def set(self,
            *,
            seed: Optional[int] = None,
            ):
        return np.concatenate([e.set(seed=seed) for e in self.envs], axis=0)

    def seed(self, seed=None):
        return [[e.seed(seed) for e in self.envs][0]]

    def save(self):
        pass

    def render(self):
        self.envs[0].render()

    def close(self):
        [e.close() for e in self.envs]

    def _get_ob(self):
        return np.concatenate([e._get_ob() for e in self.envs], axis=0)

    def get_info(self):
        return self.info()

    def get_action_lookup(self):
        raise NotImplementedError

    def transition_model(self, action):
        raise NotImplementedError

    def render_unique(self):
        raise NotImplementedError


