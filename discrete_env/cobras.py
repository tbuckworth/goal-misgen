import math
from typing import Optional

import numpy as np

from discrete_env.helper_pre_vec import StartSpace, assign_env_vars
from discrete_env.pre_vec_env import PreVecEnv, create_pre_vec
from discrete_env.helper_pre_vec import DictToArgs


class CobraEnv(PreVecEnv):
    def __init__(self, n_envs,
                 n_cobras_start_low=2,
                 n_cobras_start_high=10,
                 max_cobras = 100,
                 max_heads = 100,
                 max_steps=500,
                 seed=0,
                 render_mode: Optional[str] = None,):
        self.n_envs = n_envs
        self.n_cobras_start_low = n_cobras_start_low
        self.n_cobras_start_high = n_cobras_start_high
        n_actions = 2
        # action 0 -> breed
        self.breed = 0
        # action 1 -> decapitate
        self.decap = 1
        self.n_features = 2
        # feature 0 -> n_cobras
        # feature 1 -> n_cobra_heads
        self.state = np.zeros((self.n_envs, self.n_features))
        self.set()
        self.low = np.array([0, 0], dtype=np.float32)
        self.high = np.array([self.max_cobras, self.max_heads], dtype=np.float32)

        self.start_space = StartSpace(
            low=[self.n_cobras_start_low, 0],
            high=[self.n_cobras_start_high, 0],
            np_random=self._np_random)
        if self.sparse_rewards:
            self.reward = np.full(n_envs, -1.0)
            self.info = [{"env_reward": self.reward[i]} for i in range(n_envs)]

        self.customizable_params = [
            "goal_velocity",
            "min_start_position",
            "max_start_position",
            "left_boundary",
            "min_right_boundary",
            "max_right_boundary",
            "min_goal_position",
            "max_goal_position",
            "max_speed",
            "min_goal_position",
            "max_goal_position",
            "force",
            "max_steps",
            "max_gravity",
            "min_gravity",
            "sparse_rewards",
        ]

        super().__init__(n_envs, n_actions, "Cobras", max_steps, seed, render_mode)

    def get_ob_names(self):
        return [
            "Position",
            "Velocity",
            "Gravity",
            "Right Boundary",
            "Goal Position",
        ]

    def transition_model(self, action: np.array):
        n_cobras, n_heads = self.state.T

        breed = np.bitwise_and(self.state > 1, action == self.breed)
        decapitate = np.bitwise_and(self.state > 0, action == self.decap)
        n_cobras[breed] += 1
        n_cobras[decapitate] -= 1
        n_heads[decapitate] += 1

        self.terminated = np.bitwise_or(n_cobras>self.max_cobras,n_heads>self.max_heads)

        self.state = np.vstack((n_cobras, n_heads)).T

        self.reward = n_heads
        self.info = [{"env_reward": self.reward[i]} for i in range(self.num_envs)]

    def get_action_lookup(self):
        return {
            0: 'breed',
            1: 'decapitate',
        }

    # def set(self):
    #     self.state[...,0] = self.n_cobras_start
    #     self.state[...,1] = 0
    #
    # def step(self, action):
    #     assert len(action) == self.n_envs, f"action must be a vector of length n_envs, got {action.shape}"
    #     self.state[np.bitwise_and(self.state>1, action==self.breed), self.cobr_count] += 1
    #     self.state[np.bitwise_and(self.state>0, action==self.decap), self.head_count] += 1
    #
    #     obs = self.state.copy()
    #
    #     rew = self.rew_func(obs)
    #     done = self.terminate()
    #
    #     return obs, rew, done, info
