"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import inspect
import math
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces, Env
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

from discrete_env.helper_pre_vec import assign_env_vars, DictToArgs

class PreVecEnv(Env):
    """
    ### Description
    Absract base class for pre-vectorized discrete environments.
    Pre-vectorized means applying vectorization at the environment level as opposed to stacking concurrent environments.
    """
    screen_width = None
    screen_height = None
    input_adjust = 0
    drop_same = False

    def __init__(self, n_envs, n_actions,
                 env_name,
                 max_steps=500,
                 seed=0,
                 render_mode: Optional[str] = None):
        self.non_drop_index = self.high != self.low
        if not self.drop_same:
            self.non_drop_index = self.high == self.high
        self.env_name = env_name
        if n_envs < 2:
            raise Exception("n_envs must be greater than or equal to 2")
        self.num_envs = n_envs
        self.max_steps = max_steps
        self.n_actions = n_actions
        self.np_random_seed = None
        self._np_random = None
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(self.low[self.non_drop_index], self.high[self.non_drop_index],
                                            dtype=np.float32)
        self.render_mode = render_mode
        self.screen_width = 600 if self.screen_width is None else self.screen_width
        self.screen_height = 400 if self.screen_height is None else self.screen_width
        self.screen = None
        self.clock = None
        self.isopen = True
        self.n_inputs = len(self.high) + self.input_adjust
        self.terminated = np.full(self.num_envs, True)
        self.state = np.zeros((self.num_envs, self.n_inputs))
        self.n_steps = np.zeros((self.num_envs))
        self.reset(seed=seed)

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def check_actions(self, action):
        if isinstance(self.action_space, spaces.Box):
            h = self.action_space.high.repeat(self.num_envs)
            l = self.action_space.low.repeat(self.num_envs)
            action[action < l] = l[action < l]
            action[action > h] = h[action > h]
            return action
        assert np.all(action < self.n_actions), f"action must be less than n_actions ({self.n_actions})"
        return action

    def step(self, action):
        action = np.array(action)
        assert action.size == self.num_envs, f"number of actions ({action.size}) must match n_envs ({self.num_envs})"
        action = self.check_actions(action)
        self.transition_model(action)

        self.n_steps += 1
        truncated = self.n_steps >= self.max_steps
        self.terminated[truncated] = True

        if np.any(self.terminated):
            self.set()

        if self.render_mode == "human":
            self.render()
        return self._get_ob(), self.reward, self.terminated, self.info

    def get_params(self, suffix=""):
        return {f"{name}{suffix}": self.__getattribute__(name) for name in self.customizable_params}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.terminated = np.full(self.num_envs, True)
        self.n_steps = np.zeros((self.num_envs))
        return self.set(seed=seed)

    def set(self,
            *,
            seed: Optional[int] = None,
            ):
        self.seed(seed)
        state = self.start_space.sample(self.num_envs)
        self.state[self.terminated] = state[self.terminated]
        self.n_steps[self.terminated] = 0

        if self.render_mode == "human":
            self.render()
        return self._get_ob()

    def seed(self, seed=None):
        self.np_random_seed = seed
        if self.np_random_seed is not None:
            self._np_random, self.np_random_seed = seeding.np_random(seed)
            self.start_space.set_np_random(self._np_random)
        return [seed]

    def save(self):
        np.save(f'{self.env_name}.npy', self.state)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            self.pygame = pygame
            from pygame import gfxdraw
            self.gfxdraw = gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if not self.render_unique():
            return None

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _get_ob(self):
        assert self.state is not None, "Call reset before using PreVecEnv object."
        if self.drop_same:
            return self.state[..., self.non_drop_index]
        return self.state

    def get_info(self):
        return self.info

    def get_action_lookup(self):
        raise NotImplementedError

    def transition_model(self, action):
        raise NotImplementedError

    def render_unique(self):
        raise NotImplementedError


def create_pre_vec(args, hyperparameters, param_range, env_cons, is_valid):
    if args is None:
        args = DictToArgs({"render": False, "seed": 0})
    n_envs = hyperparameters.get('n_envs', 32)

    env_args = assign_env_vars(hyperparameters, is_valid, param_range)
    env_args = filter_out_non_relevant_params(env_args, env_cons)
    env_args["n_envs"] = n_envs
    env_args["render_mode"] = "human" if args.render else None
    env_args["seed"] = args.seed
    return env_cons(**env_args)


def filter_out_non_relevant_params(env_args, env_cons):
    params = inspect.signature(env_cons.__init__).parameters.keys()
    return {k: v for k, v in env_args.items() if k in params}
