import unittest

import numpy as np

from common.env.env_constructor import get_env_constructor
from discrete_env.acrobot_pre_vec import AcrobotVecEnv, create_acrobot
from discrete_env.cartpole_pre_vec import CartPoleVecEnv
from discrete_env.mountain_car_pre_vec import MountainCarVecEnv, create_mountain_car
from discrete_env.helper_pre_vec import DictToArgs
from train import create_stacked_env


class BaseDiscreteEnvTest(unittest.TestCase):
    def setUp(cls) -> None:
        cls.env_cons = get_env_constructor(cls.env_name)
        cls.env = cls.env_cons(DictToArgs({"render": "human", "seed": 0}), {"n_envs": 24, "max_steps": 30, "drop_same": True},
                               False)
        cls.env.reset(seed=np.random.randint(0, 1000))

    def run_step(self):
        for i in range(1000):
            actions = [self.env.action_space.sample() for _ in range(self.env.num_envs)]
            # actions = [0 for _ in range(self.env.n_envs)]
            obs, rew, done, info = self.env.step(actions)
            if done[0]:
                print(i)


class TestMountainCar(BaseDiscreteEnvTest):

    def setUp(cls) -> None:
        cls.env_name = "mountain_car"
        super(TestMountainCar, cls).setUp()

    def test_step(self):
        self.run_step()


class TestAcrobot(BaseDiscreteEnvTest):
    def setUp(cls) -> None:
        cls.env_name = "acrobot"
        super(TestAcrobot, cls).setUp()

    def test_step(self):
        self.run_step()


class TestCartpole(BaseDiscreteEnvTest):
    def setUp(cls) -> None:
        cls.env_name = "cartpole"
        super(TestCartpole, cls).setUp()

    def test_step(self):
        self.run_step()


class TestCartpoleSwing(BaseDiscreteEnvTest):
    def setUp(cls) -> None:
        cls.env_name = "cartpole_swing"
        super(TestCartpoleSwing, cls).setUp()

    def test_step(self):
        self.run_step()

class TestCobras(BaseDiscreteEnvTest):
    def setUp(cls) -> None:
        cls.env_name = "cobras"
        super(TestCobras, cls).setUp()

    def test_step(self):
        self.run_step()

class TestStackedCartpole(BaseDiscreteEnvTest):

    def setUp(cls) -> None:
        cls.env = create_stacked_env(
            DictToArgs({"render": "human", "seed": 0, "env_name": "cartpole"}),
            {"max_steps": 30, "drop_same": True},
            pct_ood=0.5
        )
        cls.env.reset(seed=np.random.randint(0, 1000))

    def test_step(self):
        self.run_step()


if __name__ == '__main__':
    unittest.main()
