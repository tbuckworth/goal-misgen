import unittest

import numpy as np

from common.env.procgen_wrappers import VecRolloutInfoWrapper
from helper_local import create_venv
from imitation_rl import get_config, get_env_args


class MyTestCase(unittest.TestCase):

    def test_something(self):
        agent_dir = "/home/titus/PycharmProjects/goal-misgen/logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
        cfg = get_config(agent_dir)
        args = get_env_args(agent_dir)
        venv = create_venv(args, cfg)
        venv = VecRolloutInfoWrapper(venv)
        obs = venv.reset()
        for i in range(100):
            act = np.array([venv.action_space.sample() for _ in range(len(obs))])
            obs, reward, done, info = venv.step(act)



if __name__ == '__main__':
    unittest.main()
