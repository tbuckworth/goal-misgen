import unittest

import numpy as np
import torch

from common.env.procgen_wrappers import VecRolloutInfoWrapper, EmbedderWrapper, DummyTerminalObsWrapper
from helper_local import create_venv, initialize_policy, get_config
from imitation_rl import get_env_args


class MyTestCase(unittest.TestCase):

    def test_something(self):
        agent_dir = "/home/titus/PycharmProjects/goal-misgen/logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = get_config(agent_dir)
        args = get_env_args(cfg)
        venv = create_venv(args, cfg)
        model, policy = initialize_policy(device, cfg, venv, venv.observation_space.shape)
        model.device = device
        venv = EmbedderWrapper(venv, embedder=model)
        venv = VecRolloutInfoWrapper(venv)
        venv = DummyTerminalObsWrapper(venv)
        obs = venv.reset()
        for i in range(100):
            act = np.array([venv.action_space.sample() for _ in range(len(obs))])
            obs, reward, done, info = venv.step(act)



if __name__ == '__main__':
    unittest.main()
