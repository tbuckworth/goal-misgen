import unittest

import numpy as np


class AscenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from common.ascent_env import AscentEnv
        cls.env = AscentEnv(num_envs=5, shifted=False)

    def test_something(self):
        self.env.state = np.arange(5)
        obs, rew, done, _ = self.env.observe()
        print(obs)

        print(obs)

    def test_step(self):
        self.env.state = np.arange(5)
        obs, rew, done, _ = self.env.observe()
        actions = np.array([1, 1, -1, -1, 1])
        obs, rew, done, _ = self.env.step(actions)
        assert np.all(self.env.state == np.array([0, 2, 3, 2, 3])), "something went wrong!"
        print("done")


if __name__ == '__main__':
    unittest.main()
