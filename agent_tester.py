import unittest

import torch

from common.logger import Logger
from common.model import RewValModel, NextRewModel
from common.storage import LirlStorage
from helper_local import get_hyperparameters, create_venv, initialize_policy, listdir, DictToArgs
from train import initialize_agent, initialize_storage, create_logdir


class PPO_LirlTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls, get_latest_model=None):
        num_checkpoints = 1
        seed = 42
        env_name = "ascent"
        exp_name = "ascent"
        model_file = ""
        param_name = "ascent-mlp"
        hidden_dims = [64, 64]
        args = DictToArgs({"env_name":env_name})

        hyperparameters = get_hyperparameters(param_name)
        n_steps = hyperparameters["n_steps"]
        n_envs = hyperparameters["n_envs"]
        algo = hyperparameters["algo"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = create_venv(args, hyperparameters)
        env_valid = create_venv(args, hyperparameters, is_valid=True)

        observation_space = env.observation_space
        observation_shape = observation_space.shape
        action_size = env.action_space.n

        model, policy = initialize_policy(device, hyperparameters, env, observation_shape)

        rew_val_model = RewValModel(model.output_dim, hidden_dims, device)
        next_rew_model = NextRewModel(model.output_dim + action_size, hidden_dims, action_size, device)

        storage, storage_valid, storage_trusted = initialize_storage(device, model, n_envs, n_steps, observation_shape, algo)

        ppo_lirl_params = dict(
            num_rew_updates=10,
            rew_val_model=rew_val_model,
            next_rew_model=next_rew_model,
            inv_temp_rew_model=1.,
            next_rew_loss_coef=1.,
            storage_trusted=storage_trusted,
        )
        hyperparameters.update(ppo_lirl_params)

        logdir = create_logdir(model_file, env_name, exp_name, get_latest_model, listdir, seed)
        logger = Logger(n_envs, logdir, use_wandb=False)

        cls.agent = initialize_agent(device, env, env_valid, hyperparameters, logger, num_checkpoints, policy, storage,
                                     storage_valid)

    def test_agent(self):
        self.agent.train(int(1e5))


if __name__ == '__main__':
    unittest.main()
