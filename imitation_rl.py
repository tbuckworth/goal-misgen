import os
import warnings

import torch

from common.env.procgen_wrappers import VecRolloutInfoWrapper, EmbedderWrapper, DummyTerminalObsWrapper
from common.model import IdentityModel
from common.policy import PolicyWrapperIRL
from helper_local import create_venv, DictToArgs, initialize_policy, latest_model_path, get_hyperparameters
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass

unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

import numpy as np
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data import rollout
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor


def get_config(logdir):
    return np.load(os.path.join(logdir, "config.npy"), allow_pickle='TRUE').item()


def get_env_args(logdir):
    # manual implementation for now
    warnings.warn("manually selecting env_args for now")
    env_args = {
        "val_env_name": "coinrun",
        "env_name": "coinrun",
        "num_levels": 10000,
        "start_level": 0,
        "distribution_mode": "hard",
        "num_threads": 8,
        "random_percent": 0,
        "step_penalty": 0,
        "key_penalty": 0,
        "rand_region": 0,
        "param_name": "hard-500",
    }
    return DictToArgs(env_args)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

    agent_dir = unshifted_agent_dir

    cfg = get_config(agent_dir)
    # cfg = {}
    args = get_env_args(agent_dir)
    hyperparameters = cfg#get_hyperparameters(args.param_name)

    use_wandb = True

    SEED = 42

    FAST = True
    SUPERFAST = False
    n_steps = 2048
    demo_batch_size = 2048

    if FAST:
        N_RL_TRAIN_STEPS = int(2e19 + 1)#100_000
    else:
        N_RL_TRAIN_STEPS = 2_000_000
    if SUPERFAST:
        N_RL_TRAIN_STEPS = 100
        hyperparameters["n_envs"] = 4
        n_steps = 4

    # make_vec_env(
    #     "seals:seals/CartPole-v0",
    #     rng=np.random.default_rng(SEED),
    #     n_envs=8,
    #     post_wrappers=[
    #         lambda env, _: RolloutInfoWrapper(env)
    #     ],  # needed for computing rollouts later
    # )

    # Wrap with a VecMonitor to collect stats and avoid errors
    venv = create_venv(args, cfg)

    model, policy = initialize_policy(device, hyperparameters, venv, venv.observation_space.shape)
    model.device = device
    # load policy
    last_model = latest_model_path(agent_dir)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])

    policy.embedder = IdentityModel()

    expert = PolicyWrapperIRL(policy, device)

    venv = EmbedderWrapper(venv, embedder=model)
    venv = VecRolloutInfoWrapper(venv)
    venv = DummyTerminalObsWrapper(venv)
    venv = VecMonitor(venv=venv)

    # # VecMonitor
    # TODO: make this systematic in a wrapper:
    import gymnasium

    venv.action_space = gymnasium.spaces.discrete.Discrete(15)
    # venv.observation_space = gymnasium.spaces.box.Box(0.0, 1.0, (3, 64, 64))

    # expert = load_policy(
    #     "ppo-huggingface",
    #     organization="HumanCompatibleAI",
    #     env_name="seals/CartPole-v0",
    #     venv=venv,
    # )



    # We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.

    rollouts = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=2048),
        rng=np.random.default_rng(SEED),
    )
    # Now we are ready to set up our AIRL trainer. Note, that the reward_net is actually the network of the discriminator. We evaluate the learner before and after training so we can see if it made any progress.
    # sb3_policy = ActorCriticPolicy(net_arch=dict(pi=[],vf=[]))
    sb3_policy = lambda *args, **kwargs: ActorCriticPolicy(*args, net_arch=dict(pi=[],vf=[]), **kwargs)

    learner = PPO(
        env=venv,
        policy=sb3_policy,#MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=SEED,
        n_steps=n_steps,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        log_dir = "results/"
    )

    venv.seed(SEED)

    if use_wandb:
        wandb_login()
        # cfg = vars(args)
        # cfg.update(hyperparameters)
        wandb.init(project="LIRL", config={}, tags=[], sync_tensorboard=True)


    airl_trainer.train(N_RL_TRAIN_STEPS)

    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )

    venv.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, n_eval_episodes=100, return_episode_rewards=True
    )

    print(
        "Rewards before training:",
        np.mean(learner_rewards_before_training),
        "+/-",
        np.std(learner_rewards_before_training),
    )

    print(
        f"Rewards after training",
        np.mean(learner_rewards_after_training),
        "+/-",
        np.std(learner_rewards_after_training),
    )
    wandb.finish()




if __name__ == "__main__":
    main()
