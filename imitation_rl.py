import copy
import os
import warnings

import imitation.util.logger
import torch
from torch import nn

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

# unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
# shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

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

def predict(policy, obs):
    with torch.no_grad():
        obs = torch.FloatTensor(obs).to(device=policy.device)
        p, v, _ = policy(obs, None, None)
        log_probs = p.logits
        action = p.sample().detach().cpu().numpy()
    return log_probs, action


def collect_data(policy, venv, n):
    obs = venv.reset()
    states = obs
    start = True
    while(len(states) < n):
        log_probs, action = predict(policy, obs)
        next_obs, rew, done, info = venv.step(action)
        if start:
            actions = action
            next_states = next_obs
            dones = done
            lp = log_probs
            start = False
        else:
            actions = np.append(actions, action, 0)
            next_states = np.append(next_states, next_obs, 0)
            dones = np.append(dones, done, 0)
            # lp = np.append(lp, log_probs, 0)
            lp = torch.cat((lp, log_probs), 0)
        states = np.append(states, next_obs, 0)
    return states[:-len(next_obs)], actions, next_states, dones, lp

def generate_data(agent, env, n):
    X, _ = env.reset()
    X = np.expand_dims(X,0)
    agent.reset()
    act = env.action_space.sample()
    A = np.expand_dims(act.copy(), 0)
    ep_count = 0
    while ep_count < n:
        x, rew, done, trunc, info = env.step(act)
        act = env.action_space.sample()
        if ep_count == 0 and not done:
            act = agent.forward(x)
        if done:
            ep_count += 1
            m_in, m_out, u_in, u_out, loss = agent.sample_pre_act(X, A)
            X = np.expand_dims(x, 0)
            A = np.expand_dims(act, 0)
            if ep_count == 1:
                M_in, M_out, U_in, U_out, Loss = m_in, m_out, u_in, u_out, loss
            else:
                M_in = np.append(M_in, m_in, axis=1)
                M_out = np.append(M_out, m_out, axis=1)
                U_in = np.append(U_in, u_in, axis=1)
                U_out = np.append(U_out, u_out, axis=1)
                Loss = np.append(Loss, loss, axis=1)

        X = np.append(X, np.expand_dims(x, 0), axis=0)
        A = np.append(A, np.expand_dims(act, 0), axis=0)

    return M_in, M_out, U_in, U_out, Loss

def reward_forward(reward_net, states, actions, next_states, dones, device, idx):
    s = torch.FloatTensor(states).to(device=device)[idx]
    # make one-hot:
    act = torch.FloatTensor(actions).to(device=device).unsqueeze(dim=-1)
    a = nn.functional.one_hot(act)[idx]
    n = torch.FloatTensor(next_states).to(device=device)[idx]
    d = torch.FloatTensor(dones).to(device=device)[idx]
    return reward_net(s, a, n, d)

def train_reward_net(reward_net, venv, policy, rollouts, args_dict):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=args_dict["reward_shaping_lr"])

    reward_net.train()
    start_weights = [copy.deepcopy(x.detach()) for x in policy.parameters()]
    for epoch in range(args_dict.get("n_reward_net_epochs")):
        # collect data
        states, actions, next_states, dones, log_probs = collect_data(policy, venv, n=10000)
        # shuffle
        idx = torch.randperm(len(states))
        rewards = reward_forward(reward_net, states, actions, next_states, dones, policy.device, idx)
        # TODO: could actually use rewards for all actions from this state
        loss = loss_function(rewards, log_probs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_net.parameters(), 40)
        optimizer.step()
        optimizer.zero_grad()
        reward_net.optimizer.step()
    end_weights = [copy.deepcopy(x.detach()) for x in policy.parameters()]

    for s,e in zip(start_weights, end_weights):
        assert (s == e).all(), "policy has changed!"




def lirl(args_dict):
    agent_dir = args_dict.get("agent_dir")
    cfg = get_config(agent_dir)
    #TODO: this is manually overriden:
    args = get_env_args(agent_dir)
    hyperparameters = cfg
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # high level args
    use_wandb = args_dict.get("use_wandb")
    seed = args_dict.get("seed")
    fast = args_dict.get("fast")


    log_path = "results/"

    # overrides:
    # cfg["learning_rate"] = 0.00005

    # learner arguments
    ppo_batch_size = cfg.get("mini_batch_size")
    ppo_ent_coef = cfg.get("entropy_coef")
    ppo_learning_rate = cfg.get("learning_rate")
    ppo_gamma = cfg.get("gamma")
    ppo_clip_range = cfg.get("eps_clip")
    ppo_vf_coef = cfg.get("value_coef")
    ppo_n_epochs = cfg.get("epoch")
    ppo_n_steps = cfg.get("n_steps")

    # reward_net arguments
    reward_hid_sizes = args_dict.get("reward_hid_sizes")
    potential_hid_sizes = args_dict.get("potential_hid_sizes")

    # AIRL arguments:
    demo_batch_size = args_dict.get("demo_batch_size")
    gen_replay_buffer_capacity = args_dict.get("gen_replay_buffer_capacity")
    n_disc_updates_per_round = args_dict.get("n_disc_updates_per_round")
    allow_variable_horizon = args_dict.get("allow_variable_horizon")
    irl_steps_low = args_dict.get("irl_steps_low")
    irl_steps_high = args_dict.get("irl_steps_high")
    if fast:
        n_rl_train_steps = irl_steps_low  # 100_000
    else:
        n_rl_train_steps = irl_steps_high

    # evals
    n_eval_episodes = args_dict.get("n_eval_episodes")

    venv = create_venv(args, cfg)

    model, policy = initialize_policy(device, hyperparameters, venv, venv.observation_space.shape)
    model.device = device
    policy.device = device
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

    # We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.

    rollouts = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=2048),
        rng=np.random.default_rng(seed),
    )
    # Now we are ready to set up our AIRL trainer. Note, that the reward_net is actually the network of the discriminator.
    # We evaluate the learner before and after training so we can see if it made any progress.

    # TODO: make this systematic:
    sb3_policy = lambda *args, **kwargs: ActorCriticPolicy(*args, net_arch=dict(pi=[], vf=[]), **kwargs)

    learner = PPO(
        env=venv,
        policy=sb3_policy,  # MlpPolicy,
        batch_size=ppo_batch_size,
        ent_coef=ppo_ent_coef,
        learning_rate=ppo_learning_rate,
        gamma=ppo_gamma,
        clip_range=ppo_clip_range,
        vf_coef=ppo_vf_coef,
        n_epochs=ppo_n_epochs,
        seed=seed,
        n_steps=ppo_n_steps,
    )

    if args_dict.get("copy_weights"):
        # copy weights into learner:
        learner.policy.action_net.load_state_dict(policy.fc_policy.state_dict())

    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        reward_hid_sizes=reward_hid_sizes,  # (32,),
        potential_hid_sizes=potential_hid_sizes,  # (32, 32),
    )
    logger = imitation.util.logger.configure(log_path, ["stdout", "csv", "tensorboard"])


    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=allow_variable_horizon,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        log_dir=log_path,
        custom_logger=logger,
    )

    venv.seed(seed)

    if use_wandb:
        wandb_login()
        args_dict.update(cfg)
        wandb.init(project="LIRL", config=args_dict, tags=[], sync_tensorboard=True)

    if args_dict.get("test_ppo"):
        with logger.accumulate_means("ppo"):
            learner.learn(
                total_timesteps=int(1e7),
                reset_num_timesteps=False,
                callback=None,#self.gen_callback,
                #**learn_kwargs,
            )
        return
    
    if args_dict.get("reward_shaping"):
        with logger.accumulate_means("reward_shaping"):
            train_reward_net(reward_net, venv, policy, rollouts, args_dict)
            return

    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, n_eval_episodes, return_episode_rewards=True
    )

    airl_trainer.train(n_rl_train_steps)

    venv.seed(seed)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
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

def main():
    unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"
    
    args_dict = dict(
        copy_weights = True,
        test_ppo = False,
        reward_shaping = True,
        reward_shaping_lr = 5e-3,
        agent_dir = shifted_agent_dir,
        use_wandb = True,
        seed = 42,
        fast = True,
        log_path = "results/",
        # learner arguments:
        # ppo_batch_size = 64,
        # ppo_ent_coef = 0.0,
        # ppo_learning_rate = 0.0005,
        # ppo_gamma = 0.95,
        # ppo_clip_range = 0.1,
        # ppo_vf_coef = 0.1,
        # ppo_n_epochs = 5,
        # ppo_n_steps = 2048,
        # reward_net arguments:
        n_reward_net_epochs = 100,
        reward_hid_sizes = (128,),
        potential_hid_sizes = (128, 128),
        # AIRL arguments:
        demo_batch_size = 2048,
        gen_replay_buffer_capacity = 512,
        n_disc_updates_per_round = 16,
        allow_variable_horizon = True,
        irl_steps_low = int(2e19 + 1),
        irl_steps_high = 2_000_000,
        # Eval args:
        n_eval_episodes = 100,
    )
    
    lirl(args_dict)



if __name__ == "__main__":
    main()
