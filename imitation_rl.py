import copy
import os
import time
import warnings

import imitation.util.logger
import torch
from torch import nn

from common import orthogonal_init
from common.env.procgen_wrappers import VecRolloutInfoWrapper, EmbedderWrapper, DummyTerminalObsWrapper
from common.model import IdentityModel, MlpModel, SeqModel, Flatten
from common.policy import PolicyWrapperIRL
from helper_local import create_venv, DictToArgs, initialize_policy, latest_model_path, get_config
from stable_baselines3.common.policies import ActorCriticPolicy
from matplotlib import pyplot as plt

from train import train

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass

# unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
# shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

import numpy as np
from imitation.data import rollout
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor


def get_env_args(cfg):
    # manual implementation for now
    warnings.warn("\n\nUSING UNSHIFTED ENV\n\n")
    env_args = {
        "val_env_name": cfg["val_env_name"],
        "env_name": cfg["env_name"],
        "num_levels": cfg["num_levels"],
        "start_level": cfg["start_level"],
        "distribution_mode": cfg["distribution_mode"],
        "num_threads": cfg["num_threads"],
        # UNSHIFTED ENV!
        "random_percent": 0,  # cfg["random_percent"],
        "step_penalty": cfg["step_penalty"],
        "key_penalty": cfg["key_penalty"],
        "rand_region": cfg["rand_region"],
        "param_name": cfg["param_name"],
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
    while (len(states) < n):
        log_probs, action = predict(policy, obs)
        next_obs, rew, done, info = venv.step(action)
        if start:
            actions = action
            next_states = next_obs
            dones = done
            lp = log_probs
            rews = rew
            start = False
        else:
            actions = np.append(actions, action, 0)
            next_states = np.append(next_states, next_obs, 0)
            dones = np.append(dones, done, 0)
            rews = np.append(rews, rew, 0)
            # lp = np.append(lp, log_probs, 0)
            lp = torch.cat((lp, log_probs), 0)
        states = np.append(states, next_obs, 0)
    return states[:-len(next_obs)], actions, next_states, dones, lp, rews


def generate_data(agent, env, n):
    X, _ = env.reset()
    X = np.expand_dims(X, 0)
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


def reward_forward(reward_net, states, actions, next_states, dones, device):
    s = torch.FloatTensor(states).to(device=device)
    s = Flatten()(s)
    s = nn.ReLU()(s)
    # make one-hot:
    act = torch.LongTensor(actions).to(device=device)
    a = nn.functional.one_hot(act)
    n = torch.FloatTensor(next_states).to(device=device)
    d = torch.FloatTensor(dones).to(device=device)
    return reward_net(s, a, n, d)


def train_reward_net(reward_net, venv, policy, args_dict, cfg, data_size, mini_epochs, save_every=50):
    logdir = os.path.join('logs', 'rew_shaping', cfg["env_name"], cfg["exp_name"])
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{args_dict["seed"]}'
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    np.save(os.path.join(logdir, "args_dict.npy"), args_dict)
    np.save(os.path.join(logdir, "config.npy"), cfg)
    loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=args_dict["reward_shaping_lr"])

    reward_net.train()
    start_weights = [copy.deepcopy(x.detach()) for x in policy.parameters()]

    for epoch in range(args_dict.get("n_reward_net_epochs")):
        # collect data
        # idx = torch.randperm(len(states))
        states, actions, next_states, dones, log_probs, true_rewards = collect_data(policy, venv, n=10000)
        for mini_epoch in range(mini_epochs):
            advantages = reward_forward(reward_net, states, actions, next_states, dones, policy.device)
            act_log_prob = log_probs[torch.arange(len(actions)), torch.tensor(actions)]
            loss = loss_function(advantages, act_log_prob)
            if mini_epoch == 0:
                loss_v = loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(reward_net.parameters(), 40)
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            rewards = reward_forward(reward_net.base, states, actions, next_states, dones, policy.device)
        reward_corr = np.corrcoef(true_rewards, rewards.cpu().numpy())[0, 1]
        adv_corr = np.corrcoef(true_rewards, advantages.detach().cpu().numpy())[0, 1]
        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "loss_v": loss_v,
            "adv_corr": adv_corr,
            "reward_corr": reward_corr,
        })
        print(f"Epoch {epoch}\t"
              f"\tLoss: {loss.item():.4f}"
              f"\tLoss V:{loss_v:.4f}"
              f"\tAdvantage-True Reward Correlation:{adv_corr:.2f}"
              f"\tReward Correlation:{reward_corr:.2f}")
        if epoch % save_every == 0:
            # save reward net:
            torch.save(reward_net.state_dict(), os.path.join(logdir, "reward_net.pth"))

    end_weights = [copy.deepcopy(x.detach()) for x in policy.parameters()]

    for s, e in zip(start_weights, end_weights):
        assert (s == e).all(), "policy has changed!"

    data = [[x, y] for (x, y) in zip(true_rewards, rewards.cpu().numpy())]
    table = wandb.Table(data=data, columns=["True Rewards", "Learned Rewards"])
    wandb.log({"Rewards": wandb.plot.scatter(table, "True Rewards", "Learned Rewards",
                                             title="True Rewards vs Learned Rewards")})

    plt.scatter(true_rewards, rewards.cpu().numpy())
    # plt.scatter(true_rewards, advantages.detach().cpu().numpy())
    # plt.savefig("results/reward_shaping_scatter.png")






def train_next_val_func(next_val_net, venv, policy, args_dict, cfg, data_size, mini_epochs, save_every=50):
    logdir = os.path.join('logs', 'next_val_finding', cfg["env_name"], cfg["exp_name"])
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{args_dict["seed"]}'
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(next_val_net.parameters(), lr=args_dict["reward_shaping_lr"])

    next_val_net.train()
    start_weights = [copy.deepcopy(x.detach()) for x in policy.parameters()]

    timesteps=0
    for epoch in range(args_dict.get("n_reward_net_epochs")):
        states, actions, next_states, dones, log_probs, true_rewards = collect_data(policy, venv, n=data_size)
        next_states = torch.FloatTensor(next_states).to(device=policy.device)
        states = torch.FloatTensor(states).to(device=policy.device)
        # collect data
        # idx = torch.randperm(len(states))
        for mini_epoch in range(mini_epochs):
            _, true_next_value, _ = policy(next_states, None, None)
            true_next_value[dones] = 0
            next_val_hat = next_val_net.value(states)

            loss = loss_function(next_val_hat, true_next_value)
            if mini_epoch == 0:
                loss_v = loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        timesteps += len(states)
        # reward_corr = np.corrcoef(true_rewards, rewards.cpu().numpy())[0, 1]
        # adv_corr = np.corrcoef(true_rewards, advantages.detach().cpu().numpy())[0, 1]
        wandb.log({
            "epoch": epoch,
            "timesteps": timesteps,
            "loss": loss.item(),
            "loss_v": loss_v,
            # "adv_corr": adv_corr,
            # "reward_corr": reward_corr,
        })
        print(
            f"Epoch {epoch}\tLoss: {loss.item():.4f}\tLoss_V: {loss_v:.4f}")
        if epoch % save_every == 0:
            # save reward net:
            torch.save(next_val_net.state_dict(), os.path.join(logdir, "next_val_net.pth"))
            np.save(os.path.join(logdir, "args_dict.npy"), args_dict)
            np.save(os.path.join(logdir, "config.npy"), cfg)
    end_weights = [copy.deepcopy(x.detach()) for x in policy.parameters()]

    for s, e in zip(start_weights, end_weights):
        assert (s == e).all(), "policy has changed!"

    # data = [[x, y] for (x, y) in zip(true_rewards, rewards.cpu().numpy())]
    # table = wandb.Table(data=data, columns=["True Rewards", "Learned Rewards"])
    # wandb.log({"Rewards": wandb.plot.scatter(table, "True Rewards", "Learned Rewards",
    #                                          title="True Rewards vs Learned Rewards")})
    #
    # plt.scatter(true_rewards, rewards.cpu().numpy())
    # # plt.scatter(true_rewards, advantages.detach().cpu().numpy())
    # plt.savefig("results/reward_shaping_scatter.png")



    # save reward net:
    torch.save(next_val_net.state_dict(), os.path.join(logdir, "next_val_net.pth"))
    np.save(os.path.join(logdir, "args_dict.npy"), args_dict)
    np.save(os.path.join(logdir, "config.npy"), cfg)


def lirl(args_dict):
    agent_dir = args_dict.get("agent_dir")
    cfg = get_config(agent_dir)
    args = get_env_args(cfg)
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
    cfg["n_envs"] = args_dict.get("n_envs_override", cfg["n_envs"])
    venv = create_venv(args, cfg)

    model, policy = initialize_policy(device, hyperparameters, venv, venv.observation_space.shape)
    model.device = device
    policy.device = device
    # load policy
    last_model = latest_model_path(agent_dir)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])

    level = args_dict.get("level")
    custom_embedder, value_network = decompose_policy(args_dict.get("new_val_weights"), device, level, model, policy)
    expert = PolicyWrapperIRL(policy, device)

    venv = EmbedderWrapper(venv, embedder=custom_embedder)
    venv = VecRolloutInfoWrapper(venv)
    venv = DummyTerminalObsWrapper(venv)
    venv = VecMonitor(venv=venv)

    # # VecMonitor
    # TODO: make this systematic in a wrapper:
    import gymnasium
    venv.action_space = gymnasium.spaces.discrete.Discrete(15)
    # venv.observation_space = gymnasium.spaces.box.Box(0.0, 1.0, (3, 64, 64))

    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        reward_hid_sizes=reward_hid_sizes,  # (32,),
        potential_hid_sizes=potential_hid_sizes,  # (32, 32),
        use_action=args_dict.get("use_action_reward_net"),
    )
    reward_net.to(device)

    # We generate some expert trajectories, that the discriminator needs to distinguish from the learner's trajectories.
    logger = imitation.util.logger.configure(log_path, ["stdout", "csv", "tensorboard"])

    if not args_dict.get("next_val_shaping") and not args_dict.get("reward_shaping"):
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
                callback=None,  # self.gen_callback,
                # **learn_kwargs,
            )
        return

    if args_dict.get("reward_shaping"):
        with logger.accumulate_means("reward_shaping"):
            train_reward_net(reward_net, venv, policy, args_dict, cfg,
                                data_size=args_dict.get("data_size"),
                                mini_epochs=args_dict.get("mini_epochs"),
                                )
            return

    if args_dict.get("next_val_shaping"):
        with logger.accumulate_means("next_val"):
            train_next_val_func(value_network, venv, policy, args_dict, cfg,
                                data_size=args_dict.get("data_size"),
                                mini_epochs=args_dict.get("mini_epochs"),
                                )
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


def decompose_policy(new_val_weights, device, level, model, policy, modify_policy=True):
    if level in ["block1", "block2", "block3"]:
        embedder_list = [copy.deepcopy(model.block1)]
        if modify_policy:
            del policy.embedder.block1
            policy.embedder.block1 = IdentityModel()
    if level in ["block2", "block3"]:
        embedder_list.append(copy.deepcopy(model.block2))
        if modify_policy:
            del policy.embedder.block2
            policy.embedder.block2 = IdentityModel()
    if level in ["block3"]:
        embedder_list.append(copy.deepcopy(model.block3))
        if modify_policy:
            del policy.embedder.block3
            policy.embedder.block3 = IdentityModel()
    if level in ["embedder"]:
        embedder_list = [copy.deepcopy(policy.embedder)]
        if modify_policy:
            del policy.embedder
            policy.embedder = IdentityModel()
    value_network = copy.deepcopy(policy)
    if new_val_weights:
        value_network.apply(orthogonal_init)
    value_network.to(device=policy.device)
    custom_embedder = SeqModel(embedder_list, device).to(device=policy.device)
    return custom_embedder, value_network


def main():
    unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

    args_dict = dict(
        level="block3",
        n_envs_override=16,
        new_val_weights=False,
        data_size=int(3e4),
        mini_epochs=15,
        copy_weights=True,
        test_ppo=False,
        next_val_shaping=True,
        reward_shaping=False,
        reward_shaping_lr=5e-4,
        agent_dir=shifted_agent_dir,
        use_wandb=True,
        seed=42,
        fast=True,
        log_path="results/",
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
        n_reward_net_epochs=10000,
        reward_hid_sizes=(128, 128, 128),
        potential_hid_sizes=(128, 128, 128, 128),
        use_action_reward_net=False,
        # AIRL arguments:
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        allow_variable_horizon=True,
        irl_steps_low=int(2e19 + 1),
        irl_steps_high=2_000_000,
        # Eval args:
        n_eval_episodes=100,
    )

    lirl(args_dict)


if __name__ == "__main__":
    main()
