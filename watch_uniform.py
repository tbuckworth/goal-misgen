import numpy as np

from helper_local import create_venv_render, DictToArgs


def create_unshifted_venv(args, hyperparameters):
    args.rand_region = 0
    args.random_percent = 0
    return create_venv_render(args, hyperparameters, False)

def create_shifted_venv(args, hyperparameters):
    args.rand_region = 10
    args.random_percent = 10
    if args.env_name == "coinrun":
        #This shouldn't be true in training?
        args.val_env_name = "coinrun_aisc"
    return create_venv_render(args, hyperparameters, True)



def run():
    n_envs = 80
    hyperparameters = dict(
        n_envs=n_envs,
    )
    args = DictToArgs(dict(
        env_name="maze_aisc",
        num_levels=10000,
        start_level=0,
        distribution_mode="hard",
        num_threads=8,#2?
        random_percent=0,
        step_penalty=0,
        key_penalty=0,
        rand_region=0,
        val_env_name=None,
    ))
    env = create_unshifted_venv(args, hyperparameters)
    env_valid = create_shifted_venv(args, hyperparameters)

    sample = lambda env: np.array([env.action_space.sample() for _ in range(n_envs)])
    t = np.zeros(n_envs)
    r = np.zeros(n_envs)
    tv = np.zeros(n_envs)
    rv = np.zeros(n_envs)
    rewards = []
    lens = []
    rewards_v = []
    lens_v = []
    while True:
        step_env(env, r, t, rewards, lens, sample, "Train")
        step_env(env_valid, rv, tv, rewards_v, lens_v, sample, "Valid")

def step_env(env, r, t, rewards, lens, sample, name):
    obs, rew, done, info = env.step(sample(env))
    t += 1
    r += rew
    if done.any():
        rewards += r[done].tolist()
        lens += t[done].tolist()
        r[done] = 0
        t[done] = 0
        mean_r = np.mean(rewards)
        mean_l = np.mean(lens)
        print(f"{name}:\tRewards:{mean_r:.2f}\tLens:{mean_l:.2f}")

if __name__ == "__main__":
    run()
