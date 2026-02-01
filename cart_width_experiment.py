#!/usr/bin/env python3
"""
Experiment: Find minimum CartPole x-boundary (h_range) for successful pole balancing.

Trains PPO agents on CartPole with different h_range values (the left-right bound
for the cart position). Default is 2.4 (cart can move ±2.4 units).

We test: 2.4, 1.2, 0.6, 0.3, 0.15 (each halving the previous).

All other CartPole parameters (pole mass, pole length, cart mass, gravity, force)
are kept at their defaults.

Success criterion: mean episode reward >= 490 (out of max 500).
"""

import os
import sys
import csv
import time
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper_local import DictToArgs, add_training_args


def run_single_experiment(h_range_val, seed=42, n_envs=64, num_timesteps=2_000_000):
    """Run a single CartPole training with given h_range."""
    from train import train

    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    parser_dict = vars(parser.parse_known_args()[0])

    hparams = {
        "env_name": "cartpole",
        "exp_name": f"cart_width_h{h_range_val}",
        "param_name": "cartpole-mlp",
        "device": "cpu",
        "num_timesteps": num_timesteps,
        "seed": seed,
        "num_checkpoints": 1,
        "use_wandb": False,
        "use_valid_env": False,
        "render": False,
        "model_file": None,
        "wandb_tags": [],
        "start_level": 100,
        "extra_overrides": [f"h_range={h_range_val}", f"n_envs={n_envs}"],
    }

    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    train(args)


def get_final_stats(logdir):
    """Parse log-append.csv to get final mean reward and timesteps."""
    csv_path = os.path.join(logdir, "log-append.csv")
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None, None
    last_row = rows[-1]
    mean_reward = float(last_row["mean_episode_rewards"])
    timesteps = int(float(last_row["timesteps"]))
    return mean_reward, timesteps


def find_latest_logdir(base_path):
    """Find the most recently created logdir under base_path."""
    if not os.path.exists(base_path):
        return None
    dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def main():
    h_range_values = [2.4, 1.2, 0.6, 0.3, 0.15]
    seed = 42
    n_envs = 64
    num_timesteps = 2_000_000

    results = {}

    for h_range in h_range_values:
        print(f"\n{'=' * 60}")
        print(f"  EXPERIMENT: h_range = {h_range}")
        print(f"  (Cart can move ±{h_range} units, total width = {2*h_range})")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        try:
            run_single_experiment(h_range, seed=seed, n_envs=n_envs, num_timesteps=num_timesteps)
        except Exception as e:
            print(f"ERROR running h_range={h_range}: {e}")
            results[h_range] = {"reward": None, "timesteps": None, "time": None, "error": str(e)}
            continue

        elapsed = time.time() - start_time

        # Find the logdir and extract results
        exp_name = f"cart_width_h{h_range}"
        base = os.path.join("logs", "train", "cartpole", exp_name)
        logdir = find_latest_logdir(base)
        if logdir:
            mean_reward, final_timesteps = get_final_stats(logdir)
            results[h_range] = {
                "reward": mean_reward,
                "timesteps": final_timesteps,
                "time": elapsed,
                "logdir": logdir,
            }
            print(f"\n>>> h_range={h_range}: mean_reward={mean_reward}, timesteps={final_timesteps}, time={elapsed:.1f}s")
        else:
            results[h_range] = {"reward": None, "timesteps": None, "time": elapsed, "error": "logdir not found"}
            print(f"\n>>> h_range={h_range}: could not find logdir")

    # Print summary
    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS SUMMARY: Minimum Cart Width for CartPole")
    print(f"{'=' * 70}")
    print(f"{'h_range':>10} | {'width':>8} | {'mean_reward':>12} | {'timesteps':>12} | {'time':>8} | {'status':>8}")
    print(f"{'-' * 70}")

    for h_range in h_range_values:
        r = results.get(h_range, {})
        reward = r.get("reward")
        timesteps = r.get("timesteps")
        elapsed = r.get("time")
        error = r.get("error")

        if error:
            status = "ERROR"
            reward_str = error[:12]
        elif reward is not None and reward >= 490:
            status = "SOLVED"
            reward_str = f"{reward:.1f}"
        elif reward is not None:
            status = "FAILED"
            reward_str = f"{reward:.1f}"
        else:
            status = "UNKNOWN"
            reward_str = "N/A"

        width_str = f"{2 * h_range:.2f}"
        ts_str = f"{timesteps}" if timesteps else "N/A"
        time_str = f"{elapsed:.0f}s" if elapsed else "N/A"

        print(f"{h_range:>10.2f} | {width_str:>8} | {reward_str:>12} | {ts_str:>12} | {time_str:>8} | {status:>8}")

    print(f"{'=' * 70}")

    # Identify minimum viable width
    solved = [h for h in h_range_values if results.get(h, {}).get("reward", 0) and results[h]["reward"] >= 490]
    if solved:
        min_solved = min(solved)
        print(f"\nMinimum h_range that achieved >= 490 reward: {min_solved}")
        print(f"Minimum total cart width: {2 * min_solved:.2f} units")
    else:
        print("\nNo h_range values achieved >= 490 reward!")

    # Save results to file
    results_path = "cart_width_results.txt"
    with open(results_path, "w") as f:
        f.write("Cart Width Experiment Results\n")
        f.write(f"{'=' * 50}\n")
        for h_range in h_range_values:
            r = results.get(h_range, {})
            f.write(f"h_range={h_range}, reward={r.get('reward')}, timesteps={r.get('timesteps')}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
