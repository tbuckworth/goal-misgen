#!/usr/bin/env python3
"""
Follow-up experiment: Pinpoint the minimum CartPole h_range boundary.

From initial experiments:
  h_range=2.4 → peak 491 (converged)
  h_range=1.2 → peak 476 (converged)
  h_range=0.6 → peak 428 (still climbing at 2M steps)
  h_range=0.3 → peak 185 (low ceiling)
  h_range=0.15 → peak 76 (plateaued)

This follow-up:
  1. h_range=0.6 with 5M steps (was still improving)
  2. h_range=0.9 (fill gap between 0.6 and 1.2)
  3. h_range=0.45 (fill gap between 0.3 and 0.6)
"""

import os
import sys
import csv
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper_local import DictToArgs, add_training_args


def run_single_experiment(h_range_val, seed=42, n_envs=64, num_timesteps=5_000_000):
    """Run a single CartPole training with given h_range."""
    from train import train

    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    parser_dict = vars(parser.parse_known_args()[0])

    hparams = {
        "env_name": "cartpole",
        "exp_name": f"cart_width_v2_h{h_range_val}",
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


def get_stats(logdir):
    """Parse log-append.csv to get detailed stats."""
    csv_path = os.path.join(logdir, "log-append.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    rewards = [float(r["mean_episode_rewards"]) for r in rows]
    return {
        "final_reward": rewards[-1],
        "max_reward": max(rewards),
        "max_reward_timestep": int(float(rows[rewards.index(max(rewards))]["timesteps"])),
        "final_timesteps": int(float(rows[-1]["timesteps"])),
        "last10_avg": np.mean(rewards[-10:]),
    }


def find_latest_logdir(base_path):
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
    experiments = [
        (0.6, 5_000_000),   # longer training for borderline case
        (0.9, 5_000_000),   # fill gap between 0.6 and 1.2
        (0.45, 5_000_000),  # fill gap between 0.3 and 0.6
    ]

    seed = 42
    n_envs = 64
    results = {}

    for h_range, num_ts in experiments:
        print(f"\n{'=' * 60}")
        print(f"  EXPERIMENT: h_range = {h_range}, timesteps = {num_ts:,}")
        print(f"  (Cart can move ±{h_range} units, total width = {2*h_range})")
        print(f"{'=' * 60}\n")

        start_time = time.time()
        try:
            run_single_experiment(h_range, seed=seed, n_envs=n_envs, num_timesteps=num_ts)
        except Exception as e:
            print(f"ERROR: {e}")
            results[h_range] = {"error": str(e)}
            continue

        elapsed = time.time() - start_time
        exp_name = f"cart_width_v2_h{h_range}"
        base = os.path.join("logs", "train", "cartpole", exp_name)
        logdir = find_latest_logdir(base)

        if logdir:
            stats = get_stats(logdir)
            if stats:
                stats["time"] = elapsed
                results[h_range] = stats
                print(f"\n>>> h_range={h_range}: max={stats['max_reward']:.1f}, "
                      f"final={stats['final_reward']:.1f}, "
                      f"last10_avg={stats['last10_avg']:.1f}, "
                      f"time={elapsed:.0f}s")

    # Combined results
    print(f"\n\n{'=' * 80}")
    print(f"  COMBINED RESULTS (Initial + Follow-up)")
    print(f"{'=' * 80}")
    print(f"{'h_range':>8} | {'width':>6} | {'max_rew':>8} | {'final_rew':>10} | {'last10_avg':>10} | {'timesteps':>12}")
    print(f"{'-' * 80}")

    # Include initial results
    all_results = {
        2.4: {"max_reward": 491.0, "final_reward": 469.0, "last10_avg": 465.8, "final_timesteps": 2015232},
        1.2: {"max_reward": 475.6, "final_reward": 450.1, "last10_avg": 444.6, "final_timesteps": 2015232},
        0.3: {"max_reward": 185.2, "final_reward": 132.5, "last10_avg": 146.4, "final_timesteps": 2015232},
        0.15: {"max_reward": 75.8, "final_reward": 65.0, "last10_avg": 67.7, "final_timesteps": 2015232},
    }
    all_results.update(results)

    for h_range in sorted(all_results.keys(), reverse=True):
        r = all_results[h_range]
        if "error" in r:
            print(f"{h_range:>8.2f} | {2*h_range:>6.2f} | ERROR: {r['error']}")
            continue
        print(f"{h_range:>8.2f} | {2*h_range:>6.2f} | {r['max_reward']:>8.1f} | {r['final_reward']:>10.1f} | "
              f"{r['last10_avg']:>10.1f} | {r.get('final_timesteps', 'N/A'):>12}")

    # Save results
    results_path = "cart_width_results_v2.txt"
    with open(results_path, "w") as f:
        f.write("Cart Width Experiment Results (Combined)\n")
        f.write(f"{'=' * 60}\n\n")
        for h_range in sorted(all_results.keys(), reverse=True):
            r = all_results[h_range]
            if "error" in r:
                f.write(f"h_range={h_range}: ERROR\n")
                continue
            f.write(f"h_range={h_range:.2f}: max_reward={r['max_reward']:.1f}, "
                    f"final={r['final_reward']:.1f}, last10_avg={r['last10_avg']:.1f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
