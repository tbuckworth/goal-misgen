import argparse

from procgen_canon_fixed import run_canonicalisation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="maze_aisc")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--config", type=str, default="soft_inf")
    parser.add_argument("--suffix", type=str)
    args = parser.parse_args()
    run_canonicalisation(args.model_dir, args.env_name, args.config, args.suffix)
