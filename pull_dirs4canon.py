import argparse
from canon_clean import get_performing_model_dirs


def main(env_name, out_file):
    model_dirs = get_performing_model_dirs(env_name)
    joined_models = '\n'.join(model_dirs)
    print(f"Found these files for canonicalisation - following performing models found:\n{joined_models}")
    # save joined_model to out_file:
    with open(out_file, "w") as f:
        f.write(joined_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="maze_aisc")
    parser.add_argument("--out_file", type=str, default="tmp_out_pull_dirs4canon.out")
    args = parser.parse_args()
    main(args.env_name, args.out_file)
