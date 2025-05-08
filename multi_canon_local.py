import argparse

from run_canon import run_tags_for_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_files', type=str, nargs='+', help='subject_policy_dirs')
    parser.add_argument('--tag', type=str, default='Test', help='Unique Identifier Wandb Tag')
    args = parser.parse_args()

    run_tags_for_files({args.tag: None}, args.model_files, ignore_errors=True)

