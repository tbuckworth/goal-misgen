#!/usr/bin/env python3

import sys
import subprocess
import multiprocessing


def run_script(_):
    """
    Run hyperparameter_optimization.py using the Python interpreter
    from the specified virtual environment.
    """
    subprocess.run(["/vol/bitbucket/tfb115/goal-misgen/opvenv/bin/python3.8", "hyperparameter_optimization.py"])


def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_train.py <n>")
        print("Example: python multi_train.py 5")
        sys.exit(1)

    n = int(sys.argv[1])  # Number of parallel processes


    # Verify that the venv's python exists

    with multiprocessing.Pool(processes=n) as pool:
        # Distribute the same venv_python path to each parallel worker
        pool.map(run_script, range(n))


if __name__ == "__main__":
    main()
