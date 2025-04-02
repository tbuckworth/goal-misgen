import argparse
import re
import time
import traceback
from math import floor, log10

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

from gp import bayesian_optimisation

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass

from helper_local import DictToArgs, add_training_args


def get_wandb_performance(hparams, project="Cartpole", id_tag="sa_rew", opt_metric="summary.mean_episode_rewards",
                          entity="ic-ai-safety"):
    wandb_login()
    api = wandb.Api()
    entity, project = entity, project
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [{"tags": id_tag, "state": "finished"}]}
                    )

    summary_list, config_list, name_list, state_list = [], [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    all_dicts = []
    for s, c, n in zip(summary_list, config_list, name_list):
        s_dict = {f"summary.{k}": v for k, v in s.items()}
        s_dict.update({f"config.{k}": v for k, v in c.items()})
        s_dict["name"] = n
        all_dicts.append(s_dict)

    df = pd.DataFrame.from_dict(all_dicts)
    try:
        y = df[opt_metric]
    except KeyError:
        return None, None

    flt = pd.notna(y)
    df = df[flt]
    y = y[flt]
    if len(df) == 0:
        return None, None
    # hp = [x for x in df.columns if re.search("config", x)]
    # hp = [h for h in hp if h not in ["config.wandb_tags"]]
    # hp = [h for h in hp if len(df[h].unique()) > 1]

    hp = [f"config.{h}" for h in hparams if f"config.{h}" in df.columns]
    dfn = df[hp].select_dtypes(include='number')
    return dfn, y


def n_sig_fig(x, n):
    if x == 0 or isinstance(x, bool):
        return x
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def select_next_hyperparameters(X, y, bounds, greater_is_better=True):
    if bounds == {}:
        return {}
    [b.sort() for b in bounds.values()]

    if X is None:
        bound_array = np.array([[x[0], x[-1]] for x in bounds.values()])
        next_params = np.random.uniform(bound_array[:, 0], bound_array[:, 1], (bound_array.shape[0]))
        col_order = bounds.keys()
    else:
        col_order = [re.sub(r"config\.", "", k) for k in X.columns]
        bo = [bounds[k] for k in col_order]

        bound_array = np.array([[x[0], x[-1]] for x in bo])

        xp = X.to_numpy()
        yp = y.to_numpy()

        params = []
        idx = np.random.permutation(len(X.columns))

        n_splits = np.ceil(len(idx) / 2)
        xs = np.array_split(xp[:, idx], n_splits, axis=1)
        bs = np.array_split(bound_array[idx], n_splits, axis=0)

        for x, b in zip(xs, bs):
            param = bayesian_optimisation(x, yp, b, random_search=True, greater_is_better=greater_is_better)
            params += list(param)

        next_params = np.array(params)[np.argsort(idx)]

    int_params = [np.all([isinstance(x, int) for x in bounds[k]]) for k in col_order]
    bool_params = [np.all([isinstance(x, bool) for x in bounds[k]]) for k in col_order]
    next_params = [int(round(v, 0)) if i else v for i, v in zip(int_params, next_params)]
    next_params = [bool(v) if b else v for b, v in zip(bool_params, next_params)]

    hparams = {k: n_sig_fig(next_params[i], 3) for i, k in enumerate(col_order)}

    return hparams


def run_next_hyperparameters(hparams):
    from train import train
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    parser_dict = vars(parser.parse_args())
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    train(args)


def get_project(env_name, exp_name):
    # TODO: make this real
    return "goal-misgen"


def optimize_hyperparams(bounds,
                         fixed,
                         project="Cartpole",
                         id_tag="sa_rew",
                         run_next=run_next_hyperparameters,
                         opt_metric="summary.mean_episode_rewards",
                         greater_is_better=True,
                         abs=False,
                         ):
    strings = {k: v for k, v in fixed.items() if isinstance(v, list) and k != "wandb_tags"}
    string_select = {k: v[np.random.choice(len(v))] for k, v in strings.items()}
    if "env_name" in string_select.keys():
        project = get_project(string_select["env_name"], fixed["exp_name"])
    try:
        X, y = get_wandb_performance(bounds.keys(), project, id_tag, opt_metric)
        if abs:
            y = y.abs()
    except ValueError as e:
        print(f"Error from wandb:\n{e}\nPicking hparams randomly.")
        X, y = None, None

    if X is not None and np.prod(X.shape) == 0:
        X, y = None, None

    hparams = select_next_hyperparameters(X, y, bounds, greater_is_better)

    fh = fixed.copy()
    hparams.update(fh)
    hparams.update(string_select)

    hparams = {k: int(v) if isinstance(v, np.int64) else v for k, v in hparams.items()}
    try:
        run_next(hparams)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        wandb.finish(exit_code=-1)


def init_wandb(cfg, prefix="symbolic_graph"):
    name = np.random.randint(1e5)
    wandb_login()
    wb_resume = "allow"  # if args.model_file is None else "must"
    project = get_project(cfg["env_name"], cfg["exp_name"])
    wandb.init(project=project, config=cfg, sync_tensorboard=True,
               tags=cfg["wandb_tags"], resume=wb_resume, name=f"{prefix}-{name}")


def run_forever(bounds, fixed, run_func, opt_metric, abs=False):
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    fixed["original_start"] = time.asctime()
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, run_func, opt_metric, greater_is_better=True, abs=abs)


def ppo():
    fixed = {
        "detect_nan": False,
        "env_name": 'cartpole',
        "exp_name": 'cartpole',
        "param_name": 'cartpole-mlp',
        "device": "gpu",
        "num_timesteps": int(5e7),
        "seed": [6033, 0, 42, 50, 81],
        "wandb_tags": ["gen_misgen"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": False,
        "anneal_temp": False,
        "entropy_coef": [0, 0.02, 0.1, 0.2],
        "l1_coef": 0,
        "anneal_lr": False,
        "reward_termination": 495,
        "train_pct_ood": [0, 0, 0, 0, 0.01, 0.1, 0.25, 0.5],
        # "hid_dims": [],
        # "dense_rewards": False,
        # "num_rew_updates": 1,
        # "n_pos_states": [10],
        # "reset_rew_model_weights": False,
        # "hidden_dims": [[128, 128, 128], [64, 64], [256, 256], [256, 256, 256], [64]],
        # "rew_learns_from_trusted_rollouts": [False, True],
    }
    bounds = {}
    #     "rew_lr": [0.0001, 0.05],
    # }
    run_forever(bounds, fixed, run_next_hyperparameters, opt_metric="summary.mean_episode_rewards")


def canonicaliser():
    fixed = {
        "detect_nan": False,
        "env_name": 'ascent',
        "exp_name": 'Ascent',
        "param_name": 'ascent-canon',
        "num_timesteps": int(5e7),
        "device": "gpu",
        "seed": [6033, 0, 42, 50, 81],
        "wandb_tags": ["canon misgen2"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        "entropy_coef": [0, 0.01, 0.02],
        # "l1_coef": [0, 0.01, 0.1, 0.3, 0.5, 1.0, 10.0, 100., 200., 1000.],
        "anneal_lr": False,
        "hid_dims": [[3]],
        "dense_rewards": False,
        "n_pos_states": 10,
        "val_epoch": 200,
    }
    bounds = {
        "l1_coef": [0, 1.],
        # "num_timesteps": [int(1e7), int(2e7)],
    }
    run_forever(bounds, fixed, run_next_hyperparameters, opt_metric="summary.val_mean_episode_rewards", abs=True)


if __name__ == "__main__":
    ppo()
