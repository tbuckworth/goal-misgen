"""Export one row per PIRC ("canon") run from Weights & Biases to docs/pirc/data/pirc_runs.csv.

Read-only on Weights & Biases. Needs the `wandb` and `pandas` packages and the untracked
`private_login.py` helper at the repository root. The CSV is not tracked (docs/pirc/data/ is
in .gitignore); make_figures.py reads it. Run from the repository root:

    python docs/pirc/export_wandb_runs.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS_CSV = Path(__file__).resolve().parent / "data" / "pirc_runs.csv"
PROJECT = "ic-ai-safety/goal-misgen"

# W&B summary / config key -> column name in the CSV.
SUMMARY_KEYS: dict[str, str] = {
    "L2_L2_Train": "dist_train",
    "L2_L2_Valid": "dist_shifted",
    "mean_episode_rewards": "reward_train",
    "val_mean_episode_rewards": "reward_shifted",
    "mean_episode_len": "len_train",
    "val_mean_episode_len": "len_shifted",
    "Return_Train": "return_train",
    "Return_Valid": "return_shifted",
    "Trusted_Return_Train": "trusted_return_train",
    "Trusted_Return_Valid": "trusted_return_shifted",
    "Trusted_Episodes_Train": "trusted_episodes_train",
    "Trusted_Episodes_Valid": "trusted_episodes_shifted",
    "Loss/value_loss_reward_Training": "td_loss_reward_train",
    "Loss/value_loss_logits_Training": "td_loss_logp_train",
    "Loss/value_loss_valid_logits_Training": "td_loss_logp_heldout_train",
}
CONFIG_KEYS: dict[str, str] = {
    "env_name": "env",
    "soft_canonicalisation": "soft",
    "infinite_value": "infinite_terminal_value",
    "adjust_logprob_mean": "mean_adjusted",
    "centered_logprobs": "centred",
    "trusted_policy": "trusted_policy",
    "n_envs": "n_envs",
    "n_steps": "n_steps",
    "val_epoch": "value_epochs",
    "gamma": "gamma",
    "normalize_rew": "normalize_rew",
    "detach_target": "detach_target",
}
SUBJECT_PATTERN = r"(\d{4}-\d\d-\d\d__\d\d-\d\d-\d\d__seed_\d+)"


def fetch_rows() -> list[dict[str, object]]:
    """Return one dict per run whose config says algo == "canon"."""
    sys.path.insert(0, str(REPO))
    import wandb
    from private_login import wandb_login

    wandb_login()
    runs = wandb.Api(timeout=60).runs(PROJECT, filters={"config.algo": "canon"}, per_page=300, lazy=False)
    rows: list[dict[str, object]] = []
    for run in runs:
        summary = run.summary._json_dict
        tags = run.config.get("wandb_tags")
        row: dict[str, object] = {
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
            "created_at": run.created_at,
            "batch_tag": tags[0] if isinstance(tags, list) and tags else str(tags),
            "model_file": str(run.config.get("model_file")),
        }
        row.update({column: summary.get(key) for key, column in SUMMARY_KEYS.items()})
        row.update({column: run.config.get(key) for key, column in CONFIG_KEYS.items()})
        rows.append(row)
    return rows


def main() -> None:
    """Fetch the runs and write the CSV."""
    table = pd.DataFrame(fetch_rows())
    for column in SUMMARY_KEYS.values():
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["subject_policy"] = table.pop("model_file").str.extract(SUBJECT_PATTERN)[0]
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.sort_values("created_at").to_csv(RUNS_CSV, index=False)
    print(f"wrote {len(table)} runs to {RUNS_CSV}")


if __name__ == "__main__":
    main()
