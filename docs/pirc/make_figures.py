"""Draw the three figures in docs/pirc/README.md.

- figures/pirc_by_environment.png and figures/cartpole_constants.png are drawn from
  docs/pirc/data/pirc_runs.csv, which export_wandb_runs.py writes (one row per PIRC run).
- figures/pirc_is_a_correlation.png is simulated: on random episodic tasks, the exact PIRC
  distance against sqrt(2 - 2 * corr), where corr is the correlation over sampled
  trusted-policy episodes between the true return and the subject's log-likelihood.

Needs numpy, pandas, scipy and matplotlib. Run from the repository root:

    python docs/pirc/make_figures.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats

from verify_maths import (
    MDP,
    as_reward,
    canonicalise,
    correlation_distance,
    occupancy,
    policy_value,
    random_episodic_mdp,
    sampled_episode_sums,
    soft_optimal,
    uniform_policy,
    unit_distance,
)

HERE = Path(__file__).resolve().parent
RUNS_CSV = HERE / "data" / "pirc_runs.csv"
FIGURES = HERE / "figures"

# Reference palette, slots 1 and 2 (validated colour-blind safe as a pair).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
UNRELATED = float(np.sqrt(2.0))  # distance between two unrelated, normalised signals


@dataclass(frozen=True)
class Panel:
    """One small multiple: which runs to show and how to title them."""

    title: str
    env: str
    batch_tags: tuple[str, ...] | None  # None = every finished run in the env
    note: str


PANELS: tuple[Panel, ...] = (
    Panel(
        "Ascender (toy gridworld)",
        "ascent",
        ("Ascent_Soft_Inf2",),
        "the setting in the Jan 2025 draft",
    ),
    Panel(
        "CartPole, Jan 2025 draft setting",
        "cartpole",
        ("Cartpole_Soft_Inf_Mean_Adjusted",),
        "implied reward shifted to the true mean",
    ),
    Panel(
        "CartPole, May 2025 rerun",
        "cartpole",
        ("cartpole_soft_inf_6556",),
        "cleaned-up pipeline, no mean shift, 96 policies",
    ),
    Panel(
        "Maze (Procgen, pixels)",
        "maze_aisc",
        ("Maze_VOrig_Soft_Inf",),
        "the draft's figure used 4 of these (r = \u22120.93)",
    ),
    Panel(
        "CoinRun (Procgen, pixels)",
        "coinrun",
        None,
        "all 14 batches, Nov 2024 to Jul 2025",
    ),
)


def load_runs(path: Path) -> pd.DataFrame:
    """Return finished runs that logged a distance in both environments."""
    runs = pd.read_csv(path)
    done = runs[(runs["state"] == "finished") & runs["dist_train"].notna() & runs["dist_shifted"].notna()]
    return done[done["reward_shifted"].notna()].copy()


def select(runs: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """Pick the runs belonging to one panel."""
    chosen = runs[runs["env"] == panel.env]
    if panel.batch_tags is not None:
        chosen = chosen[chosen["batch_tag"].isin(panel.batch_tags)]
    return chosen


def draw_panel(ax: Axes, runs: pd.DataFrame, panel: Panel) -> None:
    """Scatter distance against post-shift score, with a fit to the shifted-environment points."""
    x = runs["reward_shifted"].to_numpy(dtype=float)
    shifted = runs["dist_shifted"].to_numpy(dtype=float)
    train = runs["dist_train"].to_numpy(dtype=float)

    ax.axhline(UNRELATED, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.scatter(x, train, s=26, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.scatter(x, shifted, s=34, marker="^", color=ORANGE, edgecolors=SURFACE, linewidths=0.6, zorder=4)

    r, p = stats.pearsonr(x, shifted)
    slope, intercept = np.polyfit(x, shifted, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2.0, zorder=2)

    ax.set_title(panel.title, fontsize=10.5, color=INK, loc="left", fontweight="bold", pad=32)
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
    r_text = f"{r:+.2f}".replace("-", "\u2212")
    ax.text(
        0.0,
        1.03,
        f"r = {r_text}, {p_text}, n = {len(x)}\n{panel.note}",
        transform=ax.transAxes,
        fontsize=8.6,
        color=INK_SECONDARY,
        va="bottom",
        ha="left",
        linespacing=1.35,
    )
    ax.set_ylim(0.45, 1.9)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.set_xlabel("score after the shift (mean episode reward)", fontsize=8.6, color=INK_SECONDARY)
    ax.set_ylabel("PIRC distance", fontsize=8.6, color=INK_SECONDARY)


def draw_key(ax: Axes) -> None:
    """Use the spare grid cell as the legend and reading guide."""
    ax.set_axis_off()
    ax.scatter([0.06], [0.90], s=44, marker="^", color=ORANGE, transform=ax.transAxes, clip_on=False)
    ax.text(0.13, 0.90, "distance measured after the shift", transform=ax.transAxes, fontsize=9.2, va="center", color=INK)
    ax.scatter([0.06], [0.78], s=36, marker="o", color=BLUE, transform=ax.transAxes, clip_on=False)
    ax.text(0.13, 0.78, "distance measured in training", transform=ax.transAxes, fontsize=9.2, va="center", color=INK)
    ax.plot([0.02, 0.10], [0.66, 0.66], color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (2, 3)), transform=ax.transAxes, clip_on=False)
    ax.text(0.13, 0.66, "1.41: what two unrelated rewards give", transform=ax.transAxes, fontsize=9.2, va="center", color=INK)
    guide = (
        "Each point is one PIRC run on one\n"
        "trained policy.\n\n"
        "If PIRC works, the orange triangles\n"
        "slope down to the right (worse\n"
        "policies sit further from the true\n"
        "reward) and the blue circles stay\n"
        "low and flat."
    )
    ax.text(0.0, 0.52, guide, transform=ax.transAxes, fontsize=9.2, va="top", color=INK_SECONDARY, linespacing=1.4)


def draw_by_environment(path: Path) -> None:
    """Small multiples of distance against score after the shift, one panel per environment."""
    runs = load_runs(RUNS_CSV)
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 10.0), facecolor=SURFACE)
    for ax, panel in zip(axes.flat, PANELS):
        draw_panel(ax, select(runs, panel), panel)
    draw_key(axes.flat[len(PANELS)])
    fig.suptitle(
        "PIRC tracks how policies fare after the shift in the toy task,\nbut not reliably in CartPole or Maze, and not at all in CoinRun",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.6, w_pad=1.6)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


SEED = 1
N_MDPS = 40
EPISODES = 100_000

# (label, infinite_terminal_value, mean_adjusted) for the soft reading of log pi in CartPole.
CARTPOLE_VARIANTS: tuple[tuple[str, bool, bool], ...] = (
    ("log π as it is", False, False),
    ("end state treated as never-ending", True, False),
    ("never-ending end state + mean shift\n(the setting in the Jan 2025 draft)", True, True),
    ("mean shifted to the true reward's mean", False, True),
)


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def identity_points(rng: np.random.Generator) -> tuple[list[float], list[float], list[float]]:
    """For random MDPs and subjects, return (distance from sampled correlation, exact PIRC, gamma)."""
    from_correlation, exact, gammas = [], [], []
    for i in range(N_MDPS):
        gamma = 1.0 if i % 2 == 0 else 0.99
        mdp = random_episodic_mdp(rng, n_states=8, n_actions=3, gamma=gamma, stop=0.08)
        other = random_episodic_mdp(rng, n_states=8, n_actions=3, gamma=gamma, stop=0.08).reward
        mix = rng.uniform(-1.0, 1.0)
        subject_mdp = MDP(mdp.transition, mix * mdp.reward + (1.0 - abs(mix)) * other, gamma, mdp.terminal)
        implied = np.log(soft_optimal(subject_mdp, alpha=1.0)[0])
        trusted = uniform_policy(mdp)
        start = np.zeros(mdp.n_states)
        start[0] = 1.0
        implied_reward = as_reward(implied, mdp)
        c_implied = canonicalise(mdp, implied_reward, policy_value(mdp, trusted, implied_reward), expected=False)
        c_true = canonicalise(mdp, mdp.reward, policy_value(mdp, trusted, mdp.reward), expected=False)
        returns, _ = sampled_episode_sums(rng, mdp, trusted, start, EPISODES, [mdp.reward, implied_reward], gamma)
        from_correlation.append(correlation_distance(returns[0], returns[1]))
        exact.append(unit_distance(c_implied, c_true, occupancy(mdp, trusted, start, 1.0)))
        gammas.append(gamma)
    return from_correlation, exact, gammas


def draw_identity(path: Path) -> None:
    x, y, gammas = identity_points(np.random.default_rng(SEED))
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    style_axes(ax)
    ax.plot([0, 2], [0, 2], color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1, label="equal")
    for gamma, colour, marker in ((1.0, BLUE, "o"), (0.99, ORANGE, "^")):
        keep = [g == gamma for g in gammas]
        ax.scatter(
            np.array(x)[keep], np.array(y)[keep], s=46, marker=marker, color=colour, edgecolors=SURFACE,
            linewidths=0.7, zorder=3, label=f"discount {gamma:g} ({sum(keep)} random tasks)",
        )
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 2.05)
    ax.set_xlabel(
        "√(2 − 2·correlation), where the correlation is taken over 100,000 episodes of the\n"
        "trusted policy, between true return and the subject's log-likelihood of the actions",
        fontsize=9, color=INK_SECONDARY,
    )
    ax.set_ylabel("PIRC distance, computed exactly with value functions", fontsize=9, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle(
        "The PIRC distance is a correlation between true return\nand how likely the subject finds the trusted policy's actions",
        fontsize=12, color=INK, fontweight="bold", x=0.03, ha="left", y=0.975,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}  (max |difference| = {max(abs(a - b) for a, b in zip(x, y)):.3f})")


def draw_cartpole(path: Path) -> None:
    runs = pd.read_csv(RUNS_CSV)
    runs = runs[(runs["state"] == "finished") & (runs["env"] == "cartpole") & (runs["soft"] == True)]  # noqa: E712
    runs = runs[runs["dist_train"].notna() & runs["dist_shifted"].notna()]
    fig, ax = plt.subplots(figsize=(7.6, 4.3), facecolor=SURFACE)
    style_axes(ax)
    ax.axvline(UNRELATED, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    labels = []
    for row, (label, absorbing, shifted_mean) in enumerate(CARTPOLE_VARIANTS):
        chosen = runs[(runs["infinite_terminal_value"] == absorbing) & (runs["mean_adjusted"] == shifted_mean)]
        labels.append(f"{label}\n{len(chosen)} runs")
        series = (("dist_train", BLUE, "o", 0.13, "training environment"), ("dist_shifted", ORANGE, "^", -0.13, "shifted environment"))
        for column, colour, marker, offset, name in series:
            mean, sd = chosen[column].mean(), chosen[column].std()
            ax.errorbar(
                mean, row + offset, xerr=sd, fmt=marker, color=colour, markersize=8, markeredgecolor=SURFACE,
                elinewidth=2, capsize=0, zorder=3, label=name if row == 0 else None,
            )
            ax.text(mean, row + offset + (0.2 if offset > 0 else -0.2), f"{mean:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("PIRC distance: mean over runs, bar = one standard deviation", fontsize=9, color=INK_SECONDARY)
    top = ax.secondary_xaxis("top", functions=(lambda d: 1 - np.asarray(d) ** 2 / 2, lambda r: np.sqrt(np.clip(2 - 2 * np.asarray(r), 0, None))))
    top.set_xticks([1.0, 0.5, 0.0, -0.5, -1.0])
    top.set_xlabel("the same thing as a correlation with the true reward", fontsize=9, color=INK_SECONDARY)
    top.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    top.spines["top"].set_visible(False)
    ax.text(UNRELATED + 0.02, len(labels) - 0.48, "unrelated", fontsize=8.5, color=INK_SECONDARY, va="bottom")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle(
        "CartPole: the handling of constants moves the distance from\nagreeing with the true reward to disagreeing with it",
        fontsize=12, color=INK, fontweight="bold", x=0.02, ha="left", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_by_environment(FIGURES / "pirc_by_environment.png")
    draw_identity(FIGURES / "pirc_is_a_correlation.png")
    draw_cartpole(FIGURES / "cartpole_constants.png")


if __name__ == "__main__":
    main()
