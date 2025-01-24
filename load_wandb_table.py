import numpy as np
import pandas as pd
from imitation.scripts.config.train_rl import train_rl_ex
from matplotlib import pyplot as plt
from scipy import stats

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass

local_run_names = [
    "breezy-sun-1179",
    "bright-sound-1216",
    "celestial-butterfly-1211",
    "comfy-sky-1176",
    "confused-dragon-1210",
    "copper-oath-1175",
    "Crafted Generaliser",
    "Crafted Misgeneraliser",
    "northern-blaze-1250",
    "dainty-galaxy-1191",
    "celestial-butterfly-1211",
    "copper-oath-1175",
    "celestial-butterfly-1211",
    "deft-puddle-1177",
    "electric-wood-1278",
    "dulcet-snow-1215",
    "fine-aardvark-1230",
    "giddy-star-1232",
    "comfy-sky-1176",
    "deft-puddle-1177",
    "northern-blaze-1250",
    "dulcet-snow-1215",
    "electric-wood-1278",
]

run_names = [
    "absurd-field-1219",
    "breezy-sun-1179",
    "bright-sound-1216",
    "celestial-butterfly-1211",
    "chocolate-galaxy-1214",
    "classic-wood-1246",
    "comfy-sky-1176",
    "confused-dragon-1210",
    "copper-oath-1175",
    "Crafted Generaliser",
    "Crafted Misgeneraliser",
    "crimson-frost-1208",
    "crisp-violet-1221",
    "dainty-galaxy-1191",
    "celestial-butterfly-1211",
    "copper-oath-1175",
    "celestial-butterfly-1211",
    "deep-valley-1212",
    "deft-puddle-1177",
    "devoted-frog-1269",
    "driven-surf-1185",
    "dulcet-snow-1215",
    "fine-aardvark-1230",
    "frosty-valley-1220",
    "giddy-star-1232",
    "glad-flower-1249",
    "comfy-sky-1176",
    "deft-puddle-1177",
    "crimson-frost-1208",
    "pleasant-bee-1190",
    "dulcet-snow-1215",
    "devoted-frog-1269",
]


def graph_wandb_res():
    data = pd.read_csv("data/dist_metrics.csv")

    id_vars = ['Norm', 'Metric', 'mean_episode_rewards', 'val_mean_episode_rewards', 'run']
    df2 = data.pivot(columns='Env', values='Distance', index=id_vars).reset_index()
    df2["Diff"] = df2["Valid"] - df2["Train"]
    df = df2.melt(
        id_vars=id_vars,
        value_vars=["Train", "Valid", "Diff"],
        var_name="Env",
        value_name="Distance",
    )

    # Correcting the legend addition logic to handle it per axis instead of plt level
    # Adjusting the code to ensure proper legend handling and no overlap issues
    unique_combinations = df[['Norm', 'Metric']].drop_duplicates()
    fig, axes = plt.subplots(
        nrows=len(df['Norm'].unique()),
        ncols=len(df['Metric'].unique()),
        figsize=(15, 10),
        sharex=True,
        sharey=False
    )

    axes = axes.flatten()

    for idx, (norm, metric) in enumerate(unique_combinations.itertuples(index=False)):
        ax = axes[idx]
        subset = df[(df['Norm'] == norm) & (df['Metric'] == metric)]

        scatter = ax.scatter(
            subset['val_mean_episode_rewards'],
            subset['Distance'],
            c=subset['Env'].astype('category').cat.codes,
            cmap='viridis',
            alpha=0.7
        )

        ax.set_title(f'Norm: {norm}, Metric: {metric}')
        ax.set_xlabel('val_mean_episode_rewards')
        ax.set_ylabel('Distance')

        if idx == 0:  # Add legend to only the first subplot for simplicity
            env_categories = subset['Env'].astype('category').cat.categories
            color_legend = [
                plt.Line2D(
                    [0], [0], marker='o',
                    color='w', markerfacecolor=plt.cm.viridis(i / len(env_categories)),
                    markersize=10)
                for i in range(len(env_categories))]
            ax.legend(color_legend, env_categories, title="Env", loc='upper right')

    plt.tight_layout()
    plt.show()

    correls = df2.groupby(['Norm', 'Metric'])[['val_mean_episode_rewards', 'Diff']].corr().iloc[0::2, -1].reset_index()
    correls = correls.drop(columns=["Env"])
    correls.rename(columns={'Diff': 'Correlation'}, inplace=True)


def load():
    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": "canon misgen2", "state": "finished"}]}
                    )

    # Collect and filter data
    all_data = []
    for run in runs:
        # table = run.use_artifact("distances:latest").get("distances")
        table = [x for x in run.logged_artifacts()][0].get('distances')
        if table is not None:
            df = pd.DataFrame(data=table.data, columns=table.columns)
            df["mean_episode_rewards"] = run.summary.mean_episode_rewards
            df["val_mean_episode_rewards"] = run.summary.val_mean_episode_rewards
            df["run"] = run.name
            all_data.append(df)

    final = pd.concat(all_data).reset_index(drop=True)
    final.to_csv("data/dist_metrics.csv", index=False)


def loadtemp():
    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": "canon misgen2", "state": "finished"}]}
                    )

    # Collect and filter data
    all_data = [run.config["logdir"] for run in runs if run.name in local_run_names]
    for run in runs:
        if run.name in run_names:
            print(run.config["logdir"])


def load_all():
    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": "canon misgen2", "state": "finished"}]}
                    )

    # Collect and filter data
    all_data = []
    for run in runs:
        # table = run.use_artifact("distances:latest").get("distances")
        table = [x for x in run.logged_artifacts()][0].get('distances')
        if table is not None:
            df = pd.DataFrame(data=table.data, columns=table.columns)
            df["mean_episode_rewards"] = run.summary.mean_episode_rewards
            df["val_mean_episode_rewards"] = run.summary.val_mean_episode_rewards
            df["run"] = run.name
            df["logdir"] = run.config["logdir"]
            df["user"] = run.user

            all_data.append(df)

    final = pd.concat(all_data).reset_index(drop=True)
    final.to_csv("data/dist_metrics_full.csv", index=False)
    print(final)


def load_summary(env= "canon maze hard grouped actions", exclude_crafted=True, tag=None):
    env_name = env
    train_dist_metric = "L2_L2_Train"
    val_dist_metric = "L2_L2_Valid"
    meg_adj = False
    min_train_reward = 9
    if tag is not None:
        if env == "ascent":
            # Original:
            tag = "canon misgen3"
            train_dist_metric = "Loss/l2_normalized_l2_distance_Training"
            val_dist_metric = "Loss/l2_normalized_l2_distance_Validation"
        elif env == "maze":
            env_name = "maze_aisc"
            tag = "canon maze new1"
        elif env == "coinrun":
            tag = "canon coinrun1"
        elif env == "ascent-new":
            tag = "canon ascent1"
        elif env == "ascent-new2":
            tag = "canon ascent new1"
        elif env == "ascent-soft":
            tag = "canon ascent soft2"
        elif env == "ascent-hard":
            tag = "canon ascent hard"
        elif env == "ascent-hard_meg":
            tag = "canon ascent hard+meg3"
            meg_adj = True
        elif env == "ascent-hard_no_meg":
            tag = "canon ascent hard+meg3"
            meg_adj = False
        elif env == "ascent-soft_no_meg":
            tag = "canon ascent soft+meg"
        elif env == "ascent-soft_meg":
            tag = "canon ascent soft+meg"
            meg_adj = True
        elif env == "canon maze hard":
            tag = "canon maze hard"
            meg_adj = False
        elif env == "coinrun hard grouped":
            tag = "canon coinrun hard grouped actions"
            meg_adj = False


    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": tag, "state": "finished"}]}
                    )
    train_rewards = "Mean Training Episode Rewards"
    val_rewards = "Mean Evaluation Episode Rewards"
    train_distance = "Training Distance"
    val_distance = "Evaluation Distance"
    ratio = "Distance Ratio"
    diff = "Distance Difference"
    train_len = "Mean Training Episode Length"
    val_len = "Mean Evaluation Episode Length"
    train_rpl = "Mean Training Reward/Timestep"
    val_rpl = "Mean Evaluation Reward/Timestep"
    train_meg = "Meg Train"
    val_meg = "Meg Valid"
    train_dist_meg = "Distance * Meg - Train"
    val_dist_meg = "Distance * Meg - Valid"

    # Collect and filter data
    all_data = []
    for run in runs:
        row = {}
        if "mean_episode_rewards" not in run.summary.keys():
            continue
        if run.summary.mean_episode_rewards < min_train_reward:
            continue
        if train_dist_metric not in run.summary.keys():
            continue
        row[train_rewards] = run.summary.mean_episode_rewards
        row[val_rewards] = run.summary.val_mean_episode_rewards
        row[train_distance] = run.summary[train_dist_metric]
        row[val_distance] = run.summary[val_dist_metric]
        row["run"] = run.name
        row["logdir"] = run.config["logdir"]
        row[train_len] = run.summary.mean_episode_len
        row[val_len] = run.summary.val_mean_episode_len
        row["architecture"] = run.config["architecture"]
        if meg_adj:
            try:
                row[train_meg] = run.summary["Meg_Train"]
                row[val_meg] = run.summary["Meg_Valid"]
            except Exception as e:
                print("No meg data")
                pass
        all_data.append(row)

    df = pd.DataFrame(all_data)
    if exclude_crafted:
        df = df[df["architecture"]!="crafted-policy"]
    df[ratio] = df[val_distance] / df[train_distance]
    df[diff] = df[val_distance] - df[train_distance]
    df[train_rpl] = df[train_rewards] / df[train_len]
    df[val_rpl] = df[val_rewards] / df[val_len]
    if meg_adj:
        df[train_dist_meg] = df[train_distance]*df[train_meg]
        df[val_dist_meg] = df[val_distance]*df[val_meg]

    df.to_csv(f"data/{env_name}_l2_dist.csv", index=False)

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)  # Adjust figsize as needed
    # for ax, y_metric in zip(axes, [train_distance, val_distance]):
    #     # PLOT:
    #     df.plot.scatter(x=val_rewards, y=y_metric, ax=ax, alpha=0.7)
    #     z = np.polyfit(df[val_rewards], df[y_metric], 1)  # Linear fit (degree=1)
    #     p = np.poly1d(z)
    #     r_squared = np.corrcoef(df[val_rewards], df[y_metric])[0, 1] ** 2
    #     x_smooth = np.linspace(df[val_rewards].min(), df[val_rewards].max(),
    #                            50)  # Adjust the number of points for smoothness
    #     # Compute the corresponding y values for the trendline
    #     y_pred = p(x_smooth)
    #     # Plot the trendline
    #     ax.plot(x_smooth, y_pred,
    #             color='black',
    #             alpha=0.5,
    #             linestyle='-.',
    #             # linewidth=5,
    #             # label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})\n$R^2$ = {r_squared:.2f}',
    #             label=f'$R^2$ = {r_squared:.2f}',
    #             )
    #     ax.legend()
    #     ax.set_title(y_metric)
    #     ax.set_xlabel(val_rewards)
    #     ax.set_ylabel("Distance")
    #     # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"data/{env_name}_distances.png")
    # plt.show()

    # Alternative
    x_metric = val_rewards #val_rpl also interesting
    y_train = train_distance
    y_valid = val_distance
    if meg_adj:
        y_train = train_dist_meg
        y_valid = val_dist_meg
    ax1 = df.plot.scatter(x=x_metric, y=y_train, alpha=0.7, color='b', label=y_train)
    df.plot.scatter(x=x_metric, y=y_valid, alpha=0.7, color='r', ax=ax1, label=y_valid)

    for y_metric, color, linestyle in zip([y_train, y_valid], ['b', 'r'], [':','--']):
        # PLOT:
        # df.plot.scatter(x=x_metric, y=y_metric, ax=ax1, alpha=0.7, color=color)
        z = np.polyfit(df[x_metric], df[y_metric], 1)  # Linear fit (degree=1)
        p = np.poly1d(z)
        r_squared = np.corrcoef(df[x_metric], df[y_metric])[0, 1] ** 2
        x_smooth = np.linspace(df[x_metric].min(), df[x_metric].max(),
                               50)  # Adjust the number of points for smoothness
        # Compute the corresponding y values for the trendline
        y_pred = p(x_smooth)
        # Plot the trendline
        ax1.plot(x_smooth, y_pred,
                 color=color,
                 alpha=0.5,
                 linestyle=linestyle,
                 # linewidth=5,
                 # label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})\n$R^2$ = {r_squared:.2f}',
                 label=f'$R^2$ = {r_squared:.2f}',
                 )
    ax1.set_ylabel("Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/{env_name}_distances_overlapping.png")
    plt.show()

    print(df)
    return
    ax = df.plot.scatter(x=x_metric, y=train_meg, alpha=0.7, color='b', label=train_meg)
    df.plot.scatter(x=x_metric, y=val_meg, alpha=0.7, color='r', ax=ax, label=val_meg)
    plt.show()

    goal_misgen_dists = df[df[val_rewards] < -6][val_distance]
    misgen_dists = df[np.bitwise_and(df[val_rewards] > -6,df[val_rewards] < 7.5)][val_distance]
    gen_dists = df[df[val_rewards] > 7.5][val_distance]

    t_statistic, p_value = stats.ttest_ind(goal_misgen_dists, misgen_dists)

    df[val_distance].plot.hist(bins=100)
    plt.show()


if __name__ == "__main__":
    tag = "Maze Value Original - fixed1"
    tag = "Coinrun_Hard_Canon_corrected"
    load_summary(env=tag, tag=tag)
