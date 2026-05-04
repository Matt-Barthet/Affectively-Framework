import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

from affectively.models.linear_model import LinearSurrogateModel

if __name__ == "__main__":

    for game in ['solid']:
        
        df = pd.read_csv(f'results/analysis/{game}_personas_metrics.csv')

        models = ['','PPO', 'DQN', 'Explore']
        colors = cm.viridis(np.linspace(0, 1, len(models)))
        color_dict = dict(zip(models, colors))
        display_dict = {'PPO': 'PPO', 'DQN': 'DQN', 'Explore': 'Go-Explore'}
        cluster_nums = [3, 4, 2, 1]
        cluster_names = {1: 'Experts', 2: 'Advanced', 3: 'Beginners', 4: 'Intermediates'}
        cluster_names_list = [cluster_names[c] for c in cluster_nums]
        

        for weight in [1.0]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

            scores_means = {model: [] for model in models}
            scores_cis = {model: [] for model in models}
            arousals_means = {model: [] for model in models}
            arousals_cis = {model: [] for model in models}
            lap_time_means = {model: [] for model in models}
            lap_time_cis = {model: [] for model in models}

            for cluster_num in cluster_nums:
                subset = df[(df['weight'] == weight) & (df['cluster'] == cluster_num)]
                model_test = LinearSurrogateModel(game=game, cluster=cluster_num, classifier=True, preference=True)
                for model in models:
                    msubset = subset[subset['model'] == model]
                    scores_means[model].append(msubset['behavior_mean'].values[0] if len(msubset) > 0 else 0)
                    scores_cis[model].append(msubset['behavior_ci'].values[0] if len(msubset) > 0 else 0)
                    arousals_means[model].append(msubset['arousal_mean'].values[0] if len(msubset) > 0 else 0)
                    arousals_cis[model].append(msubset['arousal_ci'].values[0] if len(msubset) > 0 else 0)
                    lt_val = msubset['lap_time_mean'].values[0] if len(msubset) > 0 and 'lap_time_mean' in msubset.columns else 0
                    lap_time_means[model].append(0 if pd.isna(lt_val) else lt_val)
                    ltci_val = msubset['lap_time_ci'].values[0] if len(msubset) > 0 and 'lap_time_ci' in msubset.columns else 0
                    lap_time_cis[model].append(0 if pd.isna(ltci_val) else ltci_val)

            clusters = cluster_names_list
            n_algos = len(models)
            bar_width = 0.8 / n_algos
            x = np.arange(len(clusters))

            for i, algo in enumerate(models):
                print(algo)
                means = scores_means[algo]
                cis = scores_cis[algo]
                ax1.bar(x + i * bar_width, means, bar_width, yerr=cis, capsize=3, label=display_dict[algo], color=color_dict[algo], edgecolor='black')

            ax1.set_xlabel('Clusters', fontsize=13)
            ax1.set_ylabel(r'$R_b$', fontsize=13)
            ax1.set_title('Behavior', fontsize=16)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xticks(x + bar_width * (n_algos - 1) / 2)
            ax1.set_xticklabels(clusters, fontsize=13)
            ax1.tick_params(axis='y', labelsize=14)

            for i, algo in enumerate(models):
                means = arousals_means[algo]
                cis = arousals_cis[algo]
                ax2.bar(x + i * bar_width, means, bar_width, yerr=cis, capsize=3, label=display_dict[algo], color=color_dict[algo], edgecolor='black')

            ax2.set_xlabel('Clusters', fontsize=13)
            ax2.set_ylabel(r'$R_a$', fontsize=13)
            ax2.set_title('Affect', fontsize=16)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xticks(x + bar_width * (n_algos - 1) / 2)
            ax2.set_xticklabels(clusters, fontsize=13)
            ax2.tick_params(axis='y', labelsize=14)

            handles, labels = ax1.get_legend_handles_labels()

            plt.tight_layout()
            fig.subplots_adjust(top=0.85)
            fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1), fontsize=13)
            plt.savefig(f'results/analysis/{game}_personas_weight_{weight}.png')
            # plt.show()

        # Create figure comparing algorithms across weights (lumped across all clusters)
        weights = [0.0, 0.5, 1.0]
        weight_names_list = ['0.0', '0.5', '1.0']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        scores_means = {model: [] for model in models}
        scores_cis = {model: [] for model in models}
        arousals_means = {model: [] for model in models}
        arousals_cis = {model: [] for model in models}

        for weight in weights:
            subset = df[df['weight'] == weight]
            for model in models:
                model_subset = subset[subset['model'] == model]
                # Normalize by cluster score and then average across all clusters
                normalized_scores = []
                normalized_cis = []
                normalized_arousals = []
                normalized_arousal_cis = []
                
                for _, row in model_subset.iterrows():
                    cluster = row['cluster']
                    model_test = LinearSurrogateModel(game=game, cluster=int(cluster), classifier=True, preference=True)
                    normalized_scores.append(row['score_mean'] / model_test.cluster_score[-1])
                    normalized_cis.append(row['score_ci'] / model_test.cluster_score[-1])
                    normalized_arousals.append(row['arousal_mean'])
                    normalized_arousal_cis.append(row['arousal_ci'])
                
                scores_means[model].append(np.mean(normalized_scores) if len(normalized_scores) > 0 else 0)
                scores_cis[model].append(np.mean(normalized_cis) if len(normalized_cis) > 0 else 0)
                arousals_means[model].append(np.mean(normalized_arousals) if len(normalized_arousals) > 0 else 0)
                arousals_cis[model].append(np.mean(normalized_arousal_cis) if len(normalized_arousal_cis) > 0 else 0)

        n_algos = len(models)
        bar_width = 0.8 / n_algos
        x = np.arange(len(weights))

        for i, algo in enumerate(models):
            means = scores_means[algo]
            cis = scores_cis[algo]
            ax1.bar(x + i * bar_width, means, bar_width, yerr=cis, capsize=3, label=display_dict[algo], color=color_dict[algo], edgecolor='black')

        ax1.set_xlabel('$\lambda$', fontsize=13)
        ax1.set_ylabel(r'$R_b$', fontsize=13)
        ax1.set_title('Behavior', fontsize=16)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xticks(x + bar_width * (n_algos - 1) / 2)
        ax1.set_xticklabels(weight_names_list, fontsize=13)
        ax1.tick_params(axis='y', labelsize=14)

        # Plot arousals
        for i, algo in enumerate(models):
            means = arousals_means[algo]
            cis = arousals_cis[algo]
            ax2.bar(x + i * bar_width, means, bar_width, yerr=cis, capsize=3, label=display_dict[algo], color=color_dict[algo], edgecolor='black')

        ax2.set_xlabel('$\lambda$', fontsize=13)
        ax2.set_ylabel(r'$R_a$', fontsize=13)
        ax2.set_title('Affect', fontsize=16)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xticks(x + bar_width * (n_algos - 1) / 2)
        ax2.set_xticklabels(weight_names_list, fontsize=13)
        ax2.tick_params(axis='y', labelsize=14)

        handles, labels = ax1.get_legend_handles_labels()

        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1), fontsize=13)
        plt.savefig(f'results/analysis/{game}_personas_all_clusters.png')
        plt.show()



        cluster_book = pd.read_csv("./affectively/datasets/solid_cluster_book.csv")
        player_data = pd.read_csv("./affectively/datasets/solid_3000ms.csv")

        all_ranking = {}
        all_distance = {}
        all_laptime = {}
        all_offroad = {}

        for cluster in [3,4,2,1]:

            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2)
            ax_left = fig.add_subplot(gs[:, 0])   # spans both rows, left column
            ax_top_right = fig.add_subplot(gs[0, 1])   # top-right
            ax_bot_right = fig.add_subplot(gs[1, 1])   # bottom-right

            player_ids = cluster_book[cluster_book['Cluster'] == cluster]['[control]player_id'].unique()
            ranking = []
            distance = []
            laptime = []
            offroad = []
            for player in player_ids:
                player = player_data[player_data['[control]player_id'] == player]
                ranking.append(min(player['playerStanding'].values[-5:]))
                distance.append(np.mean(player['botPlayerDistance'].values))
                offroad.append(np.mean(player['playerIsOffRoad']))
                laptimes = [0, 0, 0,]
                current_lap = 0
                for idx in range(len(player)):
                    if np.ceil(player['playerLap'].iloc[idx]) != current_lap:
                        current_lap = np.ceil(player['playerLap'].iloc[idx])
                        if current_lap == max(player['playerScore'])//8:
                            laptimes = laptimes[:int(current_lap)]
                            break
                    laptimes[int(current_lap)] += 1
                laptime.append(np.mean(laptimes) * 3)
            all_ranking[cluster] = ranking
            all_distance[cluster] = distance
            all_laptime[cluster] = laptime
            all_offroad[cluster] = offroad

            model = LinearSurrogateModel('Solid', cluster, True, True)
            keys = np.array(list(model.arousal_reward_book.keys()))+1
            ax_left.plot(model.cluster_arousal, label="Mean Arousal")

            label = 0
            for score, timestamp in model.behavior_reward_book.items():
                if label == 0:
                    ax_left.vlines(x=timestamp, ymin=0, ymax=1, alpha=0.3, label='Score Changes', linestyles='--')
                    ax_left.text(timestamp, 0, 'START', ha='center', va='bottom', fontsize=10, alpha=0.6)
                    label += 1
                else:
                    ax_left.vlines(x=timestamp, ymin=0, ymax=1, alpha=0.3, linestyles='--')
                    ax_left.text(timestamp, 0, str(int(score)), ha='center', va='bottom', fontsize=10, alpha=0.6)

            ax_left.vlines(x=600, ymin=0, ymax=1, alpha=0.3, linestyles='--')
            ax_left.text(600, 0, 'END', ha='center', va='bottom', fontsize=10, alpha=0.6)


            ax_top_right.step(list(keys-1), list(np.where(model.plot_labels[1:] == -1, 0, 1)) + [np.where(model.plot_labels == -1, 0, 1)[-1]], where='post')  # plot step changes in arousal
            ax_left.set_title(f"Mean Arousal Trace ({cluster_names[cluster]})", fontsize=13)
            ax_left.set_ylabel("Arousal", fontsize=13)
            ax_left.legend()


            ax_top_right.set_title("Ordinal Arousal Trace", fontsize=13)
            ax_bot_right.set_title(f"Cumulative Ordinal Signal", fontsize=13)
            ax_top_right.set_ylabel("Arousal Change", fontsize=13)
            ax_bot_right.set_ylabel("Cumulative Changes", fontsize=13)

            ax_left.set_xlabel("Time (s)", fontsize=13)
            # ax_top_right.set_xlabel("Score")
            ax_left.set_xticks(np.linspace(0, 600, 5), np.linspace(0, 600, 5, dtype=int)//5)
            ax_bot_right.set_xlabel("Score", fontsize=13)

            xticks = np.arange(0, model.cluster_score[-1]+2, 2, int)
            xlim = (-1, model.cluster_score[-1]+2)
            ax_bot_right.set_xticks(xticks)
            ax_top_right.set_xticks(xticks)
            ax_bot_right.set_xlim(xlim)
            ax_top_right.set_xlim(xlim)
            ax_top_right.set_xticklabels([])
            ax_top_right.set_yticks([0,1], ['-1', '1'])
            for ax in [ax_left, ax_top_right, ax_bot_right]:
                ax.tick_params(axis='both', labelsize=14)
            ax_bot_right.plot(list(keys-1), list(np.cumsum(model.plot_labels)-1))
            fig.align_ylabels([ax_top_right, ax_bot_right])
            plt.tight_layout()
            plt.show()

        personas_df = pd.read_csv(f'results/analysis/{game}_personas_metrics.csv')
        explore_df = personas_df[personas_df['model'] == 'Explore']
        weights = [0.0, 0.5, 1.0]

        metrics = [
            (all_ranking,  'Ranking',        'final_position_mean',   'final_position_ci',   'final_position_raw'),
            (all_distance, 'Bot Distance',   'distance_to_cars_mean', 'distance_to_cars_ci', 'distance_to_cars_raw'),
            (all_laptime,  'Lap Time (s)',   'lap_time_mean',         'lap_time_ci',         'lap_time_raw'),
            (all_offroad,  'Off-Road (%)',   'off_road_mean',         'off_road_ci',         'off_road_raw'),
        ]

        def parse_raw_col(series_val):
            if pd.isna(series_val):
                return None
            try:
                return ast.literal_eval(series_val) if isinstance(series_val, str) else list(series_val)
            except Exception:
                return None

        n_groups = len(cluster_nums)
        n_items = 1 + len(weights)
        item_width = 0.7 / n_items
        group_centers = np.arange(n_groups)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.flatten()

        for ax_idx, (data, label, mean_col, ci_col, raw_col) in enumerate(metrics):
            ax = axes[ax_idx]

            for g_idx, c in enumerate(cluster_nums):
                center = group_centers[g_idx]
                offset_start = center - (n_items - 1) / 2 * item_width

                raw = data[c]
                plot_raw = [5 - v for v in raw] if ax_idx == 0 else raw
                ax.boxplot(plot_raw, positions=[offset_start], widths=item_width * 0.85,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=colors[0], alpha=0.8),
                           medianprops=dict(color='black', linewidth=1.5),
                           whiskerprops=dict(color='black'),
                           capprops=dict(color='black'))

                for w_idx, weight in enumerate(weights):
                    w_subset = explore_df[explore_df['weight'] == weight]
                    match = w_subset[w_subset['cluster'] == c]
                    x_pos = offset_start + (w_idx + 1) * item_width

                    agent_raw = None
                    if len(match) > 0 and raw_col in match.columns:
                        agent_raw = parse_raw_col(match[raw_col].values[0])

                    if agent_raw and len(agent_raw) > 1:
                        plot_agent_raw = [5 - v for v in agent_raw] if ax_idx == 0 else agent_raw
                        ax.boxplot(plot_agent_raw, positions=[x_pos], widths=item_width * 0.85,
                                   patch_artist=True, showfliers=False,
                                   boxprops=dict(facecolor=colors[w_idx + 1], alpha=0.8),
                                   medianprops=dict(color='black', linewidth=1.5),
                                   whiskerprops=dict(color='black'),
                                   capprops=dict(color='black'))
                    else:
                        mean_val = match[mean_col].values[0] if len(match) > 0 else 0
                        ci_val = match[ci_col].values[0] if len(match) > 0 else 0
                        plot_mean = 5 - mean_val if ax_idx == 0 else mean_val
                        ax.errorbar(x_pos, plot_mean, yerr=ci_val, fmt='D', capsize=4,
                                    color=colors[w_idx + 1], markersize=6, markeredgecolor='black',
                                    markeredgewidth=0.5, linewidth=1.5)

            if ax_idx == 0:
                ax.set_yticks([0, 1, 2, 3, 4], ['', '4th', '3rd', '2nd', '1st'])

            ax.set_title(label, fontsize=13)
            ax.set_xticks(group_centers)
            if ax_idx >= 2:
                ax.set_xlabel('Cluster', fontsize=13)
                ax.set_xticklabels(cluster_names_list, fontsize=13)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='y', labelsize=11)

        legend_handles = [Patch(facecolor=colors[0], edgecolor='black', alpha=0.8, label='Humans')]
        for w_idx, weight in enumerate(weights):
            legend_handles.append(Patch(facecolor=colors[w_idx + 1], edgecolor='black', alpha=0.8, label=f'λ={weight}'))
        fig.legend(handles=legend_handles, loc='upper center', ncol=n_items, fontsize=13, bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f'results/analysis/{game}_cluster_metrics.png', bbox_inches='tight')
        plt.show()

        # --- Input histogram comparison: gas pedal and steering ---

        def parse_array_col(val):
            if pd.isna(val) if not isinstance(val, str) else False:
                return None
            try:
                return np.array([float(x) for x in str(val).strip('[]').split()])
            except Exception:
                return None

        # Compute human 3-bin proportions per cluster (bins: negative / ~zero / positive)
        threshold = 0.01
        human_gas = {}
        human_steer = {}
        for c in cluster_nums:
            pids = cluster_book[cluster_book['Cluster'] == c]['[control]player_id'].unique()
            vals_gas = player_data[player_data['[control]player_id'].isin(pids)]['playerGasPedal'].values
            vals_steer = player_data[player_data['[control]player_id'].isin(pids)]['playerSteering'].values
            for store, vals in [(human_gas, vals_gas), (human_steer, vals_steer)]:
                neg = np.mean(vals < -threshold)
                neu = np.mean(np.abs(vals) <= threshold)
                pos = np.mean(vals > threshold)
                store[c] = np.array([neg, neu, pos])

        input_labels = ['Brake', 'Neutral', 'Gas']
        steer_labels = ['Left', 'Straight', 'Right']
        bin_x = np.arange(3)
        n_sources = 1 + len(weights)
        bar_w = 0.7 / n_sources

        fig, axes = plt.subplots(len(cluster_nums), 2, figsize=(12, 3 * len(cluster_nums)))

        for row_idx, c in enumerate(cluster_nums):
            for col_idx, (human_bins, col_name, bin_labels, arr_col) in enumerate([
                (human_gas,   'Gas Pedal', input_labels, 'gas_pedal_mean'),
                (human_steer, 'Steering',  steer_labels, 'steering_mean'),
            ]):
                ax = axes[row_idx, col_idx]

                # Human bars
                ax.bar(bin_x - (n_sources - 1) / 2 * bar_w, human_bins[c], bar_w,
                       color=colors[0], edgecolor='black', alpha=0.8, label='Humans')

                # Agent bars per weight
                for w_idx, weight in enumerate(weights):
                    match = explore_df[(explore_df['weight'] == weight) & (explore_df['cluster'] == c)]
                    if len(match) == 0:
                        continue
                    raw = parse_array_col(match[arr_col].values[0])
                    if raw is None or raw.sum() == 0:
                        continue
                    props = raw / raw.sum()
                    x_pos = bin_x + (w_idx + 1 - (n_sources - 1) / 2) * bar_w
                    ax.bar(x_pos, props, bar_w,
                           color=colors[w_idx + 1], edgecolor='black', alpha=0.8, label=f'λ={weight}')

                ax.set_xticks(bin_x)
                ax.set_xticklabels(bin_labels, fontsize=11)
                ax.set_ylim(0, 1)
                ax.tick_params(axis='y', labelsize=10)
                if col_idx == 0:
                    ax.set_ylabel(cluster_names[c], fontsize=12)
                if row_idx == 0:
                    ax.set_title(col_name, fontsize=13)

        legend_handles = [Patch(facecolor=colors[0], edgecolor='black', alpha=0.8, label='Humans')]
        for w_idx, weight in enumerate(weights):
            legend_handles.append(Patch(facecolor=colors[w_idx + 1], edgecolor='black', alpha=0.8, label=f'λ={weight}'))
        fig.legend(handles=legend_handles, loc='upper center', ncol=n_sources, fontsize=12, bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f'results/analysis/{game}_input_histograms.png', bbox_inches='tight')
        plt.show()

