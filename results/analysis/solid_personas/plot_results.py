import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

if __name__ == "__main__":

    for game in ['solid', 'fps', 'platform']:
        
        df = pd.read_csv(f'./experiment_results.csv')
        df = df[df['game'] == game]

        models = ['random','PPO', 'DQN', 'Explore']
        colors = cm.viridis(np.linspace(0, 1, len(models)))
        color_dict = dict(zip(models, colors))
        display_dict = {'random': 'Random', 'PPO': 'PPO', 'DQN': 'DQN', 'Explore': 'Go-Explore'}
        
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
                    # model_test = LinearSurrogateModel(game=game, cluster=int(cluster), classifier=True, preference=True)
                    normalized_scores.append(row['score_mean'])
                    normalized_cis.append(row['score_ci'])
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
        plt.savefig(f'./{game}_personas_all_clusters.png')
        plt.show()


       
