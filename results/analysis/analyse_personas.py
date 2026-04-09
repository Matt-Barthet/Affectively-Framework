import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":

    for game in ['solid']:
        df = pd.read_csv(f'results/analysis/{game}_personas.csv')

        models = ['PPO', 'DQN', 'Explore']
        colors = cm.viridis(np.linspace(0, 1, len(models)))
        color_dict = dict(zip(models, colors))
        cluster_nums = [1, 2, 3, 4]
        cluster_names_list = ['Beginners', 'Intermediates', 'Advanced', 'Experts']
        
        for weight in [0.0, 0.5, 1.0]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # fig.suptitle(f'Weight: {weight}')

            scores_means = {model: [] for model in models}
            scores_cis = {model: [] for model in models}
            arousals_means = {model: [] for model in models}
            arousals_cis = {model: [] for model in models}

            for cluster_num in cluster_nums:
                subset = df[(df['weight'] == weight) & (df['cluster'] == cluster_num)]
                for model in models:
                    scores_means[model].append(subset[subset['model'] == model]['score_mean'].values[0] if len(subset[subset['model'] == model]) > 0 else 0)
                    scores_cis[model].append(subset[subset['model'] == model]['score_ci'].values[0] if len(subset[subset['model'] == model]) > 0 else 0)
                    arousals_means[model].append(subset[subset['model'] == model]['arousal_mean'].values[0] if len(subset[subset['model'] == model]) > 0 else 0)
                    arousals_cis[model].append(subset[subset['model'] == model]['arousal_ci'].values[0] if len(subset[subset['model'] == model]) > 0 else 0)

            clusters = cluster_names_list
            n_algos = len(models)
            bar_width = 0.8 / n_algos
            x = np.arange(len(clusters))

            for i, algo in enumerate(models):
                means = scores_means[algo]
                cis = scores_cis[algo]
                ax1.bar(x + i * bar_width, means, bar_width, yerr=cis, capsize=3, label=algo, color=color_dict[algo], edgecolor='black')

            ax1.set_xlabel('Clusters', fontsize=14)
            ax1.set_ylabel(r'$r_b$', fontsize=14)
            ax1.set_title('Behavior', fontsize=16)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xticks(x + bar_width * (n_algos - 1) / 2)
            ax1.set_xticklabels(clusters, fontsize=14)
            ax1.tick_params(axis='y', labelsize=14)

            # Plot arousals
            for i, algo in enumerate(models):
                means = arousals_means[algo]
                cis = arousals_cis[algo]
                ax2.bar(x + i * bar_width, means, bar_width, yerr=cis, capsize=3, label=algo, color=color_dict[algo], edgecolor='black')

            ax2.set_xlabel('Clusters', fontsize=14)
            ax2.set_ylabel(r'$r_a$', fontsize=14)
            ax2.set_title('Affect', fontsize=16)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xticks(x + bar_width * (n_algos - 1) / 2)
            ax2.set_xticklabels(clusters, fontsize=14)
            ax2.tick_params(axis='y', labelsize=14)

            handles, labels = ax1.get_legend_handles_labels()

            plt.tight_layout()
            fig.subplots_adjust(top=0.85)
            fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1), fontsize=14)
            plt.savefig(f'results/analysis/{game}_personas_weight_{weight}.png')
            # plt.show()