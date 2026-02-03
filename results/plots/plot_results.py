from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    results_path = 'results/plots/experiment_results.csv'
    df_raw = pd.read_csv(results_path)

    # df_raw = df_raw[df_raw['prediction'] == 'Classification']
    # df_raw = df_raw[df_raw['signal'] == 'Ordinal']

    df_sync  = df_raw[df_raw['reward schedule'] == 'Synchronized']
    df_async = df_raw[df_raw['reward schedule'] == 'Asynchronized']

    signals = ['Interval', 'Ordinal'] 
    predictions = ['Regression', 'Classification']
    schedules = ['Synchronized', 'Asynchronized']
    models = ['Random', 'PPO', 'DQN']
    weights = [0, 0.5, 1]
    targets = ['Minimize', 'Maximize']

    bar_width = 0.2
    x = np.arange(len(weights))  

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # --- Plot for R_b ---
    ax = axs[0]
    for i, model in enumerate(models):
        means = []
        errors = []
        for weight in weights:
            subset = df_sync[(df_sync['model'] == model) & (df_sync['lambda'] == weight)] 
            means.append(subset['R_b'].mean()/24)
            errors.append(subset['R_b_95%'].mean()/24)
        ax.bar(x + i * bar_width, means, width=bar_width, yerr=errors, capsize=5, label=model, edgecolor='black')

    ax.set_xticks(x + bar_width)  
    ax.set_xticklabels(weights)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Mean $R_b$')
    ax.set_title('Mean $R_b$ in Solid')
    ax.set_ylim(-0.1, 1.2)

    ax = axs[1]
    for i, model in enumerate(models):
        means = []
        errors = []
        for weight in weights:
            subset = df_sync[(df_sync['model'] == model) & (df_sync['lambda'] == weight)]
            means.append(subset['R_a'].mean())
            errors.append(subset['R_a_95%'].mean())
        ax.bar(x + i * bar_width, means, width=bar_width, yerr=errors, capsize=5, edgecolor='black')
        
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(weights)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Mean $R_a$')
    ax.set_title('Mean $R_a$ in Solid')
    ax.set_ylim(-0.1, 1.2)

    fig.legend(loc='upper center', ncol=len(models))
    plt.savefig('results/plots/average_score_by_model_type.png')


    for weight in [0, 0.5, 1]:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        from itertools import product
        x_labels = [f"{sig} {pred}" for pred, sig in product(predictions, signals)] 
        x = np.arange(len(x_labels)) 

        ax = axs[0]
        for i, model in enumerate(models):
            means = []
            errors = []
            for prediction, signal in product(predictions, signals):
                subset = df_sync[(df_sync['model'] == model) & (df_sync['prediction'] == prediction) & (df_sync['signal'] == signal) & (df_sync['lambda'] == weight)]
                means.append(subset['R_b'].mean() / 24)
                errors.append(subset['R_b_95%'].mean() / 24)
            ax.bar(x + i * bar_width, means, width=bar_width, yerr=errors, capsize=5, label=model, edgecolor='black')

        ax.set_xticks(x + bar_width) 
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Prediction-Signal')
        ax.set_ylabel('Mean $R_b$')
        ax.set_title(f'Mean $R_b$ with Weight={weight}')
        ax.set_ylim(-0.1, 1.2)

        ax = axs[1]
        for i, model in enumerate(models):
            means = []
            errors = []
            for prediction, signal in product(predictions, signals):
                subset = df_sync[(df_sync['model'] == model) & (df_sync['prediction'] == prediction) & (df_sync['signal'] == signal) & (df_sync['lambda'] == weight)]
                means.append(subset['R_a'].mean())
                errors.append(subset['R_a_95%'].mean())
            ax.bar(x + i * bar_width, means, width=bar_width, yerr=errors, capsize=5, edgecolor='black')

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Prediction-Signal')
        ax.set_ylabel('Mean $R_a$')
        ax.set_title(f'Mean $R_a$ by Model with Weight={weight}')
        ax.set_ylim(-0.1, 1.2)

        fig.legend(loc='upper center', ncol=len(models))
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(len(targets)) 
    x_labels = signals 

    ax = axs[0]
    for i, model in enumerate(models):
        means = []
        errors = []
        for weight in targets:
            subset = df_sync[(df_sync['model'] == model) & (df_sync['arousal target'] == weight)]
            means.append(subset['R_b'].mean()/24)
            errors.append(subset['R_b_95%'].mean()/24)
        ax.bar(x + i * bar_width, means, width=bar_width, yerr=errors, capsize=5, label=model, edgecolor='black')

    ax.set_xticks(x + bar_width)  
    ax.set_xticklabels(targets)
    ax.set_xlabel('Target')
    ax.set_ylabel('Mean $R_b$')
    ax.set_title('Mean $R_b$ in Solid')
    ax.set_ylim(-0.1, 1.2)

    ax = axs[1]
    for i, model in enumerate(models):
        means = []
        errors = []
        for weight in targets:
            subset = df_sync[(df_sync['model'] == model) & (df_sync['arousal target'] == weight)]
            means.append(subset['R_a'].mean())
            errors.append(subset['R_a_95%'].mean())
        ax.bar(x + i * bar_width, means, width=bar_width, yerr=errors, capsize=5, edgecolor='black')
        
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(targets)
    ax.set_xlabel('Target')
    ax.set_ylabel('Mean $R_a$')
    ax.set_title('Mean $R_a$ in Solid')
    ax.set_ylim(-0.1, 1.2)

    fig.legend(loc='upper center', ncol=len(models))
    plt.savefig('results/plots/average_score_by_model_type.png')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

x = np.arange(len(weights))

ax = axs[0]
for i, model in enumerate(models):
    means, errors = [], []
    for weight in weights:
        subset = df_async[(df_async['model'] == model) & (df_async['lambda'] == weight)]
        means.append(subset['R_b'].mean() / 24)
        errors.append(subset['R_b_95%'].mean() / 24)

    ax.bar(x + i * bar_width, means, bar_width,
           yerr=errors, capsize=5, edgecolor='black', label=model)

ax.set_xticks(x + bar_width)
ax.set_xticklabels(weights)
ax.set_xlabel('Weight')
ax.set_ylabel('Mean $R_b$')
ax.set_title('Async: Mean $R_b$')
ax.set_ylim(-0.1, 1.2)

ax = axs[1]
for i, model in enumerate(models):
    means, errors = [], []
    for weight in weights:
        subset = df_async[(df_async['model'] == model) & (df_async['lambda'] == weight)]
        means.append(subset['R_a'].mean())
        errors.append(subset['R_a_95%'].mean())

    ax.bar(x + i * bar_width, means, bar_width,
           yerr=errors, capsize=5, edgecolor='black')

ax.set_xticks(x + bar_width)
ax.set_xticklabels(weights)
ax.set_xlabel('Weight')
ax.set_ylabel('Mean $R_a$')
ax.set_title('Async: Mean $R_a$')
ax.set_ylim(-0.1, 1.2)

fig.legend(loc='upper center', ncol=len(models))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/plots/average_score_async.png')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

x = np.arange(len(weights))
bar_width = 0.35

df_sync = df_sync[df_sync['model'] != 'Random']
df_async = df_async[df_async['model'] != 'Random']

def agg(df_subset, metric):
    return [df_subset[df_subset['lambda'] == w][metric].mean() for w in weights], [df_subset[df_subset['lambda'] == w][f"{metric}_95%"].mean() for w in weights]

print(df_sync[df_sync['lambda'] == 0])

axs[0].bar(x - bar_width/2, np.array(agg(df_sync, 'R_b')[0])/24, bar_width, yerr=np.array(agg(df_sync, 'R_b')[1])/24, label='Sync', edgecolor='black')
axs[0].bar(x + bar_width/2, np.array(agg(df_async, 'R_b')[0])/24, bar_width, yerr=np.array(agg(df_async, 'R_b')[1])/24, label='Async', edgecolor='black')

axs[0].set_xticks(x)
axs[0].set_xticklabels(weights)
axs[0].set_xlabel('Weight')
axs[0].set_ylabel('Mean $R_b$')
axs[0].set_title('Sync vs Async: $R_b$')
axs[0].set_ylim(-0.1, 1.2)

axs[1].bar(x - bar_width/2, agg(df_sync, 'R_a')[0], bar_width, yerr=agg(df_sync, 'R_a')[1], edgecolor='black')
axs[1].bar(x + bar_width/2, agg(df_async, 'R_a')[0], bar_width, yerr=agg(df_async, 'R_a')[1], edgecolor='black')

axs[1].set_xticks(x)
axs[1].set_xticklabels(weights)
axs[1].set_xlabel('Weight')
axs[1].set_ylabel('Mean $R_a$')
axs[1].set_title('Sync vs Async: $R_a$')
axs[1].set_ylim(-0.1, 1.2)

fig.legend(loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/plots/sync_vs_async.png')
plt.show()
plt.close()
