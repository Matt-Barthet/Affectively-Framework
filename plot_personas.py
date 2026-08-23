from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

CSV_PATH = 'results/analysis/solid_personas/Go-Explore-Personas.csv'

PERSONA_LABELS = {1: 'Experts', 2: 'Advanced', 3: 'Beginners', 4: 'Intermediate'}

METRICS = [
    ('trace_sim_playerScore',      'Score Similarity'),
    ('trace_sim_playerSpeed',      'Speed Similarity'),
    ('trace_sim_playerIsOffRoad',  'Off-Road Similarity'),
    ('trace_sim_botPlayerDistance','Distance to Bot Similarity'),
    ('trace_sim_playerStanding',   'Standing Similarity'),
    ('arousal',                    'Arousal'),
]

WEIGHTS = [0.0, 0.5, 1.0]
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_metric_vs_lambda(df, mean_col, ci_col, ylabel, out_path):
    clusters = sorted(df['cluster'].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, cluster in zip(COLORS, clusters):
        rows = df[df['cluster'] == cluster].set_index('weight')
        means = [rows.loc[w, mean_col] if w in rows.index else float('nan') for w in WEIGHTS]
        cis   = [rows.loc[w, ci_col]   if w in rows.index else float('nan') for w in WEIGHTS]
        label = PERSONA_LABELS.get(cluster, f'Persona {cluster}')
        ax.errorbar(WEIGHTS, means, yerr=cis, label=label, color=color,
                    marker='o', capsize=4, linewidth=1.5)
    ax.set_xlabel('λ')
    ax.set_ylabel(ylabel)
    ax.set_xticks(WEIGHTS)
    ax.set_title('')
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    out_dir = Path('results/analysis/solid_personas/plots')
    for key, ylabel in METRICS:
        mean_col = f'{key}_mean'
        ci_col   = f'{key}_ci'
        if mean_col not in df.columns:
            print(f'Skipping {key}: column {mean_col} not in CSV')
            continue
        out_path = out_dir / f'{key}.png'
        plot_metric_vs_lambda(df, mean_col, ci_col, ylabel, out_path)
