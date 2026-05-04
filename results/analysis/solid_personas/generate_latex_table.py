import pandas as pd

weights = [0.0, 0.5, 1.0]
models = ['Go-Explore']
clusters = ['Experts', 'Advanced', 'Beginners', 'Intermediates']
metrics = ['Final Position', 'Lap Time', 'Off Road', 'Dist. to Cars']

dfs = {w: pd.read_csv(f'solid_comparison_weight_{w}.csv').set_index('Cluster') for w in weights}

def row_cells(df, cluster, model):
    return ' & '.join(df.loc[cluster, f'{m} ({model})'] for m in metrics)

def orig_cells(df, cluster):
    return ' & '.join(df.loc[cluster, f'{m} (Original)'] for m in metrics)

col_spec = 'l' + 'r' * len(metrics)
header = ' & '.join(metrics)

lines = []
lines.append(r'\begin{tabular}{' + col_spec + r'}')
lines.append(r'\toprule')
lines.append(r'\textbf{Experiment Setup} & ' + ' & '.join(r'\textbf{' + m + '}' for m in metrics) + r' \\')
lines.append(r'\midrule')

for cluster in clusters:
    lines.append(r'\textbf{' + cluster + r'} & ' + orig_cells(dfs[0.0], cluster) + r' \\')
    for w in weights:
        lines.append(r'$\quad R_{' + str(w) + r'}$ & ' + row_cells(dfs[w], cluster, models[0]) + r' \\')
    lines.append(r'\midrule')

lines[-1] = r'\bottomrule'
lines.append(r'\end{tabular}')

latex = '\n'.join(lines)
print(latex)

with open('solid_comparison_table.tex', 'w') as f:
    f.write(latex)
print('\nSaved to solid_comparison_table.tex')
