import pandas as pd
import numpy as np

# Read the datasets
cluster_book = pd.read_csv('fps_cluster_book.csv')
fps_data = pd.read_csv('fps_3000ms.csv')

# Merge the datasets on player_id
merged_data = fps_data.merge(cluster_book, on='[control]player_id', how='inner')
merged_data['playerKillCount'] = np.ceil(merged_data['playerKillCount']).astype(int)  # Ensure kill counts are integers
# Calculate game length and get final stats for each player
player_stats = merged_data.groupby(['[control]player_id', 'Cluster']).agg({
    'playerScore': 'max',  # Final score
    'playerKillCount': 'sum'  # Final kill count
}).reset_index()

# Get trace count separately to avoid column name conflict
trace_counts = merged_data.groupby(['[control]player_id', 'Cluster']).size().reset_index(name='trace_count')
player_stats = player_stats.merge(trace_counts, on=['[control]player_id', 'Cluster'])

# Calculate game length
player_stats['game_length_seconds'] = np.clip(player_stats['trace_count'], 0, 40) * 3  # Each entry is 3 seconds
player_stats['game_length_minutes'] = player_stats['game_length_seconds'] / 60

# Determine wins (assuming 500+ score is a win)
player_stats['is_winner'] = (player_stats['playerScore'] >= 500).astype(int)

# Calculate cluster-level statistics
# First ensure we have one row per player (already done by groupby above)
# Now aggregate by cluster
cluster_stats = player_stats.groupby('Cluster').agg({
    'game_length_minutes': ['mean', 'std', 'min', 'max'],
    'playerKillCount': 'mean',
    'is_winner': ['sum', 'mean'],
    '[control]player_id': 'count'  # Number of unique players
}).round(2)

# Flatten column names
cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
cluster_stats.rename(columns={
    'game_length_minutes_mean': 'Avg Game Length (min)',
    'game_length_minutes_std': 'Std Dev Length (min)',
    'game_length_minutes_min': 'Min Game Length (min)',
    'game_length_minutes_max': 'Max Game Length (min)',
    'playerKillCount_mean': 'Avg Kills per Player',
    'is_winner_sum': 'Total Wins',
    'is_winner_mean': 'Win Rate',
    '[control]player_id_count': 'Number of Players'
}, inplace=True)

# Convert win rate to percentage
cluster_stats['Win Rate (%)'] = cluster_stats['Win Rate'] * 100
cluster_stats.drop('Win Rate', axis=1, inplace=True)

# Import scipy for confidence interval calculations
from scipy import stats

# Calculate 95% confidence intervals for each statistic
def calculate_ci(data, confidence=0.95):
    """Calculate confidence interval for a given dataset"""
    n = len(data)
    if n <= 1:
        return 0, 0
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - ci, mean + ci

# Calculate confidence intervals for each cluster
ci_stats = pd.DataFrame()
for cluster in sorted(cluster_stats.index):
    cluster_data = player_stats[player_stats['Cluster'] == cluster]
    
    # Game length CI
    gl_lower, gl_upper = calculate_ci(cluster_data['game_length_minutes'])
    
    # Kills CI
    kills_lower, kills_upper = calculate_ci(cluster_data['playerKillCount'])
    
    # Win rate CI (using binomial proportion)
    n_players = len(cluster_data)
    n_wins = cluster_data['is_winner'].sum()
    win_rate = n_wins / n_players
    
    # Wilson score interval for binomial proportion
    z = 1.96  # 95% confidence
    denominator = 1 + z**2/n_players
    centre_adjusted_probability = (n_wins + z*z/2) / (n_players + z*z)
    adjusted_standard_error = np.sqrt((win_rate*(1 - win_rate) + z*z/(4*n_players)) / n_players)
    
    wr_lower = max(0, (centre_adjusted_probability - z*adjusted_standard_error) / denominator) * 100
    wr_upper = min(1, (centre_adjusted_probability + z*adjusted_standard_error) / denominator) * 100
    
    ci_stats = pd.concat([ci_stats, pd.DataFrame({
        'Cluster': [cluster],
        'Game Length CI': [(gl_lower, gl_upper)],
        'Kills CI': [(kills_lower, kills_upper)],
        'Win Rate CI': [(wr_lower, wr_upper)]
    })], ignore_index=True)

# Create formatted table with 95% CIs as ±
formatted_stats = pd.DataFrame()
for cluster in sorted(cluster_stats.index):
    cluster_idx = ci_stats[ci_stats['Cluster'] == cluster].index[0]
    gl_ci = ci_stats.loc[cluster_idx, 'Game Length CI']
    kills_ci = ci_stats.loc[cluster_idx, 'Kills CI']
    wr_ci = ci_stats.loc[cluster_idx, 'Win Rate CI']
    
    # Calculate the ± values (half the width of the confidence interval)
    gl_mean = cluster_stats.loc[cluster, 'Avg Game Length (min)']
    gl_pm = (gl_ci[1] - gl_ci[0]) / 2
    
    kills_mean = cluster_stats.loc[cluster, 'Avg Kills per Player']
    kills_pm = (kills_ci[1] - kills_ci[0]) / 2
    
    wr_mean = cluster_stats.loc[cluster, 'Win Rate (%)']
    wr_pm = (wr_ci[1] - wr_ci[0]) / 2
    
    formatted_stats = pd.concat([formatted_stats, pd.DataFrame({
        'Cluster': [int(cluster)],
        'Players': [int(cluster_stats.loc[cluster, 'Number of Players'])],
        'Game Length (min)': [f"{gl_mean:.2f} ± {gl_pm:.2f}"],
        'Avg Kills': [f"{kills_mean:.2f} ± {kills_pm:.2f}"],
        'Wins': [int(cluster_stats.loc[cluster, 'Total Wins'])],
        'Win Rate': [f"{wr_mean:.1f}% ± {wr_pm:.1f}%"]
    })], ignore_index=True)

# Print the results
print("FPS Player Statistics by Cluster (with 95% Confidence Intervals)")
print("=" * 100)
print(formatted_stats.to_string(index=False))
print("\n")

# Generate LaTeX table with confidence intervals
print("LaTeX Table Code (with 95% CIs):")
print("-" * 60)
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{FPS Player Statistics by Cluster with 95\\% Confidence Intervals}")
print("\\label{tab:fps_cluster_stats_ci}")
print("\\begin{tabular}{cccccc}")
print("\\hline")
print("Cluster & Players & Game Length (min) & Avg Kills & Wins & Win Rate \\\\")
print("\\hline")
for _, row in formatted_stats.iterrows():
    # Clean up the formatting for LaTeX
    game_length = row['Game Length (min)'].replace('%', '\\%')
    kills = row['Avg Kills'].replace('%', '\\%')
    win_rate = row['Win Rate'].replace('%', '\\%')
    print(f"{row['Cluster']} & {row['Players']} & {game_length} & {kills} & {row['Wins']} & {win_rate} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")
print("\n")

# Print detailed summary statistics
print("Detailed Statistics by Cluster")
print("-" * 60)
for cluster in sorted(cluster_stats.index):
    cluster_data = player_stats[player_stats['Cluster'] == cluster]
    print(f"\nCluster {int(cluster)}:")
    print(f"  Number of players: {cluster_stats.loc[cluster, 'Number of Players']}")
    print(f"\n  Game Length:")
    print(f"    Average: {cluster_stats.loc[cluster, 'Avg Game Length (min)']:.2f} ± {cluster_stats.loc[cluster, 'Std Dev Length (min)']:.2f} minutes")
    print(f"    Range: {cluster_stats.loc[cluster, 'Min Game Length (min)']:.2f} - {cluster_stats.loc[cluster, 'Max Game Length (min)']:.2f} minutes")
    
    # Game length percentiles
    percentiles = cluster_data['game_length_minutes'].quantile([0.25, 0.5, 0.75])
    print(f"    25th percentile: {percentiles[0.25]:.2f} minutes")
    print(f"    Median: {percentiles[0.5]:.2f} minutes")
    print(f"    75th percentile: {percentiles[0.75]:.2f} minutes")
    
    print(f"\n  Performance:")
    print(f"    Average kills per player: {cluster_stats.loc[cluster, 'Avg Kills per Player']:.2f}")
    print(f"    Total wins: {int(cluster_stats.loc[cluster, 'Total Wins'])}")
    print(f"    Win rate: {cluster_stats.loc[cluster, 'Win Rate (%)']:.1f}%")
    
    # Additional kill statistics
    kill_percentiles = cluster_data['playerKillCount'].quantile([0.25, 0.5, 0.75])
    print(f"    Kill distribution (25th/50th/75th): {kill_percentiles[0.25]:.0f} / {kill_percentiles[0.5]:.0f} / {kill_percentiles[0.75]:.0f}")

# Create a comprehensive summary DataFrame
comprehensive_stats = pd.DataFrame()
for cluster in sorted(cluster_stats.index):
    cluster_data = player_stats[player_stats['Cluster'] == cluster]
    length_percentiles = cluster_data['game_length_minutes'].quantile([0.25, 0.5, 0.75])
    kill_percentiles = cluster_data['playerKillCount'].quantile([0.25, 0.5, 0.75])
    
    comprehensive_stats = pd.concat([comprehensive_stats, pd.DataFrame({
        'Cluster': [cluster],
        'Players': [len(cluster_data)],
        'Avg Length (min)': [cluster_data['game_length_minutes'].mean()],
        'Median Length (min)': [length_percentiles[0.5]],
        'Avg Kills': [cluster_data['playerKillCount'].mean()],
        'Median Kills': [kill_percentiles[0.5]],
        'Total Wins': [cluster_data['is_winner'].sum()],
        'Win Rate (%)': [cluster_data['is_winner'].mean() * 100],
        'Min Length (min)': [cluster_data['game_length_minutes'].min()],
        'Max Length (min)': [cluster_data['game_length_minutes'].max()]
    })], ignore_index=True)

# Round the comprehensive stats
comprehensive_stats = comprehensive_stats.round(2)

# Optional: Save results to CSV
cluster_stats.to_csv('cluster_statistics_with_performance.csv')
comprehensive_stats.to_csv('cluster_comprehensive_summary.csv')
formatted_stats.to_csv('cluster_formatted_summary.csv', index=False)
print("\nResults saved to:")
print("  - 'cluster_statistics_with_performance.csv' (full statistics)")
print("  - 'cluster_comprehensive_summary.csv' (comprehensive summary)")
print("  - 'cluster_formatted_summary.csv' (formatted with ± notation)")

# Generate more detailed LaTeX table with additional statistics
print("\n\nDetailed LaTeX Table Code (with percentiles):")
print("-" * 60)
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Detailed FPS Player Statistics by Cluster}")
print("\\label{tab:fps_cluster_detailed}")
print("\\begin{tabular}{cccccccc}")
print("\\hline")
print("\\multirow{2}{*}{Cluster} & \\multirow{2}{*}{Players} & \\multicolumn{3}{c}{Game Length (min)} & \\multicolumn{2}{c}{Performance} & \\multirow{2}{*}{Win Rate} \\\\")
print("\\cline{3-7}")
print(" & & Mean ± SD & Median & Range & Avg Kills & Wins & \\\\")
print("\\hline")
for cluster in sorted(cluster_stats.index):
    cluster_data = player_stats[player_stats['Cluster'] == cluster]
    percentiles = cluster_data['game_length_minutes'].quantile([0.5])
    
    print(f"{int(cluster)} & {int(cluster_stats.loc[cluster, 'Number of Players'])} & "
          f"{cluster_stats.loc[cluster, 'Avg Game Length (min)']:.2f} ± {cluster_stats.loc[cluster, 'Std Dev Length (min)']:.2f} & "
          f"{percentiles[0.5]:.2f} & "
          f"{cluster_stats.loc[cluster, 'Min Game Length (min)']:.1f}--{cluster_stats.loc[cluster, 'Max Game Length (min)']:.1f} & "
          f"{cluster_stats.loc[cluster, 'Avg Kills per Player']:.2f} & "
          f"{int(cluster_stats.loc[cluster, 'Total Wins'])} & "
          f"{cluster_stats.loc[cluster, 'Win Rate (%)']:.1f}\\% \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")