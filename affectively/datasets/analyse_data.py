import pandas as pd
import numpy as np

# Read the datasets
cluster_book = pd.read_csv('fps_cluster_book.csv')
fps_data = pd.read_csv('fps_3000ms.csv')

# Merge the datasets on player_id
merged_data = fps_data.merge(cluster_book, on='[control]player_id', how='inner')

merged_data['playerDeath'] = np.ceil(merged_data['playerDeath'])

# Group by player_id and cluster to get per-player statistics
player_stats = merged_data.groupby(['[control]player_id', 'Cluster']).agg({
    'playerScore': 'max',  # Final score (last value)
    'playerKillCount': 'max',  # Final kill count
    'playerDeath': 'sum',  # Total deaths
    'playerSprinting': 'mean',  # Average sprinting (0 or 1)
    'playerShooting': 'mean'  # Average shooting (0 or 1)
}).reset_index()

# Calculate K/D ratio
player_stats['kd_ratio'] = player_stats['playerScore'] / player_stats['playerDeath'].replace(0, 1)
# Convert percentages
player_stats['sprinting_percent'] = player_stats['playerSprinting'] * 100
player_stats['shooting_percent'] = player_stats['playerShooting'] * 100

# Group by cluster and calculate statistics
cluster_stats = player_stats.groupby('Cluster').agg({
    'playerScore': 'mean',
    'kd_ratio': 'mean',
    'playerDeath': 'mean',
    'sprinting_percent': 'mean',
    'shooting_percent': 'mean'
}).round(2)

# Rename columns for clarity
cluster_stats.columns = [
    'Mean Final Score',
    'Mean K/D Ratio',
    'Mean Deaths',
    'Time Sprinting (%)',
    'Time Shooting (%)'
]

# Print the results
print("FPS Player Statistics by Cluster")
print("=" * 80)
print(cluster_stats)
print("\n")

# Print additional summary statistics
print("Summary Statistics")
print("-" * 40)
for cluster in sorted(cluster_stats.index):
    print(f"\nCluster {int(cluster)}:")
    print(f"  Players in cluster: {len(player_stats[player_stats['Cluster'] == cluster])}")
    print(f"  Wins: {(player_stats[player_stats['Cluster'] == cluster]['playerScore'] >= 500).sum()}") # Count winners
    print(f"  Avg Final Score: {cluster_stats.loc[cluster, 'Mean Final Score']:.2f}")
    print(f"  Avg K/D Ratio: {cluster_stats.loc[cluster, 'Mean K/D Ratio']:.2f}")
    print(f"  Avg Deaths: {cluster_stats.loc[cluster, 'Mean Deaths']:.2f}")
    print(f"  Time Sprinting: {cluster_stats.loc[cluster, 'Time Sprinting (%)']:.1f}%")
    print(f"  Time Shooting: {cluster_stats.loc[cluster, 'Time Shooting (%)']:.1f}%")
    print(f"  Player Kill Count: {cluster_stats.loc[cluster, 'Mean Final Score']/20:.2f}")

# Optional: Save results to CSV
cluster_stats.to_csv('cluster_statistics.csv')
print("\nResults saved to 'cluster_statistics.csv'")