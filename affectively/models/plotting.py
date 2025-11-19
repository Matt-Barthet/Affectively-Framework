
def plot_training_curves(train_scores, val_scores, game, cluster, metric_name='Score', baseline=0):
    """Plot validation curves for KNN model"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [score[0] for score in train_scores]
    train_values = [score[1] for score in train_scores]
    val_values = [score[1] for score in val_scores]
    
    ax.plot(k_values, train_values, label=f'Training {metric_name}', alpha=0.8, marker='o')
    ax.plot(k_values, val_values, label=f'Validation {metric_name}', alpha=0.8, marker='o')
    
    if baseline > 0:
        ax.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    
    ax.set_title(f"KNN Performance - {game} (Cluster {cluster})")
    ax.set_xlabel("K (Number of Neighbors)")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(f'affectively/plots/{game}', exist_ok=True)
    plt.show()
