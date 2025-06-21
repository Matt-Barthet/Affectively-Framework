import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLResultsAnalyzer:
    def __init__(self, data_directory='.'):
        """Initialize the analyzer with data directory path."""
        self.data_dir = Path(data_directory)
        self.models = ['knn', 'linear', 'mlp', 'rf', 'svm']
        self.data = None
        
    def load_data(self):
        """Load all CSV files and combine them into a single DataFrame."""
        all_data = []
        
        for model in self.models:
            file_path = self.data_dir / f"{model}_training_results.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['model'] = model.lower()
                all_data.append(df)
            else:
                print(f"Warning: {file_path} not found")
        
        if not all_data:
            raise FileNotFoundError("No CSV files found in the specified directory")
            
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Convert boolean columns
        self.data['classifier'] = self.data['classifier'].astype(str) == 'True'
        self.data['preference'] = self.data['preference'].astype(str) == 'True'
        
        # Create meaningful labels
        self.data['task_type'] = self.data['classifier'].map({True: 'Classification', False: 'Regression'})
        self.data['approach'] = self.data['preference'].map({True: 'Ordinal', False: 'Raw'})
        self.data['type_combination'] = self.data['task_type'] + ' + ' + self.data['approach']
        
        print(f"Loaded {len(self.data)} records from {len(all_data)} models")
        print(f"Games: {sorted(self.data['game'].unique())}")
        print(f"Task combinations: {sorted(self.data['type_combination'].unique())}")
        
    def create_comparison_plots(self):
        """Create separate figures for each task combination."""
        if self.data is None:
            self.load_data()
            
        # Define the four combinations
        combinations = [
            ('Classification', 'Ordinal'),
            ('Classification', 'Raw'), 
            ('Regression', 'Ordinal'),
            ('Regression', 'Raw')
        ]
        
        # Create separate figures for each combination
        for task, approach in combinations:
            self._create_dedicated_figures(task, approach)
        
    def _create_dedicated_figures(self, task_type, approach):
        """Create dedicated figures for a specific task type and approach combination."""
        # Filter data for this combination
        mask = (self.data['task_type'] == task_type) & (self.data['approach'] == approach)
        subset = self.data[mask].copy()
        
        if subset.empty:
            print(f'No data for {task_type} + {approach}')
            return
            
        # Get metric name for this combination
        metric = subset['metric'].iloc[0]
        combo_name = f"{task_type}_{approach}"
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # fig.suptitle(f'{task_type} + {approach} - Performance Analysis', 
                    #  fontsize=16, fontweight='bold')
        
        # Left plot: Bar chart with standard deviation error bars
        self._plot_bar_chart_with_std(fig, ax1, subset, metric)
        
        # Right plot: Improvement box plot
        self._plot_improvement_boxplot(ax2, subset, metric)
        
        fig.legend(
            bbox_to_anchor=(0.5, 1),  # Centered above the plot
            loc='upper center',
            ncol=7,            # One column per model
            frameon=True,
            edgecolor='black',
            facecolor='white',
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.83)  # Lower this value to reduce top margin
        plt.savefig(f'ml_{combo_name}_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()
    
    def _plot_bar_chart_with_std(self, fig, ax, data, metric):
        """Create bar chart with standard deviation error bars and black borders."""
        # Calculate statistics by model and game
        stats = data.groupby(['model', 'game']).agg({
            'score': ['mean', 'std', 'count'],  # std = standard deviation
            'baseline': 'mean'
        }).reset_index()
        
        # Flatten column names
        stats.columns = ['model', 'game', 'score_mean', 'score_std', 'score_count', 'baseline_mean']
        
        # Fill NaN std values with 0 (for single data points)
        stats['score_std'] = stats['score_std'].fillna(0)
        
        # Group by model for plotting
        models = sorted(data['model'].unique())
        games = sorted(data['game'].unique())
        
        # Create grouped bar chart
        x = np.arange(len(games))
        width = 0.18
        
        # Create consistent color mapping for models
        color_map = dict(zip(models, plt.cm.viridis(np.linspace(0, 1, len(models)))))
        
        for i, model in enumerate(models):
            model_stats = stats[stats['model'] == model]
            
            # Create mapping from game to stats
            game_means = []
            game_stds = []
            game_baselines = []
            
            for game in games:
                game_data = model_stats[model_stats['game'] == game]
                if not game_data.empty:
                    game_means.append(game_data['score_mean'].iloc[0])
                    game_stds.append(game_data['score_std'].iloc[0])
                    game_baselines.append(game_data['baseline_mean'].iloc[0])
                else:
                    game_means.append(0)
                    game_stds.append(0)
                    game_baselines.append(0)
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, game_means, width, 
                         label=model, color=color_map[model], alpha=0.8,
                         edgecolor='black', linewidth=1.5)
            ax.errorbar(x + offset, game_means, yerr=game_stds, 
                       fmt='none', color='black', capsize=3, capthick=1.5)            
            ax.scatter(x + offset, game_baselines, 
                      color='red', marker='_', s=150, alpha=0.8, linewidth=3, label="Baseline" if i == 0 else "")
        
        # Formatting
        ax.set_xlabel('Game')
        ax.set_ylabel(f'{metric}')
        ax.set_title('Model Performance', pad=10)
        ax.set_xticks(x)
        ax.set_facecolor('white')
                # Add black border to all sides of the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        ax.grid(False)
        games = [game if game != "solid" else "racing" for game in games]
        ax.set_xticklabels(games)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to show all data clearly
        all_scores = data['score'].tolist() + data['baseline'].tolist()
        y_margin = (max(all_scores) - min(all_scores)) * 0.1
        ax.set_ylim(min(all_scores) - y_margin, max(all_scores) + y_margin)
    
    def _plot_improvement_boxplot(self, ax, data, metric):
        """Create improvement box plot."""
        data['improvement'] = data['score'] - data['baseline']
        
        # Get sorted models to ensure consistent ordering
        models = sorted(data['model'].unique())
        
        # Create consistent color mapping for models (same as bar chart)
        color_map = dict(zip(models, plt.cm.viridis(np.linspace(0, 1, len(models)))))
        palette = [color_map[model] for model in models]
        
        # Create box plot with explicit order
        sns.boxplot(data=data, x='model', y='improvement', ax=ax, 
                   palette=palette, order=models)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        # Formatting
        ax.set_xlabel('Model')
        ax.set_ylabel(f'$\Delta$ {metric}')
        ax.set_title('Improvement over Baseline', pad=10)
        ax.tick_params(axis='x')
        # ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
                # Add black border to all sides of the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        ax.grid(False)
        # Add mean improvement annotations
        for i, model in enumerate(sorted(data['model'].unique())):
            model_data = data[data['model'] == model]
            mean_improvement = model_data['improvement'].mean()
            ax.text(i, model_data['improvement'].mean(), f'μ={mean_improvement:.3f}', 
                   ha='center', fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    
    def create_improvement_analysis(self):
        """Analyze improvement over baseline for each combination (now integrated into dedicated figures)."""
        if self.data is None:
            self.load_data()
        
        print("Improvement analysis is now integrated into the dedicated figures.")
        print("Each task combination figure includes both bar chart and improvement boxplot.")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting ML Results Analysis...")
        print("="*50)
        
        try:
            self.load_data()
            self.create_comparison_plots()
            self.create_improvement_analysis()
            
            print("\n" + "="*50)
            print("Analysis completed successfully!")
            print("Generated files:")
            print("- ml_classification_ordinal_analysis.png")
            print("- ml_classification_raw_analysis.png")
            print("- ml_regression_ordinal_analysis.png")
            print("- ml_regression_raw_analysis.png")
            print("- ml_performance_heatmap.png")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

import matplotlib
# Usage example
if __name__ == "__main__":

    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["axes.labelsize"] = 16
    matplotlib.rcParams["xtick.labelsize"] = 16
    matplotlib.rcParams["ytick.labelsize"] = 16
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["legend.fontsize"] = 16
    # Initialize analyzer
    analyzer = MLResultsAnalyzer('./affectively/models/results/')  # Current directory
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Or run individual analyses:
    # analyzer.load_data()
    # analyzer.create_comparison_plots()
    # analyzer.create_heatmap_comparison()
    # analyzer.create_improvement_analysis()
    # analyzer.create_summary_statistics()