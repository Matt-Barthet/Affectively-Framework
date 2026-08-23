from pathlib import Path

def check_missing_experiments():
    results = []

    for game in ['solid', 'platform', 'fps']:
        for freq in ['Synchronized', 'Asynchronized']:
            for signal in ['Ordinal']:
                for prediction in ['Classification']:
                    for model_type in ['PPO', 'DQN', 'Explore']:
                        for weight in [0.0, 0.5, 1.0]:
                            clusters = [0] if game != 'solid' or weight != 0.0 else [0]  # adjust if needed, but for now
                            for cluster in clusters:
                                if weight == 0.0:
                                    missing_max = 0
                                    missing_min = 0
                                    for task in ['Maximize', 'Minimize']:
                                        for run in range(10):
                                            if model_type == "Explore":
                                                model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                                model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/{task} Arousal/{model_type}/{model_name}.zip"
                                            else:
                                                task_name = f"{task} Arousal"
                                                if model_type == 'PPO':
                                                    model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                                else:
                                                    model_name = f"DQN-Cluster{cluster}-{weight}λ-run{run}"
                                                model_path = f"results/{game}/{freq} Reward/{signal}/{prediction}/{task_name}/{model_type}/{model_name}.zip"
                                            if not Path(model_path).exists():
                                                if task == 'Maximize':
                                                    missing_max += 1
                                                else:
                                                    missing_min += 1
                                    total_missing = 10 - (10 - missing_max + 10 - missing_min) 
                                    results.append({
                                        'game': game,
                                        'model_type': model_type,
                                        'task': 'Maximize/Minimize',
                                        'freq': freq,
                                        'signal': signal,
                                        'prediction': prediction,
                                        'weight': weight,
                                        'cluster': cluster,
                                        'missing_runs': total_missing,
                                        'total_runs': 10
                                    })
                                else:
                                    for task in ['Maximize', 'Minimize']:
                                        missing_count = 0
                                        for run in range(10):
                                            if model_type == "Explore":
                                                model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                                model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/{task} Arousal/{model_type}/{model_name}.zip"
                                            else:
                                                task_name = f"{task} Arousal"
                                                if model_type == 'PPO':
                                                    model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                                else:
                                                    model_name = f"DQN-Cluster{cluster}-{weight}λ-run{run}"
                                                model_path = f"results/{game}/{freq} Reward/{signal}/{prediction}/{task_name}/{model_type}/{model_name}.zip"
                                            if not Path(model_path).exists():
                                                missing_count += 1
                                        results.append({
                                            'game': game,
                                            'model_type': model_type,
                                            'task': task,
                                            'freq': freq,
                                            'signal': signal,
                                            'prediction': prediction,
                                            'weight': weight,
                                            'cluster': cluster,
                                            'missing_runs': missing_count,
                                            'total_runs': 10
                                        })

                        # Handle Imitate separately for solid
                        if game == 'solid' and not freq == 'Asynchronized' :
                            for task in ['Imitate']:
                                for weight in [0.0, 0.5, 1.0]:
                                    clusters = [1, 2, 3, 4]
                                    for cluster in clusters:
                                        missing_count = 0
                                        for run in range(10):
                                            if model_type == "Explore":
                                                model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                                model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/{task} Arousal/{model_type}/{model_name}.zip"
                                            else:
                                                task_name = f"{task} Arousal"
                                                if model_type == 'PPO':
                                                    model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                                else:
                                                    model_name = f"DQN-Cluster{cluster}-{weight}λ-run{run}"
                                                model_path = f"results/{game}/{freq} Reward/{signal}/{prediction}/{task_name}/{model_type}/{model_name}.zip"
                                            if not Path(model_path).exists():
                                                missing_count += 1
                                        results.append({
                                            'game': game,
                                            'model_type': model_type,
                                            'task': task,
                                            'freq': freq,
                                            'signal': signal,
                                            'prediction': prediction,
                                            'weight': weight,
                                            'cluster': cluster,
                                            'missing_runs': missing_count,
                                            'total_runs': 10
                                        })

    return results


if __name__ == "__main__":
    all_results = check_missing_experiments()
    missing_results = sorted([r for r in all_results if r['missing_runs'] > 0], key=lambda r: r['missing_runs'], reverse=True)

    total_configs = len(all_results)
    total_runs = sum(r['total_runs'] for r in all_results)
    total_missing_runs = sum(r['missing_runs'] for r in all_results)

    if not missing_results:
        print("No missing experiments found.")
    else:
        print("Missing experiments summary (sorted by missing runs):")
        for result in missing_results:
            print(f"Game: {result['game']}, Model: {result['model_type']}, Freq: {result['freq']}, Task: {result['task']}, Weight: {result['weight']}, Cluster: {result['cluster']} - Missing {result['missing_runs']} out of {result['total_runs']} runs")

    print(f"\nTotal configurations checked: {total_configs}")
    print(f"Total runs checked: {total_runs}")
    print(f"Total missing runs: {total_missing_runs}")