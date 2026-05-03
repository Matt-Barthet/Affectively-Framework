from pathlib import Path    
import os

if __name__ == "__main__":

    for game in ['solid', 'platform', 'fps']:
        for freq in ['Synchronized', 'Asynchronized']:
            for model_type in ['PPO', 'DQN', 'Explore']:
                for weight in [0.0, 0.5, 1.0]:
                    for cluster in [0,1,2,3,4]:
                        for task in ['Maximize', 'Minimize', 'Imitate']:
                            locks = 0
                            for run in range(10):
                                if model_type != "DQN":
                                    model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                else:
                                    model_name = f"DQN-Cluster{cluster}-{weight}λ-run{run}"
                                model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/{task} Arousal/{model_type}/{model_name}.lock"
                                if os.path.exists(model_path):
                                    locks += 1

                            if locks > 0:
                                print(f"Game: {game}, Model: {model_type}, Freq: {freq}, Task: {task}, Weight: {weight}, Cluster: {cluster} - {locks} experiments running!")
