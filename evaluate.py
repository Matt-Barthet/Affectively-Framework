import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
import torch

from affectively.environments.base import compute_confidence_interval
from affectively.environments.gymnasium_wrapper import GymToGymnasiumWrapper
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Explore import Explorer
# Update this import to match your actual module structure
from agents.game_obs.Rainbow_DQN import RainbowAgent


def load_model(model_type, model_path, env, model_name):
    """Load model based on type."""
    if model_type == 'PPO':
        model = PPO(policy=model_name.split("-")[0], env=env)
        model.set_parameters(model_path)
    elif model_type == 'DQN':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RainbowAgent(env, device=device)
        model.load(model_path)
    elif model_type == 'Random':
        return None
    elif model_type == "Explore":
        file = open(model_path, 'rb')
        model = pickle.load(file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def run_evaluation(env, model, model_type, steps_per_episode=600):
    """Run single evaluation trial for a given environment and model."""
    state = env.reset()
    for i in range(steps_per_episode):
        if model_type == 'PPO':
            action, _ = model.predict(state, deterministic=True)
        elif model_type == 'DQN':
            action = model.select_action(state)
        else:  # Random agent
            action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            break

    env.callback.on_episode_end()
    arousal = env.callback.best_mean_ra
    score = env.callback.best_env_score
    return arousal, score


if __name__ == "__main__":

    weights = [0.0]
    runs = 5
    signals = ['Ordinal', ]
    tasks = ['Minimize']
    models = ["Explore"]
    predictions = ['Classification']
    cluster = 3
    results = []

    env = SolidEnvironmentGameObs(
            0,
            graphics=False,
            weight=0,
            discretize=False,
            cluster=cluster,
            target_arousal=1,
            period_ra=False,
            decision_period=10,
        )

    for game in ['solid']:
        for freq in ['Synchronized']:
            for model_type in models:
                for signal in signals:
                    for prediction in predictions:
                        for task in tasks:
                            for weight in weights:

                                for cluster in [0]:
                                    run_scores = []
                                    run_arousals = []
                                    for run in range(runs):

                                        if model_type == "Explore":
                                            model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"

                                            if weight != 0.0:
                                                model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/{task} Arousal/{model_type}/{model_name}.zipTS_16666.zip"
                                            else:
                                                model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/Maximize Arousal/{model_type}/{model_name}.zipTS_16666.zip"

                                            try:
                                                best_score = 0
                                                best_reward = -1
                                                best_arousal = 0
                                                cell_length = 0
                                                archive = load_model(model_type, model_path, env, "")
                                                arousal_trace = []

                                                for cell in archive.values():
                                                    # if cell.cumulative_score > best_score:
                                                    #     best_score = cell.cumulative_score
                                                    if cell.reward > best_reward:
                                                        best_reward = cell.reward
                                                        best_score = cell.behavior_reward
                                                        best_arousal = cell.arousal_reward

                                                        cell_length = len(cell.trajectory_dict['score_trajectory'])
                                                        arousal_trace = cell.trajectory_dict['score_trajectory']

                                                run_scores.append(best_score)
                                                run_arousals.append(best_arousal)
                                                print(cell_length, arousal_trace)

                                                if freq == "Synchronized":

                                                    run_arousals[-1] /= best_score
                                                    if weight == 0 and task == "Minimize":
                                                        run_arousals[-1] = 1 - run_arousals[-1]
                                                else:
                                                    run_arousals[-1] *= 40
                                                    run_scores[-1] *= 24

                                            except:
                                                raise
                                                pass

                                        elif model_type != 'Random':
                                            task_name = f"{task} Arousal"

                                            if model_type == 'PPO':
                                                model_name = f"MlpPolicy-Cluster0-{weight}λ-run{run}"
                                            else:
                                                model_name = f"DQN-Cluster0-{weight}λ-run{run}"

                                            if weight != 0:
                                                model_path = f"results/{game}/{freq} Reward/{signal}/{prediction}/{task_name}/{model_type}/{model_name}-Episode-16000.zip"
                                            else:
                                                model_path = f"results/{game}/Synchronized Reward/Ordinal/Classification/Maximize Arousal/{model_type}/{model_name}.zip"

                                            if not Path(model_path).exists():
                                                model_path = f"results/{game}/{freq} Reward/{signal}/{prediction}/{task_name}/{model_type}/{model_name}.zip"
                                                if not Path(model_path).exists():
                                                    print(f"Skipping: {model_path} (not found)")
                                                    continue

                                            print(f"Evaluating: {model_type}, {signal}, {prediction}, {task}, weight={weight}, run={run}")

                                            try:

                                                target_arousal = 0 if task == 'Minimize' else 1

                                                env.reset()
                                                env.weight = weight
                                                env.target_arousal = target_arousal
                                                env.cluster = cluster
                                                env.period_ra = freq == "Asynchronized"
                                                env.decision_period = 10
                                                env.discretize = False
                                                env.classifier = (prediction == 'Classification')
                                                env.preference = (signal == 'Ordinal')
                                                env.reinit()

                                                # Load model
                                                if model_type != 'Random':
                                                    model = load_model(model_type, model_path, env, model_name)
                                                else:
                                                    model = None  # Random agent

                                                env.callback = TensorBoardCallback("", env, model)
                                                arousal, score = run_evaluation(env, model, model_type, steps_per_episode=601)

                                                run_scores.append(score)
                                                run_arousals.append(arousal)

                                                print(f"  Score: {score:.2f}, Arousal: {arousal:.3f}")


                                            except Exception as e:
                                                print(f"Raised: {e}")
                                                raise
                                                # Clean up
                                                try:
                                                    if hasattr(env, 'env'):
                                                        env.env.close()
                                                    else:
                                                        env.close()
                                                    print("✓ Environment closed")
                                                except Exception as e:
                                                    print(f"Warning during env close: {e}")

                                    # Compute statistics across runs (if we have any successful evaluations)
                                    if len(run_scores) > 0:

                                        if len(run_scores) > 1:
                                            score_mean, score_ci = compute_confidence_interval(run_scores)
                                            arousal_mean, arousal_ci = compute_confidence_interval(run_arousals)
                                        else:
                                            score_mean, score_ci = run_scores[0], 0
                                            arousal_mean, arousal_ci = run_arousals[0], 0

                                        # Store aggregated results
                                        results.append({
                                            'model': model_type,
                                            'signal': signal,
                                            'prediction': prediction,
                                            'task': task,
                                            'weight': weight,
                                            'n_runs': len(run_scores),
                                            'score_mean': score_mean,
                                            'score_ci': score_ci,
                                            'arousal_mean': arousal_mean,
                                            'arousal_ci': arousal_ci,
                                            'scores_raw': run_scores,
                                            'arousal_raw': run_arousals,
                                            "frequency": freq,
                                            "cluster": cluster,
                                            'game': game
                                        })

                                        print(f"Summary for {freq} reward/{model_type}/{signal}/{prediction}/{task}/λ={weight}:")
                                        print(
                                            f"  Score: {score_mean:.2f} ± {score_ci:.2f}, Arousal: {arousal_mean:.3f} ± {arousal_ci:.3f}\n")

    df = pd.DataFrame(results)

    # Save to CSV
    output_file = 'experiment_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total parameter combinations evaluated: {len(results)}")

    if len(results) > 0:
        print("\n=== Summary Statistics ===")
        summary = df.groupby(['model', 'signal', 'prediction', 'task', 'weight']).agg({
            'score_mean': 'mean',
            'arousal_mean': 'mean',
            'n_runs': 'first'
        }).round(3)
        print(summary)