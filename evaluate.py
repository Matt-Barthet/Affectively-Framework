import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
import torch

from affectively.environments.base import compute_confidence_interval
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
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

        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()

    arousal = env.callback.best_mean_ra
    score = env.callback.best_env_score

    return arousal, score


if __name__ == "__main__":
    # Parameter combinations
    weights = [0.0, 0.5, 1.0]
    runs = 5
    signals = ['Raw', 'Ordinal']
    tasks = ['Maximize', 'Minimize']
    models = ['PPO', 'DQN', ]
    predictions = ['Classification', 'Regression']

    # Results storage
    results = []

    env = SolidEnvironmentGameObs(
            0,
            graphics=False,
            weight=0,
            discretize=False,
            cluster=0,
            target_arousal=1,
            period_ra=False,
            decision_period=10,
        )

    # Iterate over all parameter combinations
    for model_type in models:
        for signal in signals:
            for prediction in predictions:
                for task in tasks:
                    for weight in weights:

                        run_scores = []
                        run_arousals = []

                        for run in range(runs):
                            # Construct model path
                            task_name = f"{task} Arousal"

                            if model_type == 'PPO':
                                model_name = f"MlpPolicy-Cluster0-{weight}λ-run{run}"
                            else:  # DQN
                                model_name = f"DQN-Cluster0-{weight}λ-run{run}"

                            model_path = f"results/solid/{signal}/{prediction}/{task_name}/{model_type}/{model_name}.zip"

                            if not Path(model_path).exists():
                                print(f"Skipping: {model_path} (not found)")
                                continue

                            print(f"Evaluating: {model_type}, {signal}, {prediction}, {task}, weight={weight}, run={run}")

                            try:

                                target_arousal = 0 if task == 'Minimize' else 1

                                env.reset()
                                env.weight = weight
                                env.target_arousal = target_arousal
                                env.cluster = 0
                                env.period_ra = False
                                env.decision_period = 10
                                env.discretize = False
                                env.classifier = (prediction == 'Classification')
                                env.preference = (signal == 'Ordinal')
                                env.reinit()

                                # Load model
                                model = load_model(model_type, model_path, env, model_name)
                                env.callback = TensorBoardCallback("", env, model)

                                # Run single evaluation trial
                                arousal, score = run_evaluation(env, model, model_type, steps_per_episode=601)

                                run_scores.append(score)
                                run_arousals.append(arousal)

                                print(f"  Score: {score:.2f}, Arousal: {arousal:.3f}")

                                env.close()

                            except Exception as e:
                                print(f"  Error during evaluation: {str(e)}")
                                raise
                                continue

                        # Compute statistics across runs (if we have any successful evaluations)
                        if len(run_scores) > 0:
                            score_mean, score_ci = compute_confidence_interval(run_scores)
                            arousal_mean, arousal_ci = compute_confidence_interval(run_arousals)

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
                                'arousal_raw': run_arousals
                            })

                            print(f"Summary for {model_type}/{signal}/{prediction}/{task}/λ={weight}:")
                            print(
                                f"  Score: {score_mean:.2f} ± {score_ci:.2f}, Arousal: {arousal_mean:.3f} ± {arousal_ci:.3f}\n")

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_file = 'evaluation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total parameter combinations evaluated: {len(results)}")

    # Display summary statistics
    if len(results) > 0:
        print("\n=== Summary Statistics ===")
        summary = df.groupby(['model', 'signal', 'prediction', 'task', 'weight']).agg({
            'score_mean': 'mean',
            'arousal_mean': 'mean',
            'n_runs': 'first'
        }).round(3)
        print(summary)