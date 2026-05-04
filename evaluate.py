from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from affectively.environments import GymToGymnasiumWrapper
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.models.linear_model import LinearSurrogateModel
from affectively.utils import compute_confidence_interval
from affectively.utils.logging import TensorBoardCallback
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from agents import load_model
from archive_utils import get_best_cell, process_surrogate_vectors


SURROGATE_LABELS = {
    'solid': [
        'playerStanding', 'playerScore', 'playerSpeed', 'playerIsGrounded', 'playerIsMidAir',
        'playerIsLooping', 'playerIsCrashing', 'playerIsOffRoad', 'playerGasPedal', 'playerSteering',
        'playerLap', 'playerDistanceToWayPoint', 'playerRespawn', 'visibleBotCount', 'botStanding',
        'botScore', 'botSpeed', 'botIsGrounded', 'botIsLooping', 'botIsOffRoad', 'botIsCrashing',
        'botGasPedal', 'botSteering', 'botLap', 'botDistanceToWayPoint', 'botPlayerDistance',
        'visibleLoopCount',
    ],
    'fps': [
        'playerScore', 'playerKillCount', 'playerSpeedX', 'playerSpeedY', 'playerSpeedZ',
        'playerHealth', 'playerHealing', 'playerDamaged', 'playerShooting', 'playerReloading',
        'playerProjectileCount', 'playerProjectileDistance', 'playerCrouching', 'playerSprinting',
        'playerAimAtEnemy', 'visibleBotCount', 'botSpeedX', 'botSpeedY', 'botSpeedZ', 'botHealth',
        'botDamaged', 'botShooting', 'botProjectileCount', 'botProjectilePlayerDistance',
        'botAimAtPlayer', 'playerDeath', 'playerTriesShootOnReload',
    ],
    'platform': [
        'playerScore', 'playerHasCollisions', 'playerIsCollidingAbove', 'playerIsCollidingBelow',
        'playerIsCollidingLeft', 'playerIsCollidingRight', 'playerIsFalling', 'playerIsGrounded',
        'playerIsJumping', 'playerSpeedX', 'playerSpeedY', 'playerHealth', 'playerDamaged',
        'playerPointPickup', 'playerPowerPickup', 'playerHasPowerup', 'playerKillCount',
        'visibleBotCount', 'botHasCollisions', 'botIsCollidingBelow', 'botIsCollidingLeft',
        'botIsCollidingRight', 'botIsFalling', 'botIsGrounded', 'botSpeedX', 'botSpeedY',
        'botHealth', 'botPlayerDistance', 'pickUpsVisible', 'pickUpPlayerDisctance', 'playerDeath',
    ],
}


def run_evaluation(env, model, model_type, steps_per_episode=600, imitation=False):
    state = env.reset()
    max_score = 24 if env.game.lower() == "solid" else 460 if env.game.lower() == "platform" else 500
    print(max_score)
    results = {'score': 0, 'arousal': 0, 'behavior': 0, 'arousal_return': 0, 'reward': 0,
               'speed': 0, 'off_road': 0, 'gas_pedal': {-1: 0, 0: 0, 1: 0}, 'steering': {-1: 0, 0: 0, 1: 0}, 'distance_to_cars': 0, 'final_position': 0}
    surrogate_vectors = []
    prev_length = 0

    for _ in range(steps_per_episode):
        action = model.predict(state, deterministic=True)[0] if model_type != "random" else env.action_space.sample()
        state, _, done, _ = env.step(action)
        if len(env.episode_arousal_trace) != prev_length:
            if hasattr(env, 'current_surrogate') and env.current_surrogate is not None:
                surrogate_vectors.append(np.array(env.current_surrogate))
            prev_length = len(env.episode_arousal_trace)

        if int(env.current_score) == max_score:
            break

        if done:
            env.reset()

    results['surrogate'] = np.mean(surrogate_vectors, axis=0)

    for arousal in env.episode_arousal_trace:
        if arousal == env.target_arousal:
            results['arousal_return'] += 1
    
    results['arousal_return'] /= len(env.episode_arousal_trace)
    results['score'] = env.current_score
    results['arousal'] = env.cumulative_ra
    results['behavior'] = env.cumulative_rb
    results['reward'] = env.cumulative_rl
    return results


def process_archive(model_type, model_path, target_arousal, model):
    try:
        archive = load_model(model_type, model_path, target_arousal, "")
        best_cell = get_best_cell(archive)
        arousal_trace = best_cell.trajectory_dict['arousal_trajectory']
        arousal_return = sum(1 for a in arousal_trace if a == target_arousal) / len(arousal_trace)
        surrogate_mean, _, _ = process_surrogate_vectors(best_cell, model)

        return {
            'score': best_cell.score,
            'arousal': best_cell.arousal_reward,
            'behavior': best_cell.behavior_reward,
            'arousal_return': arousal_return,
            'surrogate': surrogate_mean,
        }

    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error processing archive: {e}")
        return {
            'score': best_cell.score,
            'arousal': best_cell.arousal_reward,
            'behavior': best_cell.behavior_reward,
            'arousal_return': arousal_return,
            'surrogate': [],
        }


if __name__ == "__main__":

    runs = 10
    results = []

    try:
        df = pd.read_csv('experiment_results.csv')
    except:
        df = pd.DataFrame()

    for game in ['solid', 'platform', 'fps']:
        for model_type in ['random', 'PPO', 'DQN', 'Explore']:
            for task in ['Maximize', 'Minimize']:                
                for freq in ['Synchronized', 'Asynchronized']:
                    signal, prediction = 'Ordinal', 'Classification'

                    for weight in [0.0, 0.5, 1.0]:
                        cluster = 0
                        run_scores = []
                        run_arousals = []
                        run_behaviors= []
                        run_arousal_returns = []
                        run_surrogates = []

                        if model_type == "random" and (task != "Maximize" or weight != 0 or cluster != 0 or freq != "Synchronized"):
                            print("Skipping random runs...")
                            break

                        if task == "Minimize" and weight == 0:
                            print("Already evaluated 0.0 on maximum setting (target doesn't matter in this experiment)...")
                            continue

                        try:
                            if any((df['model'] == model_type) & (df['weight'] == weight) & (df['cluster'] == cluster) & (df['task'] == task) & (df['frequency'] == freq) & (df['game'] == game)):
                                print(f"Skipping: {model_type}, {game}, {task}, {freq}, weight={weight}, cluster={cluster} (already evaluated)")
                                continue
                        except KeyError:
                            pass

                        if model_type != "Explore":
                            if game == 'solid':
                                env = SolidEnvironmentGameObs(
                                    0,
                                    graphics=True,
                                    weight=0,
                                    discretize=False,
                                    cluster=0,
                                    target_arousal=1,
                                    period_ra=False,
                                    decision_period=10,
                                    imitate=False,
                                )
                            elif game == 'platform':
                                env = PiratesEnvironmentGameObs(
                                    0,
                                    graphics=True,
                                    weight=0,
                                    discretize=False,
                                    cluster=0,
                                    target_arousal=1,
                                    period_ra=False,
                                    decision_period=10,
                                    reloadEvery=100,
                                    capture_fps=60
                                )
                            elif game == 'fps':
                                env = HeistEnvironmentGameObs(
                                    0,
                                    graphics=False,
                                    weight=0,
                                    discretize=False,
                                    cluster=0,
                                    target_arousal=1,
                                    period_ra=False,
                                    decision_period=10,
                                )
                            gymnasium_env = GymToGymnasiumWrapper(env)

                        for run in range(runs):
                            print(f"Evaluating: {model_type}, {signal}, {prediction}, {task}, weight={weight}, run={run}")

                            try:
                                if model_type == "Explore":
                                    model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                                    model_path = f"results/{game}/{freq} Reward/Ordinal/Classification/{task} Arousal/{model_type}/{model_name}.zip"
                                    model = LinearSurrogateModel(game, cluster, classifier=prediction=="Classification", preference=signal == "Ordinal")
                                    run_results = process_archive(model_type, model_path, 1 if task == "Maximize" else 0, model)

                                    if run_results is None:
                                        continue
                                else:
                                    model = None
                                    if model_type != 'random':
                                        task_name = f"{task} Arousal"
                                        model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}" if model_type == "PPO" else f"DQN-Cluster{cluster}-{weight}λ-run{run}"
                                        model_path = f"results/{game}/{freq} Reward/{signal}/{prediction}/{task_name}/{model_type}/{model_name}.zip"
                                        if not Path(model_path).exists():
                                            print(f"Skipping: {model_path} (not found)")
                                            continue

                                        target_arousal = 0 if task == 'Minimize' else 1
                                        env.weight = weight
                                        env.target_arousal = target_arousal
                                        env.cluster = cluster
                                        env.period_ra = freq == "Asynchronized"
                                        env.decision_period = 10
                                        env.discretize = False
                                        env.classifier = (prediction == 'Classification')
                                        env.preference = (signal == 'Ordinal')
                                        env.imitation_learning = False
                                        env.reinit()
                                        model = load_model(model_type, model_path, env, model_name) if model_type != "random" else None
                                        env.callback = TensorBoardCallback("", gymnasium_env, model)

                                    if game == "solid":
                                        best_reward = None
                                        best_results = None
                                        for correction in [False, True]:
                                            env.correct_step_bug = correction
                                            candidate = run_evaluation(env, model, model_type, steps_per_episode=600)
                                            if best_reward is None or candidate['reward'] > best_reward:
                                                best_reward = candidate['reward']
                                                best_results = candidate
                                        run_results = best_results
                                    else:
                                        run_results = run_evaluation(env, model, model_type, steps_per_episode=600)

                                if game == "platform":
                                    run_results['arousal'] /= 40
                                    run_results['score'] /= 460
                                elif game == "solid":
                                    run_results['arousal'] = run_results['arousal'] / 24 if freq == "Synchronized" else run_results['arousal'] / 40
                                    run_results['score'] /= 24
                                elif game == "fps":
                                    run_results['arousal'] = run_results['arousal'] / 25 if freq == "Synchronized" else run_results['arousal'] / 40
                                    run_results['score'] /= 500

                                run_behaviors.append(run_results['behavior'])
                                run_scores.append(run_results['score'])
                                run_arousals.append(run_results['arousal'])
                                run_arousal_returns.append(run_results['arousal_return'])
                                if len(run_results['surrogate']) > 0:
                                    run_surrogates.append(run_results['surrogate'])

                                print(np.mean(run_surrogates, axis=0))
                                print(f"Behavior: {run_results['behavior']:.2f}, Score: {run_results['score']:.2f}, Arousal: {run_results['arousal']:.3f}, Arousal Return: {run_results['arousal_return']:.3f}")

                            except Exception as e:
                                print(f"Raised: {e}")
                                raise

                        if len(run_scores) > 0:
                            score_mean, score_ci = compute_confidence_interval(run_scores)
                            arousal_mean, arousal_ci = compute_confidence_interval(run_arousals)
                            behavior_mean, behavior_ci = compute_confidence_interval(run_behaviors)
                            arousal_return_mean, arousal_return_ci = compute_confidence_interval(run_arousal_returns)

                            n_elements = len(run_surrogates[0])
                            surrogate_mean = np.mean(run_surrogates, axis=0)
                            surrogate_ci = np.array([compute_confidence_interval([r[i] for r in run_surrogates])[1] for i in range(n_elements)])

                            print(len(SURROGATE_LABELS[game]), len(surrogate_mean))                            
                            results.append({
                                'model': model_type,
                                'signal': signal,
                                'prediction': prediction,
                                'task': task,
                                'weight': weight,
                                'n_runs': len(run_scores),
                                'score_mean': score_mean,
                                'score_ci': score_ci,
                                'behavior_mean': behavior_mean,
                                'behavior_ci': behavior_ci,
                                'arousal_mean': arousal_mean,
                                'arousal_ci': arousal_ci,
                                'arousal_return_mean': arousal_return_mean,
                                'arousal_return_ci': arousal_return_ci,
                                **{f'{SURROGATE_LABELS[game][i]}_mean': surrogate_mean[i] for i in range(len(surrogate_mean))},
                                **{f'{SURROGATE_LABELS[game][i]}_ci': surrogate_ci[i] for i in range(len(surrogate_ci))},
                                'scores_raw': run_scores,
                                'arousal_raw': run_arousals,
                                "frequency": freq,
                                "cluster": cluster,
                                'game': game
                            })

                            print(f"Summary for {freq} reward/{model_type}/{signal}/{prediction}/{task}/λ={weight}:")
                            print(
                                f"Behavior: {behavior_mean:.2f} ± {behavior_ci:.2f}, Score: {score_mean:.2f} ± {score_ci:.2f}, Arousal: {arousal_mean:.3f} ± {arousal_ci:.3f}")       

                        if model_type != "Explore":
                            env.env.close()
                            env.close()

                        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
                        results = []
                        output_file = 'experiment_results.csv'
                        df.to_csv(output_file, index=False)
                        print(f"\nResults saved to {output_file}")
                        print(f"Total rows in CSV: {len(df)}")

                        if len(results) > 0:
                            print("\n=== Summary Statistics ===")
                            summary = df.groupby(['model', 'signal', 'prediction', 'task', 'weight']).agg({
                                'score_mean': 'mean',
                                'arousal_mean': 'mean',
                                'n_runs': 'first'
                            }).round(3)
                            print(summary)