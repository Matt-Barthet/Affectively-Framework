from pathlib import Path
import numpy as np
import pandas as pd
from affectively.environments import GymToGymnasiumWrapper
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.utils import compute_confidence_interval
from affectively.utils.logging import TensorBoardCallback
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from agents import load_model


def run_evaluation(env, model, model_type, steps_per_episode=600, imitation=False):
    state = env.reset()
    state = env.reset()
    pos_arousal = [[], [], []]

    results = {'score': [], 'arousal': 0, 'behavior': 0, 'arousal_return': 0,
               'speed': 0, 'off_road': 0, 'gas_pedal': {-1: 0, 0: 0, 1: 0}, 'steering': {-1: 0, 0: 0, 1: 0}, 'distance_to_cars': 0, 'final_position': 0}

    prev_length = 0
    for steps in range(steps_per_episode):
        action = model.predict(state, deterministic=True)[0] if model_type != "random" else env.action_space.sample()
        state, reward, done, info = env.step(action)

        if env.game == "platform":
            pos_arousal[0].append(env.customSideChannel.pos)
            pos_arousal[1].append(env.episode_arousal_trace[-1] if len(env.episode_arousal_trace) > 0 else 1)
            pos_arousal[2].append(reward)
        
        if len(env.episode_arousal_trace) != prev_length:
            results['off_road'] += np.ceil(env.current_surrogate[7])
            results['gas_pedal'][int(env.current_surrogate[8])] += 1
            results['steering'][int(env.current_surrogate[9])] += 1
            results['speed'] += env.current_surrogate[2]
            results['distance_to_cars'] += env.current_surrogate[25]
            results['score'].append(env.current_score)
            prev_length = len(env.episode_arousal_trace)
            # print(f"Off Road: {np.ceil(env.current_surrogate[7])}, Speed:{env.current_surrogate[2]}, Distance:{env.current_surrogate[25]}")

        if done:
            env.reset()

    results['off_road'] += np.ceil(env.current_surrogate[7])
    results['gas_pedal'][int(env.current_surrogate[8])] += 1
    results['steering'][int(env.current_surrogate[9])] += 1
    results['speed'] += env.current_surrogate[2]
    results['distance_to_cars'] += env.current_surrogate[25]
    results['score'].append(env.current_score)
    prev_length = len(env.episode_arousal_trace)

    for arousal in env.episode_arousal_trace:
        if arousal == env.target_arousal:
            results['arousal_return'] += 1
    
    results['arousal_return'] /= len(env.episode_arousal_trace)
    final_position = env.current_surrogate[0]
    results['off_road'] /= len(env.episode_arousal_trace)
    results['speed'] /= len(env.episode_arousal_trace)
    results['gas_pedal'] = np.asarray(list(results['gas_pedal'].values())) / len(env.episode_arousal_trace)
    results['steering'] = np.asarray(list(results['steering'].values())) / len(env.episode_arousal_trace)
    results['distance_to_cars'] /= len(env.episode_arousal_trace)
    results['final_position'] = final_position
    results['score'] = np.max(results['score'])
    results['arousal'] = env.cumulative_ra
    results['behavior'] = env.cumulative_rb
    return results


def process_archive(model_type, model_path):
    try:
        best_score = 0
        best_behavior = 0
        best_reward = -1
        best_arousal = 0
        cell_length = 0
        best_cell = None
        archive = load_model(model_type, model_path, env, "")
        arousal_trace = []

        for cell in archive.values():
            if cell.reward > best_reward:
                best_cell = cell

        best_reward = best_cell.reward
        best_score = best_cell.score
        best_behavior = best_cell.behavior_reward
        best_arousal = best_cell.arousal_reward
        cell_length = len(best_cell.trajectory_dict['state_trajectory'])
        arousal_trace = best_cell.trajectory_dict['arousal_trajectory']

        if task == "Imitate":
            normalizer = env.model.cluster_score[-1] if cluster > 0 else best_score
        else:
            if game == "platform":
                run_scores.append(best_score / 460)
                run_arousals.append(best_arousal / 40)
            elif game == "solid":
                run_scores.append(best_score / 24)
                run_arousals.append(best_arousal / 24 if freq == "Synchronized" else best_arousal / 40)

        if freq == "Synchronized":
            if weight == 0 and task == "Minimize":
                run_arousals[-1] = 1 - run_arousals[-1]

        print(cell_length, len(arousal_trace), run_arousals[-1], run_scores[-1])

    except FileNotFoundError:
        return None
    except:
        return None


if __name__ == "__main__":

    runs = 10
    results = []
    df = pd.read_csv('experiment_results.csv')

    for model_type in ['DQN', 'PPO', 'Explore']:
        env = SolidEnvironmentGameObs(
            0,
            graphics=True,
            weight=0,
            discretize=model_type=="Explore",
            cluster=0,
            target_arousal=1,
            period_ra=False,
            decision_period=10,
            imitate=True,
            capture_fps=10
        )
        gymnasium_env = GymToGymnasiumWrapper(env)

        for weight in [0.0, 0.5, 1.0]:

            for cluster in [1, 2, 3, 4]:

                run_scores = []
                run_arousals = []
                run_behaviors= []
                run_arousal_returns = []

                run_final_positions = []
                run_off_road = []
                run_speeds = []
                run_gas_pedals = []
                run_steerings = []
                run_distance_to_cars = []

                if model_type == "random" and (weight != 0):
                    print("Skipping random runs...")
                    continue
                # Check if this weight, cluster, alg combo are in the results df, if so, skip
                if any((df['model'] == model_type) & (df['weight'] == weight) & (df['cluster'] == cluster)):
                    print(f"Skipping: {model_type}, cluster={cluster}, weight={weight} (already evaluated)")
                    continue

                for run in range(runs):

                    if model_type == "Explore":
                        model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                        model_path = f"results/solid/Synchronized Reward/Ordinal/Classification/Imitate Arousal/{model_type}/{model_name}.zip"
                        scores, r_a, r_b, metrics = process_archive(model_name, model_path)
                        
                    elif model_type != 'random':
                        if model_type == 'PPO':
                            model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                        else:
                            model_name = f"DQN-Cluster{cluster}-{weight}λ-run{run}"

                        model_path = f"results/solid/Synchronized Reward/Ordinal/Classification/Imitate Arousal/{model_type}/{model_name}.zip"
                        if not Path(model_path).exists():
                            print(f"Skipping: {model_path} (not found)")
                            continue

                    print(f"Evaluating: {model_type}, {cluster}, weight={weight}, run={run}")

                    try:

                        env.weight = weight
                        env.cluster = cluster
                        discretize=model_type=="Explore",
                        env.reinit()
                        model = load_model(model_type, model_path, env, model_name) if model_type != "random" else None
                        env.callback = TensorBoardCallback("", gymnasium_env, model)
                        run_results = run_evaluation(env, model, model_type, steps_per_episode=600)
                        
                        run_results['behavior'] /= env.model.cluster_score[-1]
                        run_results['arousal'] /= env.model.cluster_score[-1]

                        run_behaviors.append(run_results['behavior'])
                        run_scores.append(run_results['score'])
                        run_arousals.append(run_results['arousal'])
                        run_arousal_returns.append(run_results['arousal_return'])       

                        run_final_positions.append(run_results['final_position'])
                        run_off_road.append(run_results['off_road'])
                        run_speeds.append(run_results['speed'])
                        run_gas_pedals.append(run_results['gas_pedal'])
                        run_steerings.append(run_results['steering'])
                        run_distance_to_cars.append(run_results['distance_to_cars'])
                        
                        print(f"Behavior: {run_results['behavior']:.2f}, Score: {run_results['score']:.2f}, Arousal: {run_results['arousal']:.3f}, Arousal Return: {run_results['arousal_return']:.3f}")
                        print(f"Final Position: {run_results['final_position']:.2f}, Off-road: {run_results['off_road']:.2f}, Speed: {run_results['speed']:.2f}, Distance to Cars: {run_results['distance_to_cars']:.2f}\n")

                    except Exception as e:
                        print(f"Raised: {e}")
                        raise

                score_mean, score_ci = compute_confidence_interval(run_scores)
                arousal_mean, arousal_ci = compute_confidence_interval(run_arousals)
                behavior_mean, behavior_ci = compute_confidence_interval(run_behaviors)
                arousal_return_mean, arousal_return_ci = compute_confidence_interval(run_arousal_returns)
                final_position_mean, final_position_ci = compute_confidence_interval(run_final_positions)
                off_road_mean, off_road_ci = compute_confidence_interval(run_off_road)
                speed_mean, speed_ci = compute_confidence_interval(run_speeds)
                distance_to_cars_mean, distance_to_cars_ci = compute_confidence_interval(run_distance_to_cars)
                gas_pedal_mean = np.mean(run_gas_pedals, axis=0)
                steering_mean = np.mean(run_steerings, axis=0)
                gas_pedal_ci = np.array([compute_confidence_interval([run[i] for run in run_gas_pedals])[1] for i in range(3)])
                steering_ci = np.array([compute_confidence_interval([run[i] for run in run_steerings])[1] for i in range(3)])

                # Store aggregated results
                results.append({
                    'model': model_type,
                    'signal': 'Ordinal',
                    'prediction': 'Classification',
                    'task': 'Imitate',
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
                    'final_position_mean': final_position_mean,
                    'final_position_ci': final_position_ci,
                    'off_road_mean': off_road_mean,
                    'off_road_ci': off_road_ci,
                    'speed_mean': speed_mean,
                    'speed_ci': speed_ci,
                    'distance_to_cars_mean': distance_to_cars_mean,
                    'distance_to_cars_ci': distance_to_cars_ci,
                    'gas_pedal_mean': str(gas_pedal_mean),
                    'gas_pedal_ci': str(gas_pedal_ci),
                    'steering_mean': str(steering_mean),
                    'steering_ci': str(steering_ci),
                    'scores_raw': run_scores,
                    'arousal_raw': run_arousals,
                    "frequency": 'Synchronized',
                    "cluster": cluster,
                    'game': 'Solid'
                })

                print(f"Summary for {model_type}/Cluster {cluster}/λ={weight}:")
                print(f"Behavior: {behavior_mean:.2f} ± {behavior_ci:.2f}, Score: {score_mean:.2f} ± {score_ci:.2f}, Arousal: {arousal_mean:.3f} ± {arousal_ci:.3f}")
                print(f"Final Position: {final_position_mean:.2f} ± {final_position_ci:.2f}, Off-road: {off_road_mean:.2f} ± {off_road_ci:.2f}, Speed: {speed_mean:.2f} ± {speed_ci:.2f}, Distance to Cars: {distance_to_cars_mean:.2f} ± {distance_to_cars_ci:.2f}\n")
                
                df = pd.concat([df, pd.DataFrame(results)])
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

        env.env.close()
        env.close()
