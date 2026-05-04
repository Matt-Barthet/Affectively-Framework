from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from affectively.environments import GymToGymnasiumWrapper
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.utils import compute_confidence_interval
from affectively.utils.logging import TensorBoardCallback
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from agents import load_model
from archive_utils import get_best_cell, process_surrogate_vectors


def discretize_input(val):
    return int(np.ceil(val) if val > 0 else np.floor(val))


def run_evaluation(env, model, model_type, steps_per_episode=600, imitation=False):
    state = env.reset()
    state = env.reset()
    pos_arousal = [[], [], []]

    results = {'score': [], 'arousal': 0, 'behavior': 0, 'arousal_return': 0, 'lap_time': 0,
               'speed': 0, 'off_road': 0, 'gas_pedal': {-1: 0, 0: 0, 1: 0}, 'steering': {-1: 0, 0: 0, 1: 0}, 'distance_to_cars': 0, 'final_position': 0}
    prev_length = 0
    lap_counter = 0
    for steps in range(steps_per_episode):
        action = model.predict(state, deterministic=True)[0] if model_type != "random" else env.action_space.sample()
        state, reward, done, info = env.step(action)

        pos_arousal[0].append(env.customSideChannel.pos)
        pos_arousal[1].append(env.episode_arousal_trace[-1] if len(env.episode_arousal_trace) > 0 else 1)
        pos_arousal[2].append(reward)
        
        if len(env.episode_arousal_trace) != prev_length:
            results['off_road'] += np.ceil(env.current_surrogate[7])
            lap_counter += 1
            if env.current_score == 8 or env.current_score == 16 or env.current_score == 24:
                results['lap_time'] += lap_counter * 3
                lap_counter = 0
            results['gas_pedal'][discretize_input(env.current_surrogate[8])] += 1
            results['steering'][discretize_input(env.current_surrogate[9])] += 1
            results['speed'] += env.current_surrogate[2]
            results['distance_to_cars'] += env.current_surrogate[25]
            results['score'].append(env.current_score)
            prev_length = len(env.episode_arousal_trace)
            # print(f"Off Road: {np.ceil(env.current_surrogate[7])}, Speed:{env.current_surrogate[2]}, Distance:{env.current_surrogate[25]}")

        if done:
            env.reset()

    results['off_road'] += env.current_surrogate[7]
    results['gas_pedal'][discretize_input(env.current_surrogate[8])] += 1
    results['steering'][discretize_input(env.current_surrogate[9])] += 1
    results['speed'] += env.current_surrogate[2]
    results['distance_to_cars'] += env.current_surrogate[25]
    results['score'].append(env.current_score)
    prev_length = len(env.episode_arousal_trace)

    pd.DataFrame(np.asarray(pos_arousal).T, columns=['Position', 'Arousal', 'Score']).to_csv("Pos_Arousal_Score.csv")

    for idx in range(len(env.episode_arousal_trace)-1):
        if env.episode_arousal_trace[idx] == 0 if np.sign(env.model.cluster_arousal[idx+1] - env.model.cluster_arousal[idx]) <= 0 else 1:
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
        results = {'score': [], 'arousal': 0, 'behavior': 0, 'arousal_return': 0, 'lap_time': 0,
               'speed': 0, 'off_road': 0, 'gas_pedal': {-1: 0, 0: 0, 1: 0}, 'steering': {-1: 0, 0: 0, 1: 0}, 'distance_to_cars': 0, 'final_position': 0}

        archive = load_model(model_type, model_path, env, "")
        best_cell = get_best_cell(archive)
        print(f"Best cell: reward={best_cell.reward}, behavior={best_cell.behavior_reward}, arousal={best_cell.arousal_reward}, length={best_cell.get_cell_length()}")

        arousal_trace = best_cell.trajectory_dict['arousal_trajectory']
        _, valid_vectors, counter = process_surrogate_vectors(best_cell)

        lap_counter = 0
        current_lap = 0
        scores = []

        for vector in valid_vectors:
            results['off_road'] += vector[7]
            results['gas_pedal'][discretize_input(vector[8])] += 1
            results['steering'][discretize_input(vector[9])] += 1
            results['speed'] += vector[2]
            results['distance_to_cars'] += vector[25]

            score = np.ceil(vector[1])
            lap_counter += 1
            if current_lap < score // 8 and (score == 8 or score == 16 or score == 24):
                print(f"Score: {score}, Lap Counter: {lap_counter}, Current Lap: {current_lap}")
                results['lap_time'] += lap_counter / 4
                current_lap = score // 8
                lap_counter = 0

            scores.append(score)
            if int(vector[1]) == env.model.cluster_score[-1]:
                results['final_position'] = vector[0]

        for idx in range(len(arousal_trace)-1):
            if arousal_trace[idx] == 0 if np.sign(env.model.cluster_arousal[idx+1] - env.model.cluster_arousal[idx]) <= 0 else 1:
                results['arousal_return'] += 1

        results['final_position'] = best_cell.trajectory_dict['arousal_vectors'][-1][0] if results['final_position'] == 0 else results['final_position']
        results['arousal_return'] /= len(arousal_trace)
        results['score'] = best_cell.score
        results['off_road'] /= counter
        results['speed'] /= counter
        results['behavior'] = best_cell.behavior_reward 

        results['arousal'] = best_cell.arousal_reward
        results['gas_pedal'] = np.asarray(list(results['gas_pedal'].values())) / len(arousal_trace)
        results['steering'] = np.asarray(list(results['steering'].values())) / len(arousal_trace)
        results['lap_time'] /= best_cell.score // 8
        results['distance_to_cars'] /= len(arousal_trace) * 15

        
        pos_arousal = [[], [], []]
        pos_arousal[0] = [list(best_cell.trajectory_dict['raw_state'][idx][0][0:3]) for idx in range(300)]
        pos_arousal[1] = [v for v in arousal_trace for _ in range(15)][:300]
        pos_arousal[2] = scores[:300]
        pd.DataFrame(np.asarray(pos_arousal).T, columns=['Position', 'Arousal', 'Score']).to_csv("Pos_Arousal_Score.csv")

        return results

    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error processing archive: {e}")
        return results


if __name__ == "__main__":

    runs = 10
    results = []

    try:
        df = pd.read_csv('experiment_results.csv')
    except:
        df = pd.DataFrame()
    for model_type in ['Explore']:

        env = SolidEnvironmentGameObs(
            0,
            graphics=False,
            weight=0,
            discretize=model_type=="Explore",
            cluster=0,
            target_arousal=1,
            period_ra=False,
            decision_period=10,
            imitate=True,
            capture_fps=-60
        )
        gymnasium_env = GymToGymnasiumWrapper(env)

        for weight in [0.0, 0.5, 1.0]:

            for cluster in [1,2,3,4]:

                run_scores = []
                run_arousals = []
                run_behaviors= []
                run_arousal_returns = []

                run_final_positions = []
                run_lap_times = []
                run_off_road = []
                run_speeds = []
                run_gas_pedals = []
                run_steerings = []
                run_distance_to_cars = []

                if model_type == "random" and (weight != 0):
                    print("Skipping random runs...")
                    continue
                # Check if this weight, cluster, alg combo are in the results df, if so, skip
                try:
                    if any((df['model'] == model_type) & (df['weight'] == weight) & (df['cluster'] == cluster)):
                        print(f"Skipping: {model_type}, cluster={cluster}, weight={weight} (already evaluated)")
                        continue
                except KeyError:
                    pass

                for run in range(10):

                    if model_type == "Explore":
                        model_name = f"MlpPolicy-Cluster{cluster}-{weight}λ-run{run}"
                        model_path = f"results/solid/Synchronized Reward/Ordinal/Classification/Imitate Arousal/{model_type}/{model_name}"                        
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

                        if model_type == "Explore":
                            run_results = process_archive(model_type, model_path)
                        else:
                            model = load_model(model_type, model_path, env, model_name) if model_type != "random" else None
                            env.callback = TensorBoardCallback("", gymnasium_env, model)
                            run_results = run_evaluation(env, model if model_type != "random" else None, model_type, steps_per_episode=600)
                        
                        if run_results is None: 
                            continue

                        run_results['behavior'] /= env.model.cluster_score[-1]
                        run_results['arousal'] /= env.model.cluster_score[-1]

                        run_behaviors.append(run_results['behavior'])
                        run_scores.append(run_results['score'])
                        run_arousals.append(run_results['arousal'])
                        run_arousal_returns.append(run_results['arousal_return'])       

                        run_final_positions.append(run_results['final_position'])
                        run_lap_times.append(run_results['lap_time'])
                        run_off_road.append(run_results['off_road'])
                        run_speeds.append(run_results['speed'])
                        run_gas_pedals.append(run_results['gas_pedal'])
                        run_steerings.append(run_results['steering'])
                        run_distance_to_cars.append(run_results['distance_to_cars'])
                        
                        print(f"Behavior: {run_results['behavior']:.2f}, Score: {run_results['score']:.2f}, Arousal: {run_results['arousal']:.3f}, Arousal Return: {run_results['arousal_return']:.3f}")
                        print(f"Final Position: {run_results['final_position']:.2f}, Lap Time: {run_results['lap_time']:.2f}, Off-road: {run_results['off_road']:.2f}, Speed: {run_results['speed']:.2f}, Distance to Cars: {run_results['distance_to_cars']:.2f}\n")

                    except Exception as e:
                        print(f"Raised: {e}")
                        raise

                score_mean, score_ci = compute_confidence_interval(run_scores)
                arousal_mean, arousal_ci = compute_confidence_interval(run_arousals)
                behavior_mean, behavior_ci = compute_confidence_interval(run_behaviors)
                arousal_return_mean, arousal_return_ci = compute_confidence_interval(run_arousal_returns)
                final_position_mean, final_position_ci = compute_confidence_interval(run_final_positions)
                lap_time_mean, lap_time_ci = compute_confidence_interval(run_lap_times)
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
                    'lap_time_mean': lap_time_mean,
                    'lap_time_ci': lap_time_ci,
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
                    'final_position_raw': run_final_positions,
                    'lap_time_raw': run_lap_times,
                    'off_road_raw': run_off_road,
                    'distance_to_cars_raw': run_distance_to_cars,
                    "frequency": 'Synchronized',
                    "cluster": cluster,
                    'game': 'Solid'
                })

                print(f"Summary for {model_type}/Cluster {cluster}/λ={weight}:")
                print(f"Behavior: {behavior_mean:.2f} ± {behavior_ci:.2f}, Score: {score_mean:.2f} ± {score_ci:.2f}, Arousal: {arousal_mean:.3f} ± {arousal_ci:.3f}")
                print(f"Final Position: {final_position_mean:.2f} ± {final_position_ci:.2f}, Lap Time: {lap_time_mean:.2f} ± {lap_time_ci:.2f}, Off-road: {off_road_mean:.2f} ± {off_road_ci:.2f}, Speed: {speed_mean:.2f} ± {speed_ci:.2f}, Distance to Cars: {distance_to_cars_mean:.2f} ± {distance_to_cars_ci:.2f}\n")
                
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
