import numpy as np
from stable_baselines3 import PPO

from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs

import pickle

if __name__ == "__main__":

    run = 0
    target_arousal = 1 # 1 to maximize arousal, 0 to minimize
    comulative_reward = False
    
    label = 'Arousal' # 'Optimize', 'Blended', 'Arousal'
    cluster = 0  # Cluster index for Arousal Persona
    
    num_runs = 500  # Number of runs to simulate
    steps = 600  # Number of steps per run

    env_name = "heist" # "pirates", "solid" or "heist"
    
    agentType = "PPO"  # Type of agent to use, e.g., PPO, DQN, etc.
    
    model_path = 'agents\\game_obs\\Matts\\FPS\\Maximize Arousal\\MlpPolicy-Cluster0-1.0λ-run0-Episode-8000.zip'

    if label == 'Optimize':
        weight = 0
    elif label == 'Blended':
        weight = 0.5
    elif label == 'Arousal':
        weight = 1

    if env_name == "pirates":
        env = PiratesEnvironmentGameObs(
                    id_number=0,
                    weight=weight,
                    graphics=True,
                    cluster=cluster,
                    target_arousal=target_arousal,
                    period_ra=0,
                    discretize=0
                )
    elif env_name == "solid":
        env = SolidEnvironmentGameObs(
                    id_number=0,
                    weight=weight,
                    graphics=True,
                    cluster=cluster,
                    target_arousal=target_arousal,
                    period_ra=0,
                    discretize=0
                )
    elif env_name == "heist":
        env = HeistEnvironmentGameObs(
                    id_number=0,
                    weight=weight,
                    graphics=True,
                    cluster=cluster,
                    target_arousal=target_arousal,
                    period_ra=0,
                    discretize=0
                )
    
    # print(f"Action space: {env.action_space}, Observation space: {env.observation_space}")
    
    model = PPO("MlpPolicy", device='cpu', env=env)
    model.load(model_path)
    model.set_parameters(model_path)

    observations, actions, rewards, dones = [], [], [], []
    episode = 0
    
    restart_episode = 0  # False or True to resume from a specific episode if needed
    
    if restart_episode != 0:
        episode = restart_episode
        pickle_path = f'agents\\game_obs\\DT\\Datasets\\{env_name}\\{env_name}_{agentType}_{label}_Cluster{cluster}_Run{run}_dataset.pkl'
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        observations = dataset['observations'][:episode]
        actions = dataset['actions'][:episode]
        rewards = dataset['rewards'][:episode]
        dones = dataset['dones'][:episode]
    
    for episode in range(num_runs):
        
        print(f"Running episode {episode+1} of {num_runs}")
        state = env.reset()
        episode_observations, episode_actions, episode_rewards, episode_dones = [], [], [], []
        for i in range(steps):
            # Use the model to predict the action
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            # Append the data to the episode lists
            episode_observations.append(state)
            episode_actions.append(action)
            episode_dones.append(done)
            
            episode_rewards.append(reward)
            
            state = next_state
            
            if done:
                break

        episode += 1
        # Append episode data to the main lists
        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        dones.append(episode_dones)
        
        if episode % 50 == 0: # Save dataset every 50 episodes
            print(f"Saving dataset at episode {episode}")
            dataset = {
                'game': env_name,
                'comulative_reward': comulative_reward,
                'episodes': num_runs,
                'steps_per_episode': steps,
                'observations': observations,
                'observation_space': env.observation_space,
                'actions': actions,
                'acion_space': env.action_space,
                'rewards': rewards,
                'dones': dones
            }
            
            with open(f'agents\\game_obs\\DT\\Datasets\\{env_name}\\{env_name}_{agentType}_{label}_Cluster{cluster}_Run{run}_dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)
                
            print(f"Dataset checkpoint saved")
            
    print(f"{env_name} - {label} Agent finished")
    print(f"Dataset Created")
