import torch
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
)
import numpy as np
import random
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# import imageio

from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.base import compute_confidence_interval
import csv
# from trainObsDT import plot_metrics

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Evaluation (Online Rollout) ---
def evaluate_online(env, model_path, target_rtg, num_episodes=10, output_attentions=False):
    """Evaluates the trained model online in the environment."""

    logger.info(f"Starting online evaluation with target RTG: {target_rtg}")
    if not model_path or not os.path.exists(model_path):
         logger.error(f"Model path not found: {model_path}")
         return None, None # Cannot evaluate without model

    # Load model and normalization stats
    try:
        model = DecisionTransformerModel.from_pretrained(model_path)
        model.to(DEVICE).eval()

        stats_path = os.path.join(model_path, "normalization_stats.npz")
        stats = np.load(stats_path)
        state_mean = stats['mean']
        state_std = stats['std']
        max_ep_len = int(stats['max_ep_len'])
    except Exception as e:
         logger.error(f"Error loading model or stats from {model_path}: {e}")
         return None, None


    # Action space details (must match training)
    _num_actions_per_dim = NUM_ACTIONS_PER_DIM
    _num_action_dims = len(_num_actions_per_dim)
    _action_slice_starts = np.concatenate(([0], np.cumsum(_num_actions_per_dim)[:-1]))
    _concatenated_action_dim = sum(_num_actions_per_dim)

    # Create environment
    try:
        env.reset()
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return None, None


    episode_returns = []
    episode_lengths = []
    arousal, scores = [], []

    ep = 0
    
    while ep < num_episodes:
    # for ep in range(num_episodes):
        
        # --- DATA COLLECTION SETUP FOR THE FIRST EPISODE ---
        if ep == 0 and output_attentions:
            attention_history_line = {
                "rtg": [],
                "state": [],
                "action": []
            }
            attention_history_heatmap = []
        
        try:
            state = env.reset()
            state = np.array(state, dtype=np.float32) # Ensure numpy float32
            if state.size > STATE_DIM: #To manage extra info in observation
                logger.warning(f"Initial state shape mismatch: got {state.size}, expected {STATE_DIM}. truncating.")
                state = state[:STATE_DIM]
                    
        except Exception as e:
            logger.error(f"Environment reset failed: {e}")
            continue # Skip episode if reset fails

        done = False
        ep_return = 0
        ep_len = 0
        current_target_rtg = float(target_rtg) # Ensure target is float

        # Initialize context buffers
        # States need normalization
        norm_state = (state - state_mean) / state_std
        context_states = [np.zeros_like(state, dtype=np.float32)] * (CONTEXT_LENGTH - 1) + [norm_state]
        # Actions are one-hot encoded for the model, but store indices for history
        context_actions_one_hot = [np.zeros(_concatenated_action_dim, dtype=np.float32)] * CONTEXT_LENGTH
        context_rtgs = [np.array([current_target_rtg], dtype=np.float32)] * CONTEXT_LENGTH
        context_timesteps = [np.array([0], dtype=np.int64)] * CONTEXT_LENGTH


        while not done:
            # Prepare model input tensors from context
            states_tensor = torch.tensor(np.array(context_states)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            actions_tensor = torch.tensor(np.array(context_actions_one_hot)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            rtgs_tensor = torch.tensor(np.array(context_rtgs)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Extract scalar timesteps into a list
            current_context_timesteps_list = [ts[0] for ts in context_timesteps[-CONTEXT_LENGTH:]]
            # Create a 1D numpy array
            timesteps_np = np.array(current_context_timesteps_list, dtype=np.int64) # Shape (K,)
            # Convert to tensor and add batch dimension
            timesteps_tensor = torch.tensor(timesteps_np, dtype=torch.long).unsqueeze(0).to(DEVICE) # Shape (1, K)

            attn_mask_tensor = torch.ones_like(timesteps_tensor).to(DEVICE) # Shape (1, K)

            with torch.no_grad():
                outputs = model(
                    states=states_tensor,
                    actions=actions_tensor,
                    returns_to_go=rtgs_tensor,
                    timesteps=timesteps_tensor, # Pass the (1, K) tensor
                    attention_mask=attn_mask_tensor,
                    return_dict=True,
                    output_attentions=output_attentions,
                )
                # Get logits for the last timestep prediction
                logits = outputs.action_preds[0, -1] # Shape (concatenated_logit_dim,) e.g., (8,)
                
                # --- ATTENTION ANALYSIS FOR THE FIRST EPISODE ---
                if ep == 0 and outputs.attentions:
                    last_layer_attention = outputs.attentions[-1] # Shape: (1, num_heads, 3*K, 3*K)
                    
                    # Average over all attention heads
                    avg_attention = last_layer_attention.squeeze(0).mean(dim=0).cpu().numpy()
                    
                    # We want the attention row for the *current state prediction*.
                    # The model uses the state token to predict the next action.
                    # This token is at index (3 * K) - 2.
                    s_last_idx = (CONTEXT_LENGTH * 3) - 2
                    attention_row = avg_attention[s_last_idx, :]

                    # The input tokens are interleaved: [R1, S1, A1, R2, S2, A2, ...]
                    # We can sum the attention paid to each modality using slicing.
                    sum_rtg_att = np.sum(attention_row[0::3])
                    sum_state_att = np.sum(attention_row[1::3])
                    sum_action_att = np.sum(attention_row[2::3])

                    # Append to our history lists
                    attention_history_line["rtg"].append(sum_rtg_att)
                    attention_history_line["state"].append(sum_state_att)
                    attention_history_line["action"].append(sum_action_att)
                    attention_history_heatmap.append([sum_rtg_att, sum_state_att, sum_action_att])
                

            # Sample action INDICES from the logits for each dimension
            predicted_action_indices = np.zeros(_num_action_dims, dtype=np.int64)
            predicted_action_one_hot = np.zeros(_concatenated_action_dim, dtype=np.float32) # For next input
            for i in range(_num_action_dims):
                start_idx = _action_slice_starts[i]
                end_idx = start_idx + _num_actions_per_dim[i]
                dim_logits = logits[start_idx:end_idx]
                # Take argmax for deterministic evaluation
                action_idx = torch.argmax(dim_logits).item()
                predicted_action_indices[i] = action_idx
                # Set the corresponding one-hot bit for the next input
                predicted_action_one_hot[start_idx + action_idx] = 1.0
                

            # logger.info(f"Predicted action: {predicted_action_indices}")

            # Step environment with action INDICES
            try:
                next_state, reward, done, _ = env.step(predicted_action_indices)
                next_state = np.array(next_state, dtype=np.float32) # Ensure numpy float32
                if next_state.size > STATE_DIM: #To manage extra info in observation
                    # logger.warning(f"Initial state shape mismatch: got {state.shape}, expected {STATE_DIM}. truncating.")
                    next_state = next_state[:STATE_DIM]
                reward = float(reward) # Ensure float
            except Exception as e:
                logger.error(f"Environment step failed: {e}")
                done = True # End episode if step fails


            # Update context buffers for the next iteration
            norm_next_state = (next_state - state_mean) / state_std
            context_states.append(norm_next_state)
            context_actions_one_hot.append(predicted_action_one_hot) # Append the one-hot action taken
            current_target_rtg -= reward
            context_rtgs.append(np.array([current_target_rtg], dtype=np.float32))
            current_timestep = context_timesteps[-1][0] + 1
            # Clamp timestep to avoid exceeding max_ep_len used for embeddings
            context_timesteps.append(np.array([min(current_timestep, max_ep_len - 1)], dtype=np.int64))

            # Remove oldest entries from context buffers
            context_states.pop(0)
            context_actions_one_hot.pop(0)
            context_rtgs.pop(0)
            context_timesteps.pop(0)

            ep_return += reward
            ep_len += 1
            state = next_state # Update state for next loop

            if ep_len >= max_ep_len: # Add safety break
                 logger.warning(f"Episode {ep} reached max length {max_ep_len}, terminating.")
                 done = True

        if ep == 0 and output_attentions:
            # Save attention history to file    
            with open(ATTENTION_HISTORY_PATH.replace('.txt', '.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['RTG', 'State', 'Action'])
                writer.writerows(attention_history_heatmap)
            logger.info(f"Saved attention history to {ATTENTION_HISTORY_PATH.replace('.txt', '.csv')}")
            
            # plot_episode_attention_heatmap(
            #     ATTENTION_HISTORY_PATH,
            #     dt_name_for_plot
            # )
            # exit()
            
        if len(env.episode_arousal_trace) == 0:
            print("No arousal data collected.") #To avoid env error if no arousal data collected
        else:
            ep += 1
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)
            arousal.append(np.mean(env.episode_arousal_trace))
            scores.append(env.current_score)
        
            logger.info(f"Episode {ep-1}/{num_episodes}: Return={ep_return:.2f}, Length={ep_len}, Arousal={arousal[-1]:.2f}, Score={scores[-1]:.2f}")
        

    if not episode_returns:
        logger.warning("No episodes completed successfully during evaluation.")
        return None, None
    
    env.close()

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    logger.info("-" * 30)
    logger.info(f"Evaluation Results (Target RTG: {target_rtg}):")
    logger.info(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    logger.info(f"Mean Length: {mean_length:.2f}")
    logger.info(f"Avg. Score: {compute_confidence_interval(scores)}")
    logger.info(f"Norm. Arousal: {compute_confidence_interval(arousal)}")
    logger.info("-" * 30)
    
    return mean_return, std_return, arousal, scores
    
# To record a GIF of an episode, we need to run the model in the environment and capture frames.

def record_gif_episode(env_creator, model_path, target_rtg, gif_path="episode.gif", max_frames=200):
    """Runs one episode, records frames, and saves a GIF."""
    # Load model and normalization stats (reuse your logic)
    model = DecisionTransformerModel.from_pretrained(model_path)
    model.to(DEVICE).eval()
    stats_path = os.path.join(model_path, "normalization_stats.npz")
    stats = np.load(stats_path)
    state_mean = stats['mean']
    state_std = stats['std']
    max_ep_len = int(stats['max_ep_len'])

    _num_actions_per_dim = NUM_ACTIONS_PER_DIM
    _num_action_dims = len(_num_actions_per_dim)
    _action_slice_starts = np.concatenate(([0], np.cumsum(_num_actions_per_dim)[:-1]))
    _concatenated_action_dim = sum(_num_actions_per_dim)

    env = env_creator()
    state = env.reset()
    if isinstance(state, tuple):  # Some envs return (obs, info)
        state = state[0]
    state = np.array(state, dtype=np.float32)
    norm_state = (state - state_mean) / state_std
    context_states = [np.zeros_like(state, dtype=np.float32)] * (CONTEXT_LENGTH - 1) + [norm_state]
    context_actions_one_hot = [np.zeros(_concatenated_action_dim, dtype=np.float32)] * CONTEXT_LENGTH
    context_rtgs = [np.array([target_rtg], dtype=np.float32)] * CONTEXT_LENGTH
    context_timesteps = [np.array([0], dtype=np.int64)] * CONTEXT_LENGTH

    frames = []
    done = False
    ep_len = 0
    current_target_rtg = float(target_rtg)

    # --- Capture initial frame ---
    if hasattr(env, "render"):
        frame = env.render(mode="rgb_array") if "mode" in env.render.__code__.co_varnames else env.render()
        if frame is not None:
            frames.append(frame)
        else:
            print("Warning: Initial render returned None.")
    else:
        print("Warning: Environment does not support render.")
    
    while not done and ep_len < max_frames:
        states_tensor = torch.tensor(np.array(context_states)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        actions_tensor = torch.tensor(np.array(context_actions_one_hot)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        rtgs_tensor = torch.tensor(np.array(context_rtgs)[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        current_context_timesteps_list = [ts[0] for ts in context_timesteps[-CONTEXT_LENGTH:]]
        timesteps_np = np.array(current_context_timesteps_list, dtype=np.int64)
        timesteps_tensor = torch.tensor(timesteps_np, dtype=torch.long).unsqueeze(0).to(DEVICE)
        attn_mask_tensor = torch.ones_like(timesteps_tensor).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                states=states_tensor,
                actions=actions_tensor,
                returns_to_go=rtgs_tensor,
                timesteps=timesteps_tensor,
                attention_mask=attn_mask_tensor,
                return_dict=True,
            )
            # Get logits for the last timestep prediction
            logits = outputs.action_preds[0, -1]

        predicted_action_indices = np.zeros(_num_action_dims, dtype=np.int64)
        predicted_action_one_hot = np.zeros(_concatenated_action_dim, dtype=np.float32)
        for i in range(_num_action_dims):
            start_idx = _action_slice_starts[i]
            end_idx = start_idx + _num_actions_per_dim[i]
            dim_logits = logits[start_idx:end_idx]
            action_idx = torch.argmax(dim_logits).item()
            predicted_action_indices[i] = action_idx
            predicted_action_one_hot[start_idx + action_idx] = 1.0

        next_state, reward, terminated, _ = env.step(predicted_action_indices)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = np.array(next_state, dtype=np.float32)
        done = terminated

        norm_next_state = (next_state - state_mean) / state_std
        context_states.append(norm_next_state)
        context_actions_one_hot.append(predicted_action_one_hot)
        current_target_rtg -= reward
        context_rtgs.append(np.array([current_target_rtg], dtype=np.float32))
        current_timestep = context_timesteps[-1][0] + 1
        context_timesteps.append(np.array([min(current_timestep, max_ep_len - 1)], dtype=np.int64))
        context_states.pop(0)
        context_actions_one_hot.pop(0)
        context_rtgs.pop(0)
        context_timesteps.pop(0)

        ep_len += 1

        # --- Capture frame ---
        if hasattr(env, "render"):
            frame = env.render(mode="rgb_array") if "mode" in env.render.__code__.co_varnames else env.render()
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Render returned None at frame {ep_len}.")
        else:
            print("Warning: Environment does not support render.")
        
    # Save GIF if any frames were captured
    if frames:
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"Saved episode GIF to {gif_path}")
    else:
        print("No valid frames were captured. GIF was not saved.")
        
# --- Constants and Hyperparameters ---

NUM_ACTIONS_PER_DIM = [3, 3, 2]  # <<< --- MUST BE SET CORRECTLY
NUM_ACTION_DIMS = len(NUM_ACTIONS_PER_DIM)
STATE_DIM = 86
CONTEXT_LENGTH = 5 # K: How many steps the model sees
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTENTION_HISTORY_PATH = ""

# --- Main Execution ---
if __name__ == "__main__":
    
    data_from = 'PPO' # 'PPO' or 'Explore'
    
    # Target return for evaluation
    starting_target_reward = 40 
    final_target_reward = starting_target_reward  # Set to same as starting for attention analysis
    
    target_arousal = 0 # 1 to maximize arousal, 0 to minimize
    period_ra = 0
    label = 'Arousal' # 'Optimize', 'Blended', 'Arousal'
    cluster = 0
    run = 0
    env_name = "heist" # solid, pirates, heist, or dummy for testing
    plot_ep_attention = False # True or False for attention plotting during evaluation
    num_episodes = 30

    if label == 'Optimize':
        weight = 0
    elif label == 'Blended':
        weight = 0.5
    elif label == 'Arousal':
        weight = 1
    
    discretize=0
    
    if data_from == 'Explore':
        discretize=1
        
    if plot_ep_attention:
        num_episodes = 1  # Only need one episode for attention plotting
        
        
    if env_name == "pirates":
        env = PiratesEnvironmentGameObs(
                    id_number=0,
                    weight=weight,
                    graphics=True,
                    cluster=cluster,
                    target_arousal=target_arousal,
                    period_ra=period_ra,
                    discretize=discretize
                )
        STATE_DIM = 288
        NUM_ACTIONS_PER_DIM = [3, 2, 2] 
        NUM_ACTION_DIMS = len(NUM_ACTIONS_PER_DIM)
    elif env_name == "solid":
        env = SolidEnvironmentGameObs(
                    id_number=0,
                    weight=weight,
                    graphics=True,
                    cluster=cluster,
                    target_arousal=target_arousal,
                    period_ra=period_ra,
                    discretize=discretize
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
        STATE_DIM = 152
        NUM_ACTIONS_PER_DIM = [9, 9, 3, 3, 2, 2] 
        NUM_ACTION_DIMS = len(NUM_ACTIONS_PER_DIM)
    elif env_name == "dummy":
        env = create_dummy_env()
        
        
    dt_name = f"DT_{env_name}_{data_from}_{label}_cluster{cluster}_run{run}_final"
    
    # Set the path to the saved model artifacts
    final_model_path = f"agents\\game_obs\\DT\\Results\\{env_name}\\{dt_name}"
    # final_model_path = f"examples\\Agents\\DT\\Results\\Explore_Blended_moreTrained_DT"
    
    ATTENTION_HISTORY_PATH = f"agents\\game_obs\\DT\\Attentions\\{env_name}\\DT_{env_name}_{label}_attention_history.txt"
    
    print(f'Starting to evaluate {dt_name}')

    # Run evaluation from starting to final target return
    results_dict = {}
    while starting_target_reward <= final_target_reward:
        mean_return, std_return, arousal, scores = evaluate_online(
            env=env,
            model_path=final_model_path,
            target_rtg=starting_target_reward,
            num_episodes=num_episodes,
            output_attentions=plot_ep_attention
        )
        results_dict[starting_target_reward] = {
            "arousal": compute_confidence_interval(arousal),
            "scores": compute_confidence_interval(scores)
        }
        starting_target_reward += 10  # Increment target return for next evaluation
        
        # Save results to a .txt file
        with open(f"agents\\game_obs\\DT\\{dt_name}_evaluation.txt", "w") as f:
            for target_rtg, metrics in results_dict.items():
                arousal_ci = metrics["arousal"]
                scores_ci = metrics["scores"]
                f.write(f"Target RTG: {target_rtg} => Scores CI: [{scores_ci[0]:.4f}, {scores_ci[1]:.4f}] Norm. Arousal CI: [{arousal_ci[0]:.4f}, {arousal_ci[1]:.4f}]\n")
    
    print(f"Evaluation Summary for DT with CW = {CONTEXT_LENGTH}:")
    for target_rtg, metrics in results_dict.items():
        arousal_ci = metrics["arousal"]
        scores_ci = metrics["scores"]
        print(f"Target RTG: {target_rtg} => Scores CI: [{scores_ci[0]:.4f}, {scores_ci[1]:.4f}] Norm. Arousal CI: [{arousal_ci[0]:.4f}, {arousal_ci[1]:.4f}]")
    # print(f"Avg. Score: {compute_confidence_interval(scores)}, Norm. Arousal: {compute_confidence_interval(arousal)}")
    print(f'Done evaluating {dt_name} in {label}')
    
    # --- Record a GIF of one episode ---
    # record_gif_episode(
    #     env_creator=create_env,
    #     model_path=final_model_path,
    #     target_rtg=target_return,
    #     gif_path=f"examples\\Agents\\DT\\GIF\{data_from}_{label}.gif",
    #     max_frames=200
    # )