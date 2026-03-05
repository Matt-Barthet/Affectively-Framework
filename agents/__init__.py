from stable_baselines3 import PPO
import torch
import pickle

from agents.game_obs.go_explore.agent import Explorer
from agents.game_obs.rainbow_dqn.agent import RainbowAgent


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

def init_model(args):
    if args.algorithm.lower() == "ppo":
        model_class = PPO
    elif args.algorithm.lower() == "dqn":
        args.policy = "DQN"
        model_class = RainbowAgent
        if args.cv == 1:
            print("Model not implemented yet! Aborting...")
            exit()
    elif args.algorithm.lower() == "explore":
        model_class = Explorer
        args.discretize = 1
    elif args.algorithm.lower() in ["eq", "envelopeq", "envelope_q"]:
        model_class = "ENVELOPE_Q"
        args.weight = -1.0
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    return model_class