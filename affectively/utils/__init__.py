import numpy as np
import scipy.stats as stats

def compute_confidence_interval(data, confidence: float = 0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return np.round(mean, 4), np.round(ci, 4)

def init_parser(parser):
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--game", required=True, help="Name of environment")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--periodic_ra", type=int, required=True, help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    parser.add_argument("--cv", required=True, type=int, help="0 for GameObs, 1 for CV")
    parser.add_argument("--headless", required=True, type=int, help="0 for headless mode, 1 for graphics mode")
    parser.add_argument("--logdir", required=True, help="Log directory for TensorBoard")
    parser.add_argument("--grayscale", required=True, type=int, help="0 for RGB, 1 for grayscale")
    parser.add_argument("--discretize", required=True, type=int, help="0 for continuous, 1 for discretized observations")
    parser.add_argument("--algorithm", required=True, help="Algorithm to use for training")
    parser.add_argument("--policy", required=False, help="Policy to use for training for PPO agents")
    parser.add_argument("--use_gpu", required=True, help="Use device GPU for models", type=int)
    parser.add_argument("--classifier", required=True, help="Use classifier model and reward for training", type=int)
    parser.add_argument("--preference", required=False, help="Use preference model for training", type=int)
    parser.add_argument("--decision_period", required=False, help="Decision period for environments", type=int, default=10)
    parser.add_argument("--max_retries", required=False, help="Max retries for Unity timeout recovery", type=int, default=500)
    parser.add_argument("--timesteps", required=False, help="Total timesteps for training", type=int, default=5_000_000)
    return parser