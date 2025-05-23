import platform
import uuid
from abc import ABC

import gym
import numpy as np
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from scipy import stats
from affectively.utils.sidechannels import AffectivelySideChannel
from affectively.utils.surrogatemodel import KNNSurrogateModel


def compute_confidence_interval(data,
                                confidence: float = 0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return np.round(mean, 4), np.round(ci, 4)


class BaseEnvironment(gym.Env, ABC):
    """
	This is the base unity-gym environment that all environments should inherit from. It sets up the
	unity-gym wrapper, configures the game engine parameters and sets up the custom side channel for
	communicating between our python scripts and unity's update loop.
	"""

    def __init__(self, id_number, graphics, obs_space, weight, game, capture_fps=5, time_scale=1, args=None,
                 target_arousal=1, cluster=0, period_ra=False):

        super(BaseEnvironment, self).__init__()
        if args is None:
            args = []
        socket_id = uuid.uuid4()

        args += [f"-socketID", str(socket_id)]

        self.game_obs = []
        self.game = game
        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=capture_fps, time_scale=time_scale)
        self.customSideChannel = AffectivelySideChannel(socket_id)
        self.env = self.load_environment(id_number, graphics, args)

        self.env = UnityToGymWrapper(self.env, allow_multiple_obs=True)

        self.action_space, self.action_size = self.env.action_space, self.env.action_space.shape

        # Hide actions - not working.
        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            # self.action_space = gym.spaces.MultiDiscrete(self.env.action_space.nvec[:-2])
            pass
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # Exclude the last two actions for Box
            # low = self.env.action_space.low[:-2] # Exclude the last two elements of low bounds
            # high = self.env.action_space.high[:-2] # Exclude the last two elements of high bounds
            # self.action_space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=self.env.action_space.dtype)
            pass
        else:

            raise NotImplementedError("Action space type not supported")

        try:
            dtype = obs_space['type']
        except:
            dtype = np.float32

        self.obs_size = obs_space['shape']
        self.observation_space = gym.spaces.Box(low=obs_space['low'], high=obs_space['high'], shape=obs_space['shape'],
                                                dtype=dtype)
        self.model = KNNSurrogateModel(5, game, cluster=cluster)
        self.scaler = self.model.scaler

        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.arousal_trace = []

        self.current_score, self.current_reward, self.cumulative_reward = 0, 0, 0
        self.best_reward, self.best_score, self.best_cumulative_reward = 0, 0, 0

        self.previous_score = 0

        self.episode_length = 0
        self.weight = weight

        self.save_digit, self.vector_digit, self.cell_name_digit = 0, 0, 0

        self.current_score, self.previous_score, = 0, 0
        self.best_rb, self.cumulative_rb = 0, 0
        self.best_ra, self.cumulative_ra = 0, 0
        self.best_rl, self.cumulative_rl = 0, 0

        self.surrogate_list = []
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.episode_arousal_trace, self.period_arousal_trace = [], []
        self.behavior_ticks = 0
        self.score_change = False
        self.period_ra = period_ra

        self.episode_length, self.arousal_episode_length = 0, 0
        self.target_arousal = target_arousal
        self.surrogate_length = self.model.surrogate_length
        self.callback = None
        self.create_and_send_message("[Save States]:Seed")

    def reset(self, **kwargs):
        if self.callback is not None and len(self.episode_arousal_trace) > 0:
            self.callback.on_episode_end()
        state = self.env.reset()
        self.cumulative_ra, self.cumulative_rb, self.cumulative_rl = 0, 0, 0
        self.current_score, self.previous_score = 0, 0
        self.episode_length, self.arousal_episode_length = 0, 0
        self.behavior_ticks = 0
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.episode_arousal_trace.clear()
        self.period_arousal_trace.clear()
        return state

    def reward_behavior(self):
        r_b = 1 if self.score_change else 0
        self.behavior_ticks += 1 if self.score_change else 0
        self.score_change = False
        self.best_rb = np.max([r_b, self.best_rb])
        self.cumulative_rb += r_b
        return r_b


    def reward_affect(self):
        # Reward similarity of mean arousal this period to target arousal (0 = minimize, 1 = maximize)
        mean_arousal = np.mean(self.period_arousal_trace) if len(self.period_arousal_trace) > 0 else 0 # Arousal range [0, 1]
        r_a = 1 - np.abs(self.target_arousal - mean_arousal)
        self.best_ra = np.max([self.best_ra, r_a])
        self.cumulative_ra += r_a
        self.period_arousal_trace.clear()
        return r_a


    def generate_arousal(self):
        arousal = 0
        stacked_surrogates = np.asarray(self.surrogate_list)
        stacked_surrogates = np.stack(stacked_surrogates, axis=-1) # stack the surrogates vertically
        self.current_surrogate = np.mean(stacked_surrogates, axis=1) # calculate the mean of each feature across the stack

        # print(list(self.current_surrogate))

        if self.current_surrogate.size != 0:
            scaled_obs = np.array(self.scaler.transform(self.current_surrogate.reshape(1, -1))[0])
            if self.previous_surrogate.size == 0:
                self.previous_surrogate = np.zeros(len(self.current_surrogate))
            previous_scaler = np.array(self.scaler.transform(self.previous_surrogate.reshape(1, -1))[0])
            unclipped_tensor = np.array(list(previous_scaler) + list(scaled_obs))
            
            if np.min(scaled_obs) < 0 or np.max(scaled_obs) > 1:
                # print(f"Values outside of range: Max={np.max(scaled_obs):.3f}@{self.model.columns[np.argmax(scaled_obs)]}(other={np.where(scaled_obs > 1)[0]})", end=", ")
                # print(f"Min={np.min(scaled_obs):.3f}@{self.model.columns[np.argmin(scaled_obs)]}(other={np.where(scaled_obs < 0)[0]})")
                pass
            
            tensor = torch.Tensor(np.clip(unclipped_tensor, 0, 1))
            tensor= torch.nan_to_num(tensor, nan=0)
            self.previous_surrogate = previous_scaler
            arousal = self.model(tensor)[0]
            if not np.isnan(arousal):
                self.episode_arousal_trace.append(arousal)
                self.period_arousal_trace.append(arousal)
            self.previous_surrogate = self.current_surrogate.copy()
            self.customSideChannel.arousal_vector.clear()
        return arousal

    def step(self, action):
        self.episode_length += 1
        self.arousal_episode_length += 1

        change_in_score = (self.current_score - self.previous_score)
        self.score_change = self.score_change or change_in_score > 0
        self.previous_score = self.current_score
                
        try:
            state, env_score, done, info = self.env.step(list(action)) 
        except:
            print("Caught step error, trying again to bypass double agent error on reset...")
            state, env_score, done, info = self.env.step(list(action))

        for modality in state:
            if len(np.asarray(modality).shape) == 1:
                surrogate = modality[-self.surrogate_length:]
                break
        
        self.surrogate_list.append(surrogate)

        if self.arousal_episode_length % 15 == 0:  # Read the surrogate vector on the 15th tick
            self.generate_arousal()
            self.arousal_episode_length = 0
            self.surrogate_list.clear()

        final_reward = 0

        if self.period_ra and (len(self.episode_arousal_trace) > 0):
            print("assigning reward asynchronously!")
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)

        elif not self.period_ra and self.score_change:
            print("assigning reward synchronously based on score change!")
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)


        self.cumulative_rl += final_reward
        self.best_rl = np.max([self.best_rl, final_reward])

        self.current_score = env_score

        return state, final_reward, done, info

    def handle_level_end(self):
        """
		Override this method to handle a "Level End" Message from the unity environment
		"""
        pass

    def construct_state(self, state):
        """
		Override this method to add any custom code for reading the state received from unity.
		"""
        return state

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def load_environment(self, identifier, graphics, args):
        system = platform.system()
        if system == "Darwin":
            game_suffix = "app"
        elif system == "Linux":
            game_suffix = "x86_64"
        else:
            game_suffix = "exe"
        system="Mac" if system == "Darwin" else system
        try:
            env = UnityEnvironment(f"./affectively/builds/{self.game}/{system}/{self.game}.{game_suffix}",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics,
                                   additional_args=args)
        except:
            print("Checking next ID!") 
            # raise # the error if you get a stack overflow
            return self.load_environment(identifier + 1, graphics, args)
        return env

    @staticmethod
    def tuple_to_vector(s):
        obs = []
        for i in range(len(s)):
            obs.append(s[i])
        return obs

    @staticmethod
    def one_hot_encode(matrix_obs, num_categories):
        one_hot_encoded = np.zeros((matrix_obs.shape[0], matrix_obs.shape[1], num_categories))
        for i in range(matrix_obs.shape[0]):
            for j in range(matrix_obs.shape[1]):
                one_hot_encoded[i, j, int(matrix_obs[i][j]) - 1] = 1
        return one_hot_encoded
