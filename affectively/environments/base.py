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
from affectively.models.linear_model import LinearSurrogateModel
from affectively.models.mlp_model import MLPSurrogateModel


def compute_confidence_interval(data, confidence: float = 0.95):
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
                 target_arousal=1, cluster=0, period_ra=False, classifier=True, preference=True, decision_period=10):

        super(BaseEnvironment, self).__init__()
        if args is None:
            args = []
        socket_id = uuid.uuid4()

        self.decision_period = decision_period
        args += [f"-socketID", str(socket_id), "-decisionPeriod", str(decision_period)]

        self.game_obs = []
        self.game = game
        self.engineConfigChannel = EngineConfigurationChannel()

        if np.sign(capture_fps) == -1:
            self.engineConfigChannel.set_configuration_parameters(target_frame_rate=-capture_fps, time_scale=time_scale)
        else:
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
        self.model = LinearSurrogateModel(game=game, cluster=cluster, classifier=classifier, preference=preference)
        self.cluster = cluster
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.arousal_trace = []

        self.weight = weight

        self.save_digit, self.vector_digit, self.cell_name_digit = 0, 0, 0
        self.current_score, self.previous_score, = 0, 0

        self.ra, self.rb, self.rl = [], [], []
        self.cumulative_ra, self.cumulative_rb, self.cumulative_rl = 0, 0, 0

        self.surrogate_list = []
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.episode_arousal_trace, self.period_arousal_trace = [], []
        self.behavior_ticks = 0
        self.score_change = False
        self.period_ra = period_ra

        self.episode_length, self.arousal_episode_length = 0, 0
        self.target_arousal = target_arousal
        self.preference = preference
        self.classifier = classifier
        self.surrogate_length = self.model.surrogate_length
        self.callback = None
        self.multi_objective = self.weight == -1
        if self.multi_objective:
            self.reward_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2,),
                dtype=np.float32
            )

    def reinit(self):
        self.model = LinearSurrogateModel(game=self.game, cluster=self.cluster, classifier=self.classifier, preference=self.preference)

    def reset(self, **kwargs):
        state = self.env.reset()
        
        for modality in range(len(state)):
            if len(np.asarray(state[modality]).shape) == 1:
                arousal_window = self.episode_arousal_trace[-5:]
                arousal_window = list(arousal_window) + list(np.zeros(5-len(arousal_window))) if len(arousal_window) < 5 else arousal_window
                state[modality] = np.concatenate((state[modality], arousal_window))
                break

        self.cumulative_ra, self.cumulative_rb, self.cumulative_rl = 0, 0, 0
        self.current_score, self.previous_score = 0, 0
        self.score_change = False
        self.episode_length, self.arousal_episode_length = 0, 0
        self.behavior_ticks = 0
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.episode_arousal_trace.clear()
        self.period_arousal_trace.clear()
        return state

    def reward_behavior(self):
        """
        Behavior reward component: optimize behavior and update reward statistics.
        Assign behavior reward if the environment score increases.
        Zero rewards otherwise.
        """
        r_b = 1 if self.score_change else 0
        self.behavior_ticks += 1 if self.score_change else 0
        self.score_change = False
        self.cumulative_rb += r_b
        # print(f"Behavior rewarded, cumulative reward: {self.cumulative_rb}")
        return r_b

    def reward_affect(self):
        """
        Affect reward component: optimize behavior and update reward statistics.
        If environment is in classifier mode, reward based on predicting correct target label.
        If environment is in regression mode, reward using Mean Squared Error to target value.
        """
        mean_arousal = np.mean(self.period_arousal_trace) if len(self.period_arousal_trace) > 0 else 0

        if self.classifier:
            mean_arousal_label = 0 if mean_arousal < 0.5 else 1
            r_a = 1 if mean_arousal_label == self.target_arousal else 0 # Binary classification
        else:
            r_a = (1 - np.abs(self.target_arousal - mean_arousal))**2 # MSE for regression tasks - Inverted to reward proximity 

        self.ra = r_a
        self.cumulative_ra += r_a
        # print("\n", self.period_arousal_trace, mean_arousal, r_a, self.cumulative_ra)
        self.period_arousal_trace.clear()
        return r_a


    def generate_arousal(self):
        """
        Generate an arousal value for the previous time window using the following process:
        1) Take the list of surrogate vectors for this time window and stack them vertically.
        2) Calculate the mean of each element and normalize using the loaded minmax scaler.
        3) If the environment is in preference mode, concatenate current mean vector with the last time window's vector.
        4) Create a tensor with the vector and pass it through the arousal model.
        Returns: arousal value between 0 and 1.
        """
        arousal = 0
        stacked_surrogates = np.asarray(self.surrogate_list)
        stacked_surrogates = np.stack(stacked_surrogates, axis=-1) # stack the surrogates vertically
        self.current_surrogate = np.mean(stacked_surrogates, axis=1) # calculate the mean of each feature across the stack

        if self.current_surrogate.size != 0:
            if self.previous_surrogate.size == 0:
                self.previous_surrogate = np.zeros(len(self.current_surrogate))
            previous_scaler = np.array(self.previous_surrogate)

            if self.preference:
                tensor = np.array(list(previous_scaler) + list(self.current_surrogate))
            else:
                tensor = self.current_surrogate

            tensor= np.nan_to_num(torch.tensor(tensor), nan=0)
            arousal = np.clip(self.model(tensor), 0, 1)
            self.episode_arousal_trace.append(arousal)
            self.period_arousal_trace.append(arousal)
            self.previous_surrogate = self.current_surrogate.copy()
            self.customSideChannel.arousal_vector.clear()
        return arousal

    def step(self, action):

        # If a load is called don't do the rest of the env logic.
        if action[0] == np.inf and action[1] == np.inf:
            state, env_score, done, info = self.env.step((1,1,action[2]))
            return state, 0, done, info

        self.episode_length += 1
        self.arousal_episode_length += 1
        self.previous_score = self.current_score

        state, env_score, done, info = self.env.step(action)

        for modality in range(len(state)):
            if len(np.asarray(state[modality]).shape) == 1:
                surrogate = state[modality][-self.surrogate_length:]
                arousal_window = self.episode_arousal_trace[-5:]
                arousal_window = list(arousal_window) + list(np.zeros(5-len(arousal_window))) if len(arousal_window) < 5 else arousal_window
                state[modality] = np.concatenate((state[modality], arousal_window))
                break

        self.surrogate_list.append(surrogate)
        self.current_score = env_score  
        change_in_score = (self.current_score - self.previous_score)
        self.score_change = self.score_change or change_in_score > 0

        if self.arousal_episode_length * self.decision_period % 150 == 0:  # Read the surrogate vector on the 15th tick
            self.generate_arousal()
            self.arousal_episode_length = 0
            self.surrogate_list.clear()

        if self.multi_objective:
            final_reward = np.array([0,0], dtype=np.float32)
            if self.period_ra and (len(self.period_arousal_trace) > 0):
                final_reward = np.array([self.reward_behavior(), self.reward_affect()], dtype=np.float32)
            elif not self.period_ra and self.score_change:
                final_reward = np.array([self.reward_behavior(), self.reward_affect()], dtype=np.float32)   
        else:
            final_reward = 0

            if self.period_ra and (len(self.period_arousal_trace) > 0):
                final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)

            elif not self.period_ra and self.score_change:
                final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)
            self.cumulative_rl += final_reward
            
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

    def sample_weighted_action(self):
        raise NotImplementedError()

    def sample_action(self):
        try:
            return self.sample_weighted_action()
        except NotImplementedError:
            return self.action_space.sample()

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
            env = UnityEnvironment(f"./affectively/builds/{self.game.lower()}/{system}/{self.game.lower()}.{game_suffix}",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics,
                                   additional_args=args)
        except:
            print("Checking next ID!") 
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
