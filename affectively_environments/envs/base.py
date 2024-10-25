import uuid
from abc import ABC
from typing import Tuple

import gym
import numpy as np
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from scipy import stats

from ..utils.logging import TensorBoardCallback
from ..utils.sidechannels import AffectivelySideChannel
from ..utils.surrogatemodel import KNNSurrogateModel


# TODO: data dtype?
def compute_confidence_interval(data,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
	Compute the confidence interval of some data.
	
	Args:
		data: The data.
		confidence: The confidence interval percentile.

	Returns: The confidence interval.
	"""
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

    def __init__(self, id_number, graphics, obs_space, path, weight, game, capture_fps=5, time_scale=1, args=None,
                 logging=True, log_prefix=""):

        super(BaseEnvironment, self).__init__()
        if args is None:
            args = []
        socket_id = uuid.uuid4()

        args += [f"-socketID", str(socket_id)]

        self.game_obs = []

        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=capture_fps, time_scale=time_scale)
        self.customSideChannel = AffectivelySideChannel(socket_id)
        self.env = self.load_environment(path, id_number, graphics, args)

        self.env = UnityToGymWrapper(self.env, allow_multiple_obs=True)

        self.action_space, self.action_size = self.env.action_space, self.env.action_space.shape

        try:
            dtype = obs_space['type']
        except:
            dtype = np.float32

        self.observation_space = gym.spaces.Box(low=obs_space['low'], high=obs_space['high'], shape=obs_space['shape'],
                                                dtype=dtype)

        self.model = KNNSurrogateModel(5, game)
        self.scaler = self.model.scaler

        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.arousal_trace = []

        self.current_score, self.current_reward, self.cumulative_reward = 0, 0, 0
        self.best_reward, self.best_score, self.best_cumulative_reward = 0, 0, 0

        self.previous_score = 0

        self.episode_length = 0
        self.weight = weight

        if weight == 0:
            label = 'optimize'
        elif weight == 0.5:
            label = 'blended'
        else:
            label = 'arousal'

        self.callback = None

        if logging:
            self.callback = TensorBoardCallback(f'./Tensorboard/{log_prefix}{game}-{label}-{id_number}', self)

    def reset(self, **kwargs):
        if self.callback is not None and len(self.arousal_trace) > 0:
            self.callback.on_episode_end()
        self.episode_length = 0
        self.current_reward, self.current_score, self.cumulative_reward, self.previous_score = 0, 0, 0, 0
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.arousal_trace = []
        state = self.env.reset()
        return state

    def step(self, action):
        self.episode_length += 1
        self.previous_score = self.current_score

        state, env_score, done, info = self.env.step(action)
        for _ in range(9):
            _, env_score, done, info = self.env.step(action)

        self.current_score = env_score
        self.best_score = np.max([self.current_score, self.best_score])

        if self.episode_length % 13 == 0:  # Request the surrogate vector 2 ticks in advanced due to potential delay
            self.create_and_send_message("Send Vector")

        arousal = 0
        if self.episode_length % 15 == 0:  # Read the surrogate vector on the 15th tick
            self.current_surrogate = np.array(self.customSideChannel.arousal_vector.copy(), dtype=np.float32)
            if self.current_surrogate.size != 0:
                try:
                    scaled_obs = np.array(self.scaler.transform(self.current_surrogate.reshape(1, -1))[0])
                except:
                    scaled_obs = np.zeros((len(self.previous_surrogate)))
                if self.previous_surrogate.size == 0:
                    self.previous_surrogate = np.zeros(len(self.current_surrogate))
                previous_scaler = np.array(self.scaler.transform(self.previous_surrogate.reshape(1, -1))[0])
                tensor = torch.Tensor(np.clip(list(previous_scaler) + list(scaled_obs), 0, 1))
                self.previous_surrogate = tensor
                arousal = self.model(tensor)[0]
                # print(arousal)
                if not np.isnan(arousal):
                    self.arousal_trace.append(arousal)
                self.previous_surrogate = self.current_surrogate.copy()
        return state, env_score, arousal, done, info

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

    def load_environment(self, path, identifier, graphics, args):
        try:
            env = UnityEnvironment(f"{path}",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics,
                                   args=args)
        except UnityEnvironmentException:
            print("Path not found! Please specify the right environment path.")
            raise
        except TypeError:
            try:
                env = UnityEnvironment(f"{path}",
                                       side_channels=[self.engineConfigChannel, self.customSideChannel],
                                       worker_id=identifier,
                                       no_graphics=not graphics,
                                       additional_args=args)
            except:
                print("Checking next ID!")
                return self.load_environment(path, identifier + 1, graphics, args)
        except:
            print("Checking next ID!")
            return self.load_environment(path, identifier + 1, graphics, args)
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
