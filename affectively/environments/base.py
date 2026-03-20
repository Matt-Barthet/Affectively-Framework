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

from affectively.utils import AffectivelySideChannel
from affectively.models.linear_model import LinearSurrogateModel


class BaseEnvironment(gym.Env, ABC):
    """
	This is the base unity-gym environment that all environments should inherit from. It sets up the
	unity-gym wrapper, configures the game engine parameters and sets up the custom side channel for
	communicating between our python scripts and unity's update loop.
	"""

    def __init__(self, id_number, graphics, obs_space, weight, game, capture_fps=5, time_scale=1, args=None,
                 target_arousal=1, cluster=0, period_ra=False, classifier=True, preference=True, decision_period=10,
                 imitate=False):

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

        if not isinstance(self.env.action_space, (gym.spaces.MultiDiscrete, gym.spaces.Discrete)):
            raise NotImplementedError("Action space type not supported")

        try:
            dtype = obs_space['type']
        except:
            dtype = np.float32

        self.discretize = False
        if imitate:
            obs_space['shape'] = (obs_space['shape'][0] + 3,)

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

        self.prev_score_error = 0

        self.surrogate_list = []
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.episode_arousal_trace, self.period_arousal_trace = [], []
        self.behavior_ticks = 0
        self.score_change = False
        self.period_ra = period_ra

        self.episode_length, self.arousal_episode_length = 0, 0

        if self.decision_period != 1 or self.game == "solid":
            # all_target_scores = [k for k, v in self.model.behavior_reward_book.items() if v != 0]
            # self.max_target_score = max(all_target_scores)
            pass
        else:
            self.max_target_score = 100

        self.target_arousal = target_arousal
        self.preference = preference
        self.classifier = classifier
        self.surrogate_length = self.model.surrogate_length
        self.callback = None
        self.imitation_learning = imitate
        self.multi_objective = self.weight == -1
        self.target_time_idx = 0

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
                if self.imitation_learning == 1:
                    added = [self.episode_length, len(self.episode_arousal_trace), self.target_time_idx]
                    state[modality] = np.concatenate((added, state[modality]))
                break

        self.cumulative_ra, self.cumulative_rb, self.cumulative_rl = 0, 0, 0
        self.current_score, self.previous_score = 0, 0
        self.score_change = False
        self.episode_length, self.arousal_episode_length = 0, 0
        self.behavior_ticks = 0
        self.prev_score_error = 0
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.episode_arousal_trace.clear()
        self.period_arousal_trace.clear()
        self.target_time_idx = 0
        return state

    def reward_behavior(self):
        """
        Return a behavior reward based on the agent's outward behavior in the level.

        * If ``self.score_change`` is False, returns 0 immediately.
        * With imitation learning disabled, always returns a flat reward of 1.
        * Otherwise, if the current score is within the known target range,
        compute the pacing reward where clipped to the [0, 1] interval.

        The function then clears the score‑change flag, updates the previous score,
        adds the reward to ``self.cumulative_rb``, and returns the reward value.
        """
        if not self.score_change:
            return 0

        self.behavior_ticks += 1


        r_b = 0
        if self.imitation_learning == 0:
            r_b = 1

        elif self.current_score <= self.max_target_score:
            all_target_scores = np.array([k for k, v in self.model.behavior_reward_book.items() if v != 0])
            future_targets = all_target_scores[all_target_scores >= self.current_score]
            if len(future_targets) > 0:
                nearest_target_score = np.min(future_targets)
                self.target_time_idx = self.model.behavior_reward_book[nearest_target_score]
                pacing_error = abs(self.episode_length - self.target_time_idx)
                reward_pacing = max(0.0, 1 - (pacing_error / 600))
                r_b = reward_pacing

        self.score_change = False
        self.previous_score = self.current_score
        self.cumulative_rb += r_b
        return r_b

    def reward_affect(self):
        """
        Calculates an affect reward based on the agent’s arousal trace.

        • When imitation learning is enabled:
            – If we are rewarding arousal asynchronously to behavior, the target 
            comes from the model’s arousal ordinal sequence for the current episode length.
            – Otherwise, the target comes from the model’s arousal reward book
            indexed by the current score.

        • When imitation learning is disabled, the target is the explicit
        target_arousal value.

        • In classifier mode the reward is 1 when the predicted label matches
        the target label (label 0 for mean arousal below 0.5, label 1 otherwise),
        otherwise it is 0.

        • In regression mode the reward is the squared proximity:
        (1 – absolute difference between target and mean arousal) squared,
        always clipped to the range 0–1.

        After computing the reward, the method clears the arousal trace,
        updates the current reward and the cumulative reward counter,
        and returns the calculated reward value.
        """
        mean_arousal = np.mean(self.period_arousal_trace) if len(self.period_arousal_trace) > 0 else 0
        if self.imitation_learning:
            if self.period_ra and self.episode_length < 600:
                if self.preference:
                    target = self.model.cluster_arousal_ordinal[self.episode_length]
                else:
                    target = self.model.cluster_arousal[self.episode_length]
            # Only reward if we are only within the score range of the cluster
            elif not self.period_ra and  self.current_score <= self.max_target_score:
                target = self.model.arousal_reward_book[self.current_score]
            else:
                return 0
        else:
            target = int(self.target_arousal)

        if self.classifier:
            mean_arousal_label = 0 if mean_arousal < 0.5 else 1
            r_a = 1 if mean_arousal_label == target else 0
        else:
            r_a = (1 - np.abs(target - mean_arousal))**2 

        self.ra = r_a
        self.cumulative_ra += r_a
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
        stacked_surrogates = np.stack(stacked_surrogates, axis=-1) 
        self.current_surrogate = np.mean(stacked_surrogates, axis=1) 
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

        if action[0] == np.inf and action[1] == np.inf:
            state, env_score, done, info = self.env.step((1,1,action[2]))
            return state, 0, done, info

        self.episode_length += 1
        self.arousal_episode_length += 1

        state, env_score, done, info = self.env.step(action)

        for modality in range(len(state)):
            if len(np.asarray(state[modality]).shape) == 1:
                surrogate = state[modality][-self.model.surrogate_length:]
                arousal_window = self.episode_arousal_trace[-5:]
                arousal_window = list(arousal_window) + list(np.zeros(5-len(arousal_window))) if len(arousal_window) < 5 else arousal_window
                state[modality] = np.concatenate((state[modality], arousal_window))
                if self.imitation_learning == 1 and not self.discretize:
                    # if self.episode_length < 600:
                    #     target_arousal = self.model.cluster_arousal[self.episode_length + 1]
                    added = [self.episode_length, len(self.episode_arousal_trace), self.target_time_idx]
                    state[modality] = np.concatenate((added, state[modality]))
                break

        self.surrogate_list.append(surrogate)
        self.current_score = env_score  
        change_in_score = self.current_score - self.previous_score
        self.score_change = self.score_change or change_in_score > 0

        if self.arousal_episode_length * self.decision_period % 150 == 0:  
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
                # print()
                # print(self.previous_score, self.current_score)
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
        except Exception as e:
            print(f"Checking next ID! Error {e}")
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
