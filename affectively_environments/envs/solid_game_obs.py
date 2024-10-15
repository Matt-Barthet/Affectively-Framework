import numpy as np

from .solid import SolidEnvironment


def discretize_observations(game_obs):

    velocity = game_obs[0:3]
    velocity_discretized = [round(v / 20) * 20 for v in velocity]

    rotation = game_obs[3:6]
    rotation_discretized = [int(rot / 45) % 8 for rot in rotation]

    raycast_values = game_obs[6:42]
    raycast_bins = []
    for value in raycast_values:
        if value < 5:
            bin_value = 0
        elif value < 15:
            bin_value = 1
        elif value < 30:
            bin_value = 2
        elif value < 60:
            bin_value = 3
        else:
            bin_value = 4
        raycast_bins.append(bin_value)

    steering = game_obs[42]
    gas_pedal = game_obs[43]
    steering_bin = -1 if steering < 0 else (0 if steering == 0 else 1)
    gas_pedal_bin = -1 if gas_pedal < 0 else (0 if gas_pedal == 0 else 1)

    score = game_obs[44]
    if score < 8:
        score_bin = 0
    elif score < 16:
        score_bin = 1
    else:
        score_bin = 2

    is_off_road = game_obs[45]
    is_in_loop_zone = game_obs[46]
    is_jumping = game_obs[47]

    distance_to_checkpoint = game_obs[48]
    if distance_to_checkpoint < 10:
        distance_bin = 0
    elif distance_to_checkpoint < 30:
        distance_bin = 1
    elif distance_to_checkpoint < 60:
        distance_bin = 2
    elif distance_to_checkpoint < 100:
        distance_bin = 3
    else:
        distance_bin = 4

    angle = game_obs[49]
    if angle < 15:
        angle_bin = 0
    elif angle < 45:
        angle_bin = 1
    elif angle < 90:
        angle_bin = 2
    else:
        angle_bin = 3

    standing_state = game_obs[50]

    discretized_obs = (
        velocity_discretized +
        rotation_discretized +
        raycast_bins +
        [
            steering_bin,
            gas_pedal_bin,
            score_bin,
            is_off_road,
            is_in_loop_zone,
            is_jumping,
            distance_bin,
            angle_bin,
            standing_state
        ]
    )

    return discretized_obs


class SolidEnvironmentGameObs(SolidEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix="", discretize=False):
        self.discretize = discretize
        super().__init__(id_number=id_number, graphics=graphics,
                         obs={"low": -np.inf, "high": np.inf, "shape": (51,), "type": np.float32},
                         path=path, weight=weight, logging=logging, frame_buffer=False, log_prefix=log_prefix)

    def construct_state(self, state):
        if self.frameBuffer:
            game_obs = state[1]
        else:
            game_obs = state[0]

        if self.discretize:
            game_obs = discretize_observations(game_obs)
        else:
            game_obs = game_obs[3:]
        self.game_obs = self.tuple_to_vector(game_obs)
        return self.game_obs
