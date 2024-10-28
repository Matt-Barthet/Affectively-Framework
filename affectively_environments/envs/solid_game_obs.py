import numpy as np

from .solid import SolidEnvironment


def discretize_observations(game_obs):

    position = game_obs[:3]
    position_disc = [round(v / 40) * 40 for v in position]

    velocity = game_obs[3:6]
    velocity_discretized = np.round(np.linalg.norm(velocity) / 30) * 30

    # rotation = game_obs[6:9]
    # rotation_discretized = [int(rot / 45) % 8 for rot in rotation]

    # raycast_values = game_obs[9:45]
    # raycast_bins = []
    # for idx in range(0, len(raycast_values), 8):
    #     value = raycast_values[idx]
    #     if value < 10:
    #         bin_value = 0
    #     else:
    #         bin_value = 1
    #     raycast_bins.append(bin_value)

    # steering = game_obs[45]
    # gas_pedal = game_obs[46]
    # steering_bin = -1 if steering < 0 else (0 if steering == 0 else 1)
    # gas_pedal_bin = -1 if gas_pedal < 0 else (0 if gas_pedal == 0 else 1)

    score = game_obs[47]
    if score < 8:
        score_bin = 0
    elif score < 16:
        score_bin = 1
    else:
        score_bin = 2

    is_off_road = game_obs[48]
    is_in_loop_zone = game_obs[49]
    # is_jumping = game_obs[50]

    # distance_to_checkpoint = game_obs[51]
    # if distance_to_checkpoint < 10:
    #     distance_bin = 0
    # elif distance_to_checkpoint < 30:
    #     distance_bin = 1
    # elif distance_to_checkpoint < 60:
    #     distance_bin = 2
    # elif distance_to_checkpoint < 100:
    #     distance_bin = 3
    # else:
    #     distance_bin = 4

    angle = game_obs[52]
    if angle < 90:
        angle_bin = 0
    else:
        angle_bin = 1

    # standing_state = game_obs[53]

    discretized_obs = (
        position_disc +
        # rotation_discretized +
        # raycast_bins +
        [
            velocity_discretized,
            # steering_bin,
            # gas_pedal_bin,
            score_bin,
            is_off_road,
            is_in_loop_zone,
            # is_jumping,
            # distance_bin,
            angle_bin,
            # standing_state
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
