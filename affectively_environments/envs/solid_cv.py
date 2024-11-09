import numpy as np
from matplotlib import pyplot as plt

from .base import BaseEnvironment
from gym import spaces

from .solid import SolidEnvironment


class SolidEnvironmentCV(SolidEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, grayscale=True, log_prefix=""):

        width = 128
        height = 96
        self.stackNo = 1
        self.grayscale = grayscale
        if grayscale:
            shape = (height, width, self.stackNo)
        else:
            shape = (height, width, 3)
        args = ['-bufferWidth', f"{width}", "-bufferHeight", f"{height}", "-useGrayscale", f"{grayscale}"]
        super().__init__(id_number=id_number, graphics=graphics,
                         obs={"low": 0, "high": 255, "shape": shape, "type": np.uint8},
                         path=path, weight=weight, frame_buffer=True, logging=logging, args=args, log_prefix=log_prefix)
        self.frame_buffer = []

    def construct_state(self, state) -> np.ndarray:
        self.game_obs = self.tuple_to_vector(state[1])

        visual_buffer = np.asarray(state[0])
        if self.grayscale:
            if len(self.frame_buffer) == 0:
                self.frame_buffer = [np.squeeze(visual_buffer)] * self.stackNo
            elif len(self.frame_buffer) == self.stackNo:
                self.frame_buffer.pop(0)
                self.frame_buffer.append(np.squeeze(visual_buffer))
            stacked_frames = np.stack(self.frame_buffer, axis=-1)
        else:
            stacked_frames = visual_buffer
            # plt.imshow(stacked_frames)
            # plt.show()
        return stacked_frames
