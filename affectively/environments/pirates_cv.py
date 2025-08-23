import numpy as np
from affectively.environments.pirates import PiratesEnvironment


class PiratesEnvironmentCV(PiratesEnvironment):

    def __init__(self, id_number, weight, grayscale, cluster, classifier=True, preference=True):
        self.width, self.height, self.stackNo = 128, 96, 1
        self.grayscale = grayscale
        if grayscale:
            shape = (self.height, self.width, 1)
        else:
            shape = (self.height, self.width, 3)
        args = ['-bufferWidth', f"{self.width}", "-bufferHeight", f"{self.height}", "-useGrayscale", f"{grayscale}"]
        super().__init__(id_number=id_number, graphics=True,
                         obs={"low": 0, "high": 255, "shape": shape, "type": np.uint8},
                         weight=weight, frame_buffer=True, args=args, cluster=cluster, classifier=classifier, preference=preference)
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
        return stacked_frames
