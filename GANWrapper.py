from GANGenerate import generateNewLevel
import numpy as np

class GANWrapper:
    def generate(self, vector):
        # vector = numpy array → convert to python list
        values = vector.tolist()

        # call your GAN function
        level = generateNewLevel(values)

        # return a placeholder observation for PPO (must match your observation_space)
        # example: 64x64 RGB black image
        return level.astype(np.int32)
