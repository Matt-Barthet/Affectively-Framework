from GANGenerate import generateNewLevel
import numpy as np

class GANWrapper:
    def generate(self, vector, worker_id):
        try:
            values = vector.tolist()
            level = generateNewLevel(values, worker_id)
            return level.astype(np.int32)
        except Exception as e:
            return self._empty_level()
        
    def _empty_level(self):
        return np.zeros((14, 32), dtype=np.int32)