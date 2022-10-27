import numpy as np

from distractor_dmc2gym.distractors import ImageSource


class RandomColorSource(ImageSource):
    def __init__(self, *args, intensity=1):
        super().__init__(*args)
        self.intensity = intensity
        self.bg = np.zeros((self.shape[0], self.shape[1], 3))
        self.mask = np.ones((self.shape[0], self.shape[1]))
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        self._color = self._np_random.integers(0, 256, size=(3,))
        self.bg[:, :] = self._color

    def get_info(self):
        info = super().get_info()
        info["color"] = self._color
        return info

    def get_image(self):
        return self.bg, self.mask
