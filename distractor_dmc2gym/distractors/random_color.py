import cv2
import numpy as np

from distractor_dmc2gym.distractors import ImageSource


class RandomColorSource(ImageSource):
    def __init__(self, shape, intensity=1):
        self.shape = shape
        self.intensity = intensity
        self.bg = np.zeros((self.shape[0], self.shape[1], 3))
        self.mask = np.ones((self.shape[0], self.shape[1]))
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))
        self.bg[:, :] = self._color

    def get_info(self):
        info = super().get_info()
        info["color"] = self._color
        return info

    def get_image(self):
        return self.bg, self.mask
