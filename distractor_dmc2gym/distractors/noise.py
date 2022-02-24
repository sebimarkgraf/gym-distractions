import numpy as np

from distractor_dmc2gym.distractors import ImageSource


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255, intensity=1):
        self.strength = strength
        self.shape = shape
        self.intensity = intensity

    def get_info(self):
        info = super().get_info()
        info["strength"] = self.strength
        return info

    def get_image(self):
        w, h = self.shape
        img = np.random.rand(w, h, 3) * self.strength
        img = img.astype(np.uint8)
        return img, np.ones((w, h))
