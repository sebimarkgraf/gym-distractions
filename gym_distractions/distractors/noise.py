import numpy as np

from .background_source import ImageSource


class NoiseSource(ImageSource):
    def __init__(self, *args, strength=255, intensity=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.strength = strength
        self.intensity = intensity

    def get_info(self):
        info = super().get_info()
        info["strength"] = self.strength
        return info

    def get_image(self):
        h, w = self.shape
        img = self._np_random.random((h, w, 3)) * self.strength
        img = img.astype(np.uint8)
        return img, np.ones((h, w), dtype=bool)
