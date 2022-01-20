from abc import ABCMeta, abstractmethod

import numpy as np

from ..distractors import ImageSource


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, source: ImageSource, intensity=1):
        self.source = source
        self.intensity = intensity

    @abstractmethod
    def merge(self, obs: np.array) -> np.array:
        pass

    @abstractmethod
    def get_last_mask(self):
        pass





class FrontMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array) -> np.array:
        img, mask = self.source.get_image()
        obs[mask] = img[mask]
        self._mask = mask
        return obs

    def get_last_mask(self):
        return self._mask


class BackgroundMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array) -> np.array:
        img, mask = self.source.get_image()
        dmc_background_mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        combined_mask = np.logical_and(mask, dmc_background_mask)
        obs[combined_mask] = img[combined_mask]
        self._mask = combined_mask
        return obs

    def get_last_mask(self):
        return self._mask


strategies = {
    'foreground': FrontMerge,
    'background': BackgroundMerge
}
