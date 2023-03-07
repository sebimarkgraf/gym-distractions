from __future__ import annotations

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

    def merge_timeseries(self, obs: np.array) -> np.array:
        """Merge timeseries of observations.

        Used for offline adding of observations to datasets.

        :param obs:
        :return:
        """
        self.source.reset()
        augmented_obs = []
        for timestep in obs:
            augmented_obs.append(self.merge(timestep))
        return np.array(augmented_obs)

    @abstractmethod
    def get_last_mask(self):
        pass


class FrontMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array) -> np.array:
        img, mask = self.source.get_image()
        augmented_obs = np.copy(obs)
        augmented_obs[mask] = img[mask]
        self._mask = mask
        return augmented_obs

    def get_last_mask(self):
        return self._mask


class BackgroundMerge(BaseStrategy):
    _mask: np.array = None

    def merge(self, obs: np.array) -> np.array:
        img, mask = self.source.get_image()
        dmc_background_mask = np.logical_and(
            (obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0])
        )
        combined_mask = np.logical_and(mask, dmc_background_mask)
        augmented_obs = np.copy(obs)
        augmented_obs[combined_mask] = img[combined_mask]
        self._mask = dmc_background_mask
        return augmented_obs

    def get_last_mask(self):
        return self._mask


class FrontAndBackMerge(BaseStrategy):
    def merge(self, obs: np.array) -> np.array:
        img, mask = self.source.get_image()
        dmc_background_mask = np.logical_and(
            (obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0])
        )
        backgroudn_mask = np.logical_and(~mask, dmc_background_mask)
        augmented_obs = np.copy(obs)
        # background
        augmented_obs[backgroudn_mask] = img[backgroudn_mask]
        # foreground
        augmented_obs[mask] = img[mask]

        self._mask = np.logical_or(backgroudn_mask, mask)
        return augmented_obs

    def get_last_mask(self):
        return self._mask
