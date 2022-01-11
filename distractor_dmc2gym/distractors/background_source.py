import numpy as np
import cv2
import os
import copy
import json

from distractor_dmc2gym import DistractorLocations

DIFFICULTY_SCALE = dict(easy=0.1, medium=0.2, hard=0.3)


class ImageSource(object):
    def __init__(self, intensity=1):
        self.intensity = intensity

    def get_image(self):
        pass

    def reset(self):
        pass

    def get_info(self):
        info = {}
        info['intensity'] = self.intensity
        return info

    def save_info(self, path):
        info = {}
        info[self.__class__.__name__] = self.get_info()
        with open(os.path.join(path, 'distractors_info.json'),"w") as f:
            json.dump(info, f, indent=4)


class RandomColorSource(ImageSource):
    def __init__(self, shape, intensity=1):
        self.shape = shape
        self.intensity = intensity
        self.bg = np.zeros((self.shape[0], self.shape[1], 3))
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))
        self.bg[:, :] = self._color

    def get_info(self):
        info = super().get_info()
        info['color'] = self._color
        return info

    def get_image(self, obs, action=None):
        self.bg = cv2.resize(self.bg, (obs.shape[1], obs.shape[0]))
        mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        obs[mask] = self.intensity * self.bg[mask] + (1 - self.intensity) * obs[mask]
        return obs


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255, intensity=1):
        self.strength = strength
        self.shape = shape
        self.intensity = intensity

    def get_info(self):
        info = super().get_info()
        info['strength'] = self.strength
        return info

    def get_image(self, obs, action=None):
        self.bg = np.random.rand(obs.shape[0], obs.shape[1], 3) * self.strength
        self.bg = self.bg.astype(np.uint8)
        mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        obs[mask] = self.intensity * self.bg[mask] + (1 - self.intensity) * obs[mask]
        return obs


