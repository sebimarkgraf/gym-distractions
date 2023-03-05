import json
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
from gym.utils.seeding import np_random


class ImageSource(object, metaclass=ABCMeta):
    def __init__(self, shape2d, difficulty="hard"):
        self._np_random, seed = np_random()
        self.shape = shape2d
        self.difficulty = difficulty

    @abstractmethod
    def get_image(self) -> Tuple[np.array, np.array]:
        pass

    def reset(self, seed=None):
        if seed is not None:
            self._np_random, seed = np_random(seed)

    def get_info(self):
        info = {}
        return info

    def save_info(self, path: Path):
        info = {self.__class__.__name__: self.get_info()}
        with (path / "distractors_info.json").open("w") as f:
            json.dump(info, f, indent=4)
