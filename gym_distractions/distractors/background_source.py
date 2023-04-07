from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
from gymnasium.utils.seeding import np_random


class ImageSource(object, metaclass=ABCMeta):
    def __init__(
        self,
        shape2d,
        difficulty="hard",
        background_dataset_path: Path = Path("~/background_datasets"),
    ):
        self._np_random, seed = np_random()
        self.shape = shape2d
        self.difficulty = difficulty
        self.background_dataset_path = background_dataset_path

    @abstractmethod
    def get_image(self) -> Tuple[np.array, np.array]:
        pass

    def reset(self, seed=None):
        if seed is not None:
            self._np_random, seed = np_random(seed)
