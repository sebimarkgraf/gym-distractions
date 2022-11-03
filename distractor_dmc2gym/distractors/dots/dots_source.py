import copy
from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum
from typing import NamedTuple

import cv2
import numpy as np

from ..background_source import ImageSource

DIFFICULTY_NUM_SETS = dict(easy=1, medium=2, hard=4)


class Limits(NamedTuple):
    low: float
    high: float


class GeneralDotsSource(ImageSource, metaclass=ABCMeta):
    def __init__(self, *args, dots_size=0.12, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dots = 12
        self.dots_size = dots_size
        self.x_lim = Limits(0.05, 0.95)
        self.y_lim = Limits(0.05, 0.95)
        self.dots_init = self.init_dots()
        self.reset()

    def get_info(self):
        info = super().get_info()
        return {
            **info,
            "num_dots": self.num_dots,
            "size": self.dots_size,
        }

    def init_dots(self) -> dict:
        return {
            "colors": self._np_random.random((self.num_dots, 3)),
            "positions": np.concatenate(
                [
                    self._np_random.uniform(*self.x_lim, size=(self.num_dots, 1)),
                    self._np_random.uniform(*self.y_lim, size=(self.num_dots, 1)),
                ],
                axis=-1,
            ),
            "sizes": self._np_random.uniform(0.8, 1.2, size=(self.num_dots, 1)),
        }

    @abstractmethod
    def update_positions(self):
        pass

    def reset_dots(self):
        self.dots_init = self.init_dots()
        self.colors, self.positions, self.sizes = (
            self.dots_init["colors"].copy(),
            self.dots_init["positions"].copy(),
            self.dots_init["sizes"].copy(),
        )

    def reset(self, seed=None):
        super().reset(seed)
        self.reset_dots()

    def build_bg(self, w, h):
        bg = np.zeros((h, w, 3))
        positions = self.positions * [[w, h]]
        for position, size, color in zip(positions, self.sizes, self.colors):
            cv2.circle(
                bg,
                (int(position[0]), int(position[1])),
                int(size * w * self.dots_size),
                color,
                -1,
            )

        self.update_positions()
        bg *= 255
        return bg.astype(np.uint8)

    def get_image(self):
        h, w = self.shape
        img = self.build_bg(w, h)
        mask = np.logical_or(img[:, :, 0] > 0, img[:, :, 1] > 0, img[:, :, 2] > 0)
        return img, mask
