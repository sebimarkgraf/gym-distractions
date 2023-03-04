from abc import ABCMeta, abstractmethod
from typing import NamedTuple, TypeVar

try:
    from typing import Protocol
except ImportError:
    from abc import ABC

    Protocol = ABC

import cv2
import numpy as np

from ..background_source import ImageSource

DIFFICULTY_NUM_SETS = dict(easy=1, medium=2, hard=4)

T = TypeVar("T", bound=dict)


class Limits(NamedTuple):
    low: float
    high: float


class DotsBehaviour(Protocol):
    @abstractmethod
    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> T:
        pass

    @abstractmethod
    def update_state(self, state):
        pass

    @abstractmethod
    def get_positions(self, state) -> np.array:
        pass


class DotsSource(ImageSource, metaclass=ABCMeta):
    def __init__(self, *args, dots_size=0.12, dots_behaviour: DotsBehaviour, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dots = 12
        self.dots_size = dots_size
        self.x_lim = Limits(0.05, 0.95)
        self.y_lim = Limits(0.05, 0.95)

        self.dots_behaviour = dots_behaviour
        self.dots_state = self.dots_behaviour.init_state(
            self.num_dots, self.x_lim, self.y_lim, self._np_random
        )
        self.positions = self.dots_behaviour.get_positions(self.dots_state)
        self.dots_parameters = self.init_dots()

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
            "sizes": self._np_random.uniform(0.8, 1.2, size=(self.num_dots, 1)),
        }

    def reset(self, seed=None):
        super().reset(seed)
        self.dots_parameters = self.init_dots()
        self.dots_state = self.dots_behaviour.init_state(
            self.num_dots, self.x_lim, self.y_lim, self._np_random
        )

    def build_bg(self, w, h):
        bg = np.zeros((h, w, 3))
        positions = self.dots_behaviour.get_positions(self.dots_state) * [[w, h]]
        sizes = self.dots_parameters["sizes"]
        colors = self.dots_parameters["colors"]
        for position, size, color in zip(positions, sizes, colors):
            cv2.circle(
                bg,
                (int(position[0]), int(position[1])),
                int(size * w * self.dots_size),
                color,
                -1,
            )

        self.dots_state = self.dots_behaviour.update_state(self.dots_state)
        bg *= 255
        return bg.astype(np.uint8)

    def get_image(self):
        h, w = self.shape
        img = self.build_bg(w, h)
        mask = np.logical_or(img[:, :, 0] > 0, img[:, :, 1] > 0, img[:, :, 2] > 0)
        return img, mask
