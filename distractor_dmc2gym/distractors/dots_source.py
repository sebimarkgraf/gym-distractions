import copy

import cv2
import numpy as np

from .background_source import ImageSource

DIFFICULTY_NUM_SETS = dict(easy=1, medium=2, hard=4)
GRAVITATIONAL_CONSTANT = dict(Planet=1, Electrons=-1, IdealGas=0)


def compute_a(i, positions, sizes, type):
    relative_positions = positions - positions[i]
    distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
    distances[i] = 1

    force_vectors = (
        relative_positions
        * GRAVITATIONAL_CONSTANT[type]
        * (sizes**2)
        / (distances**2)
    )
    accelerations = 0.00001 * np.sum(force_vectors, axis=0)

    return accelerations


class RandomDotsSource(ImageSource):
    def __init__(self, shape, difficulty, dots_size=0.12):
        self.shape = shape
        num_sets = DIFFICULTY_NUM_SETS[difficulty]
        self.num_dots = 12
        self.num_sets = num_sets
        self.num_frames = 1000  # after num_frames steps reset sizes, positions, colors, velocities of dots, -1 means no reset.
        self.v = 0.5
        self.x_lim_low = 0.05
        self.x_lim_high = 0.95
        self.y_lim_low = 0.05
        self.y_lim_high = 0.95
        self.dots_size = dots_size
        self.gravity_type = "IdealGas"
        self._init()
        self.reset()

    def get_info(self):
        info = super().get_info()
        info["gravity"] = self.gravity_type
        info["num_dots"] = self.num_dots
        info["num_sets"] = self.num_sets
        info["set_frames"] = self.num_frames
        info["size"] = self.dots_size
        info["velocity"] = self.v
        info["position_limit"] = {
            "x": (self.x_lim_low, self.x_lim_high),
            "y": (self.y_lim_low, self.y_lim_high),
        }
        # info['dots'] = {a: self.dots_init[a].tolist() for a in self.dots_init}
        return info

    def _init(self):
        self.dots_init = {
            "colors": np.random.rand(self.num_sets, self.num_dots, 3),
            "positions": np.concatenate(
                [
                    np.random.uniform(
                        self.x_lim_low,
                        self.x_lim_high,
                        size=(self.num_sets, self.num_dots, 1),
                    ),
                    np.random.uniform(
                        self.y_lim_low,
                        self.y_lim_high,
                        size=(self.num_sets, self.num_dots, 1),
                    ),
                ],
                axis=2,
            ),
            "sizes": np.random.uniform(
                0.8, 1.2, size=(self.num_sets, self.num_dots, 1)
            ),
            "velocities": (
                np.random.normal(0, 0.01, size=(self.num_sets, self.num_dots, 2))
                * self.v
            ),
        }

    def reset(self):
        self.idx = 0
        self.set_idx = np.random.randint(0, self.num_sets)

        dots_init = self.dots_init
        self.colors, self.positions, self.sizes, self.velocities = (
            dots_init["colors"][self.set_idx],
            dots_init["positions"][self.set_idx],
            dots_init["sizes"][self.set_idx],
            dots_init["velocities"][self.set_idx],
        )

    def limit_pos(self, i):
        if not self.x_lim_high >= self.positions[i][0] >= self.x_lim_low:
            self.velocities[i][0] = -self.velocities[i][0]
        if not self.y_lim_high >= self.positions[i][1] >= self.y_lim_low:
            self.velocities[i][1] = -self.velocities[i][1]

    def build_bg(self, w, h):
        bg = np.zeros((h, w, 3))
        for i in range(self.num_dots):
            color, position, size, move = (
                self.colors[i],
                self.positions[i],
                self.sizes[i],
                self.velocities[i],
            )
            position = (int(position[0] * w), int(position[1] * h))
            cv2.circle(bg, position, int(size * w * self.dots_size), color, -1)
            a = compute_a(
                i, np.array(self.positions), np.array(self.sizes), self.gravity_type
            )
            self.velocities[i] += a
            self.positions[i] += move
            self.limit_pos(i)
            # self.colors[i] += np.random.normal(1 / 255, 0.005, 3)  # change color
        bg *= 255
        return bg.astype(np.uint8)

    def get_image(self):
        if self.idx == self.num_frames:
            self.reset()

        h, w = self.shape
        img = self.build_bg(w, h)
        mask = np.logical_or(img[:, :, 0] > 0, img[:, :, 1] > 0, img[:, :, 2] > 0)
        return img, mask
