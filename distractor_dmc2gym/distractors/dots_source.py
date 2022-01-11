import copy

import cv2
import numpy as np

from .background_source import ImageSource
from .. import DistractorLocations

DIFFICULTY_NUM_SETS = dict(easy=1, medium=2, hard=4)
GRAVITATIONAL_CONSTANT = dict(Planet=1, Electrons=-1, IdealGas=0)



def compute_a(i, positions, sizes, type):
    relative_positions = positions - positions[i]
    distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
    distances[i] = 1

    force_vectors = relative_positions * GRAVITATIONAL_CONSTANT[type] * (sizes ** 2) / (distances ** 2)
    accelerations = 0.00001 * np.sum(force_vectors, axis=0)

    return accelerations




class RandomDotsSource(ImageSource):
    def __init__(self, shape, difficulty, ground=None, intensity=1):
        self.shape = shape
        num_sets = DIFFICULTY_NUM_SETS[difficulty]
        self.num_dots = 16
        self.num_sets = num_sets
        self.num_frames = 1000  # after num_frames steps reset sizes, positions, colors, velocities of dots, -1 means no reset.
        self.ground = ground
        self.intensity = intensity
        self.v = 0.5
        self.x_lim_low = 0.05
        self.x_lim_high = 0.95
        self.y_lim_low = 0.05
        self.y_lim_high = 0.95
        self.dots_size = 5
        self.gravity_type = 'IdealGas'
        self.reset()

    def get_info(self):
        info = super().get_info()
        info['ground'] = self.ground
        info['gravity'] = self.gravity_type
        info['num_dots'] = self.num_dots
        info['num_sets'] = self.num_sets
        info['set_frames'] = self.num_frames
        info['size'] = self.dots_size
        info['velocity'] = self.v
        info['position_limit'] = {'x': (self.x_lim_low, self.x_lim_high), 'y':(self.y_lim_low, self.y_lim_high)}
        # info['dots'] = {a: self.dots_init[a].tolist() for a in self.dots_init}
        return info

    def reset(self, new=True):
        self.idx = 0
        self.set_idx = np.random.randint(0, self.num_sets)
        if new:
            self.dots_init = {}
            self.dots_init['colors'] = np.random.rand(self.num_sets, self.num_dots, 3)
            self.dots_init['positions'] = np.concatenate(
                [np.random.uniform(self.x_lim_low, self.x_lim_high, size=(self.num_sets, self.num_dots, 1)),
                 np.random.uniform(self.y_lim_low, self.y_lim_high, size=(self.num_sets, self.num_dots, 1))], axis=2)
            self.dots_init['sizes'] = np.random.uniform(0.8, 1.2, size=(self.num_sets, self.num_dots, 1))
            self.dots_init['velocities'] = np.random.normal(0, 0.01, size=(self.num_sets, self.num_dots, 2)) * self.v
        dots_init = copy.deepcopy(self.dots_init)
        self.colors, self.positions, self.sizes, self.velocities = dots_init['colors'][self.set_idx], \
                                                                   dots_init['positions'][self.set_idx], \
                                                                   dots_init['sizes'][self.set_idx], \
                                                                   dots_init['velocities'][self.set_idx]

    def limit_pos(self, i):
        if not self.x_lim_high >= self.positions[i][0] >= self.x_lim_low:
            self.velocities[i][0] = -self.velocities[i][0]
        if not self.y_lim_high >= self.positions[i][1] >= self.y_lim_low:
            self.velocities[i][1] = -self.velocities[i][1]

    def build_bg(self, w, h, action=None):
        self.bg = np.zeros((h, w, 3))
        for i in range(self.num_dots):
            color, position, size, move = self.colors[i], self.positions[i], self.sizes[i], self.velocities[i]
            position = (int(position[0] * w), int(position[1] * h))
            cv2.circle(self.bg, position, int(size * w * self.dots_size), color, -1)
            if action is not None:
                # a = np.random.normal(0, 0.01, 2) * 0.01 if np.random.uniform() < 0.1 else 0
                a = compute_a(i, np.array(self.positions), np.array(self.sizes), self.gravity_type)
                self.velocities[i] += a
                self.positions[i] += move
                self.limit_pos(i)
                # self.colors[i] += np.random.normal(1 / 255, 0.005, 3)  # change color
        self.bg *= 255
        self.bg = self.bg.astype(np.uint8)

    def get_image(self, obs, action=None):
        if self.idx == self.num_frames:
            self.reset(
                new=False)  # if new=True, will random reset dots, else will reset dots the same as the first time(distractors repeated).
        h, w, _ = obs.shape
        self.build_bg(w, h, action)

        if self.ground == DistractorLocations.FOREGROUND:
            mask = np.logical_or(self.bg[:, :, 0] > 0, self.bg[:, :, 1] > 0, self.bg[:, :, 2] > 0)
            # obs[mask] = self.bg[mask]
        else:
            mask1 = np.logical_or(self.bg[:, :, 0] > 0, self.bg[:, :, 1] > 0, self.bg[:, :, 2] > 0)
            mask2 = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
            mask = np.logical_and(mask1, mask2)
        obs[mask] = self.intensity * self.bg[mask] + (1 - self.intensity) * obs[mask]
        if action is not None:
            self.idx += 1
        return obs
