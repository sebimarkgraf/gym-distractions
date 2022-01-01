import numpy as np
import cv2
import os
import copy
import json
import random

DIFFICULTY_SCALE = dict(easy=0.1, medium=0.2, hard=0.3)
DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)
DIFFICULTY_NUM_SETS = dict(easy=1, medium=2, hard=4)
GRAVITATIONAL_CONSTANT = dict(Planet=1, Electrons=-1, IdealGas=0)

TRAINING_VIDEOS = [
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'
]
VALIDATION_VIDEOS = [
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox'
]


def compute_a(i, positions, sizes, type):
    relative_positions = positions - positions[i]
    distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
    distances[i] = 1

    force_vectors = relative_positions * GRAVITATIONAL_CONSTANT[type] * (sizes ** 2) / (distances ** 2)
    accelerations = 0.00001 * np.sum(force_vectors, axis=0)

    return accelerations


def get_img_paths(difficulty, data_path, train_or_val=None):
    num_frames = DIFFICULTY_NUM_VIDEOS[difficulty]
    if train_or_val is None:
        dataset_images = sorted(os.listdir(data_path))
    elif train_or_val in ['trian', 'training']:
        dataset_images = TRAINING_VIDEOS
    elif train_or_val in ['val', 'validation']:
        dataset_images = VALIDATION_VIDEOS
    else:
        raise Exception("train_or_val %s not defined." % train_or_val)

    image_paths = [os.path.join(data_path, subdir) for subdir in dataset_images]
    random.shuffle(image_paths)
    if num_frames is not None:
        if num_frames > len(image_paths) or num_frames < 0:
            raise ValueError(f'`num_bakground_paths` is {num_frames} but should not be larger than the '
                             f'number of available background paths ({len(image_paths)}) and at least 0.')
        image_paths = image_paths[:num_frames]

    return image_paths


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
        self.dots_size = 0.08
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
            self.dots_init['sizes'] = np.random.uniform(0.7, 1, size=(self.num_sets, self.num_dots, 1))
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

        if self.ground == 'forground':
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


class RandomVideoSource(ImageSource):
    def __init__(self, shape, difficulty, data_path, train_or_val=None, ground=None, intensity=1):
        self.ground = ground
        self.shape = shape
        self.intensity = intensity
        self.image_paths = get_img_paths(difficulty, data_path, train_or_val)
        self.num_path = len(self.image_paths)
        self.reset()

    def get_info(self):
        info = super().get_info()
        info['ground'] = self.ground
        info['data_set'] = self.image_paths
        return info

    def build_bg_arr(self):
        self.image_path = self.image_paths[self._loc]
        self.image_files = sorted(os.listdir(self.image_path))
        self.bg_arr = []
        self.mask_arr = []
        for fname in self.image_files:
            fpath = os.path.join(self.image_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            if self.ground == 'forground' or self.ground == 'both':
                mpath = fpath.replace("JPEGImages", "Annotations").replace("jpg", "png")
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                mask = np.logical_and(mask, True)
                img0 = np.zeros_like(img)
                img0[mask] = img[mask]
                self.mask_arr.append(img0)
            if not self.ground == 'forground':
                self.bg_arr.append(img)

    def reset(self):
        self.idx = 0
        self._loc = np.random.randint(0, self.num_path)
        self.build_bg_arr()

    def get_image(self, obs, action=None):
        if self.idx == len(self.image_files):
            self.reset()

        if self.ground == 'forground':
            self.bg = self.mask_arr[self.idx]
        else:
            self.bg = self.bg_arr[self.idx]

        self.bg = cv2.resize(self.bg, (obs.shape[1], obs.shape[0]))

        if self.ground == 'forground':
            mask = np.logical_and(self.bg, True)
            # obs[mask] = self.bg[mask]

        elif self.ground == 'background':
            mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
            # obs[mask] = self.bg[mask]

        elif self.ground == 'both':
            mask1 = cv2.resize(self.mask_arr[self.idx], (obs.shape[1], obs.shape[0]))
            mask1 = np.logical_or(mask1[:, :, 0], mask1[:, :, 1], mask1[:, :, 2])
            mask2 = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
            mask = np.logical_or(mask1, mask2)
            # obs[mask] = self.bg[mask]
        obs[mask] = self.intensity * self.bg[mask] + (1 - self.intensity) * obs[mask]
        if action is not None:
            self.idx += 1
        return obs
