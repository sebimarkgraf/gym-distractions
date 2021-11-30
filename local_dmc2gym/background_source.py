import numpy as np
import cv2
import random
import tqdm
import os

DIFFICULTY_SCALE = dict(easy=0.1, medium=0.2, hard=0.3)
DIFFICULTY_NUM_VIDEOS = dict(easy=5, medium=8, hard=None)

DAVIS17_TRAINING_VIDEOS = [
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
DAVIS17_VALIDATION_VIDEOS = [
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox'
]


def get_img_paths(difficulty, date_path, train_or_val=None):
    num_frames = DIFFICULTY_NUM_VIDEOS[difficulty]
    if train_or_val is None:
        dataset_images = sorted(os.listdir(date_path))
    elif train_or_val in ['trian', 'training']:
        dataset_images = DAVIS17_TRAINING_VIDEOS
    elif train_or_val in ['val', 'validation']:
        dataset_images = DAVIS17_VALIDATION_VIDEOS
    else:
        raise Exception("train_or_val %s not defined." % train_or_val)

    image_paths = [os.path.join(date_path, subdir) for subdir in dataset_images]
    if num_frames is not None:
        if num_frames > len(image_paths) or num_frames < 0:
            raise ValueError(f'`num_bakground_paths` is {num_frames} but should not be larger than the '
                             f'number of available background paths ({len(image_paths)}) and at least 0.')
        image_paths = image_paths[:num_frames]

    return image_paths


class ImageSource(object):

    def get_image(self):
        pass

    def reset(self):
        pass


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        self.shape = shape
        self.bg = np.zeros((self.shape[0], self.shape[1], 3))
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))
        self.bg[:, :] = self._color

    def get_image(self, obs):
        self.bg = cv2.resize(self.bg, (obs.shape[1], obs.shape[0]))
        mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        obs[mask] = self.bg[mask]
        return obs


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255):
        self.strength = strength
        self.shape = shape

    def get_image(self, obs):
        self.bg = np.random.rand(obs.shape[0], obs.shape[1], 3) * self.strength
        self.bg = self.bg.astype(np.uint8)
        mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        obs[mask] = self.bg[mask]
        return obs


class RandomDotsSource(ImageSource):
    def __init__(self, shape, difficulty, ground=None, intensity=1):
        self.shape = shape
        num_dots = DIFFICULTY_NUM_VIDEOS[difficulty]
        self.num_dots = num_dots if num_dots else 16
        self.num_frames = 1000
        self.ground = ground
        self.intensity = intensity
        self.x_lim_low = 0.05
        self.x_lim_high = 0.95
        self.y_lim_low = 0.2
        self.y_lim_high = 0.8
        self.dots_size = 0.07
        self.reset()

    def reset(self):
        self.idx = 0
        self.colors = []
        self.positions = []
        self.sizes = []
        self.move = []
        for i in range(self.num_dots):
            self.colors.append(np.random.rand(3))
            self.positions.append([np.random.uniform(self.x_lim_low, self.x_lim_high), np.random.uniform(self.y_lim_low, self.y_lim_high)])
            self.sizes.append(np.random.uniform(0.7, 1))
            self.move.append([0, 0])

    def limit_pos(self, i):
        if self.positions[i][0] < self.x_lim_low:
            self.positions[i][0] = self.x_lim_low
            self.move[i][0] = 0
        elif self.positions[i][0] > self.x_lim_high:
            self.positions[i][0] = self.x_lim_high
            self.move[i][0] = 0
        if self.positions[i][1] < self.y_lim_low:
            self.positions[i][1] = self.y_lim_low
            self.move[i][1] = 0
        elif self.positions[i][1] > self.y_lim_high:
            self.positions[i][1] = self.y_lim_high
            self.move[i][1] = 0

    def build_bg(self, w, h):
        self.bg = np.zeros((h, w, 3))
        for i in range(self.num_dots):
            color, position, size = self.colors[i], self.positions[i], self.sizes[i]
            position = (int(position[0] * w), int(position[1] * h))
            cv2.circle(self.bg, position, int(size * w * self.dots_size), color, -1)
            self.move[i] = np.random.normal(self.move[i], 0.005, 2)
            self.move[i] = self.move[i] if np.random.rand() < 0.8 else self.move[i] / 5
            self.positions[i] += self.move[i]
            self.limit_pos(i)
            # self.colors[i] += np.random.normal(1 / 255, 0.005, 3)  # change color
        self.bg *= 255
        self.bg = self.bg.astype(np.uint8)

    def get_image(self, obs):
        if self.idx == self.num_frames:
            self.reset()
        h, w, _ = obs.shape
        self.build_bg(w, h)

        if self.ground == 'forground':
            mask = np.logical_or(self.bg[:, :, 0] > 0, self.bg[:, :, 1] > 0, self.bg[:, :, 2] > 0)
            # obs[mask] = self.bg[mask]
        else:
            mask1 = np.logical_or(self.bg[:, :, 0] > 0, self.bg[:, :, 1] > 0, self.bg[:, :, 2] > 0)
            mask2 = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
            mask = np.logical_and(mask1, mask2)
        obs[mask] = self.intensity * self.bg[mask] + (1 - self.intensity) * obs[mask]
        self.idx += 1
        return obs


class RandomVideoSource(ImageSource):
    def __init__(self, shape, difficulty, date_path, train_or_val=None, ground=None):
        self.ground = ground
        self.shape = shape
        self.image_paths = get_img_paths(difficulty, date_path, train_or_val)
        self.num_path = len(self.image_paths)
        self.reset()

    def build_bg_arr(self):
        self.image_path = self.image_paths[self._loc]
        self.image_files = os.listdir(self.image_path)
        self.bg_arr = []
        self.mask_arr = []
        for fname in self.image_files:
            fpath = os.path.join(self.image_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
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

    def get_image(self, obs):
        if self.idx == len(self.image_files):
            self.reset()

        if self.ground == 'forground':
        	self.bg = self.mask_arr[self.idx]
        else:
        	self.bg = self.bg_arr[self.idx]
        
        self.bg = cv2.resize(self.bg, (obs.shape[1], obs.shape[0]))

        if self.ground == 'forground':
            mask = np.logical_and(self.bg, True)
            obs[mask] = self.bg[mask]

        elif self.ground == 'background':
            mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
            obs[mask] = self.bg[mask]

        elif self.ground == 'both':
        	mask1 = cv2.resize(self.mask_arr[self.idx], (obs.shape[1], obs.shape[0]))
        	mask1 = np.logical_or(mask1[:, :, 0], mask1[:, :, 1], mask1[:, :, 2])
        	mask2 = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        	mask = np.logical_or(mask1, mask2)
        	obs[mask] = self.bg[mask]
        self.idx += 1
        return obs
