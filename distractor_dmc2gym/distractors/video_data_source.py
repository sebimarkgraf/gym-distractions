import json
import logging
import os
import random
from io import BytesIO

import cv2
import numpy as np
import requests
from pathlib import Path
from zipfile import ZipFile

from .background_source import ImageSource




DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)

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

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip"


def check_empty(path: Path):
    return not path.exists() or any(path.iterdir())

def download_dataset(output_path: Path):
    r = requests.get(DAVIS_URL)
    zipfile = ZipFile(BytesIO(r.content))
    zipfile.extractall(output_path)


def get_img_paths(difficulty, data_path: Path, train_or_val=None):
    num_frames = DIFFICULTY_NUM_VIDEOS[difficulty]
    if train_or_val is None:
        dataset_images = sorted(data_path.iterdir())
    elif train_or_val in ['train', 'training']:
        dataset_images = TRAINING_VIDEOS
    elif train_or_val in ['val', 'validation']:
        dataset_images = VALIDATION_VIDEOS
    else:
        raise Exception(f"train_or_val {train_or_val} not defined.")

    image_paths = [data_path / subdir.name for subdir in dataset_images]
    random.shuffle(image_paths)
    if num_frames is not None:
        if num_frames > len(image_paths) or num_frames < 0:
            raise ValueError(f'`num_background_paths` is {num_frames} but should not be larger than the '
                             f'number of available background paths ({len(image_paths)}) and at least 0.')
        image_paths = image_paths[:num_frames]

    return image_paths


class RandomVideoSource(ImageSource):
    def __init__(self, shape, difficulty, data_path, train_or_val=None, ground=None, intensity=1):
        self.ground = ground
        self.shape = shape
        self.intensity = intensity
        self.image_paths = get_img_paths(difficulty, data_path, train_or_val)
        self.num_path = len(self.image_paths)
        self.num_images = 0
        self.reset()

    def get_info(self):
        info = super().get_info()
        info['ground'] = self.ground
        info['data_set'] = self.image_paths
        return info

    def build_bg_arr(self):
        self.image_path = self.image_paths[self._loc]
        self.bg_arr = []
        self.mask_arr = []
        for fpath in self.image_path.glob('*.jpg'):
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            img = cv2.resize(img, (self.shape[1], self.shape[0]))
            fpath = str(fpath)
            mpath = fpath.replace("JPEGImages", "Annotations_unsupervised").replace("jpg", "png")
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.shape[1], self.shape[0]))
            mask = np.logical_and(mask, True)
            self.mask_arr.append(mask)
            self.bg_arr.append(img)

        self.num_images = len(self.bg_arr)

    def reset(self):
        self.idx = 0
        self._loc = np.random.randint(0, self.num_path)
        self.build_bg_arr()

    def get_image(self):
        if self.idx == self.num_images:
            self.reset()

        img, mask = self.bg_arr[self.idx], self.mask_arr[self.idx]
        self.idx += 1
        return img, mask


class DAVISDataSource(RandomVideoSource):
    def __init__(self, shape, difficulty, data_path: Path, train_or_val=None, ground=None, intensity=1):
        self.ground = ground
        self.shape = shape
        self.intensity = intensity

        if check_empty(data_path):
            self.download_dataset(data_path)

        path = data_path / "DAVIS" / "JPEGImages" / "480p"
        self.image_paths = get_img_paths(difficulty, path, train_or_val)
        self.num_path = len(self.image_paths)

        self.reset()

    def get_info(self):
        info = {}
        info['ground'] = self.ground
        info['data_set'] = "DAVIS_2017"
        return info

    def download_dataset(self, path):
        path.mkdir(exist_ok=True)
        logging.info("Downloading DAVIS dataset.")
        download_dataset(path)
        logging.info("Download finished.")
