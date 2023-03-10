import json
import logging
import random
import tarfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import requests
import skvideo.io
import tqdm
from pytube import YouTube

from gym_distractions.errors import GymDistractionsError

from .background_source import ImageSource

DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)

TRAINING_VIDEOS = [
    "bear",
    "bmx-bumps",
    "boat",
    "boxing-fisheye",
    "breakdance-flare",
    "bus",
    "car-turn",
    "cat-girl",
    "classic-car",
    "color-run",
    "crossing",
    "dance-jump",
    "dancing",
    "disc-jockey",
    "dog-agility",
    "dog-gooses",
    "dogs-scale",
    "drift-turn",
    "drone",
    "elephant",
    "flamingo",
    "hike",
    "hockey",
    "horsejump-low",
    "kid-football",
    "kite-walk",
    "koala",
    "lady-running",
    "lindy-hop",
    "longboard",
    "lucia",
    "mallard-fly",
    "mallard-water",
    "miami-surf",
    "motocross-bumps",
    "motorbike",
    "night-race",
    "paragliding",
    "planes-water",
    "rallye",
    "rhino",
    "rollerblade",
    "schoolgirls",
    "scooter-board",
    "scooter-gray",
    "sheep",
    "skate-park",
    "snowboard",
    "soccerball",
    "stroller",
    "stunt",
    "surf",
    "swing",
    "tennis",
    "tractor-sand",
    "train",
    "tuk-tuk",
    "upside-down",
    "varanus-cage",
    "walking",
]
VALIDATION_VIDEOS = [
    "bike-packing",
    "blackswan",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "dogs-jump",
    "drift-chicane",
    "drift-straight",
    "goat",
    "gold-fish",
    "horsejump-high",
    "india",
    "judo",
    "kite-surf",
    "lab-coat",
    "libby",
    "loading",
    "mbike-trick",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "pigs",
    "scooter-black",
    "shooting",
    "soapbox",
]

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip"  # noqa: E501
KINETICS400_URL = (
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"
)


def check_empty(path: Path):
    return not path.exists() or not any(path.iterdir())


def get_img_paths(difficulty, data_path: Path, train_or_val=None):
    num_frames = DIFFICULTY_NUM_VIDEOS[difficulty]
    if train_or_val is None:
        dataset_images = sorted(data_path.iterdir())
    elif train_or_val in ["train", "training"]:
        dataset_images = TRAINING_VIDEOS
    elif train_or_val in ["val", "validation"]:
        dataset_images = VALIDATION_VIDEOS
    else:
        raise GymDistractionsError(f"train_or_val {train_or_val} not defined.")

    image_paths = [data_path / subdir.name for subdir in dataset_images]
    random.shuffle(image_paths)
    if num_frames is not None:
        if num_frames > len(image_paths) or num_frames < 0:
            raise GymDistractionsError(
                f"`num_background_paths` is {num_frames} but should "
                f"not be larger than the number of available "
                f"background paths ({len(image_paths)}) and at least 0."
            )
        image_paths = image_paths[:num_frames]

    return image_paths


class RandomVideoSource(ImageSource):
    def __init__(self, *args, data_path, train_or_val=None, intensity=1):
        super(RandomVideoSource, self).__init__(*args)
        self.intensity = intensity
        self.image_paths = get_img_paths(self.difficulty, data_path, train_or_val)
        self.num_path = len(self.image_paths)
        self.num_images = 0
        self.reset()

    def get_info(self):
        info = super().get_info()
        info["data_set"] = self.image_paths
        return info

    def build_bg_arr(self):
        pass

    def reset(self, seed=None):
        super().reset(seed)
        self.idx = 0
        self._loc = self._np_random.randint(0, self.num_path)
        self.build_bg_arr()

    def get_image(self):
        if self.idx == self.num_images:
            self.reset()

        img, mask = self.bg_arr[self.idx], self.mask_arr[self.idx]
        self.idx += 1
        return img, mask


class DAVISDataSource(RandomVideoSource):
    def __init__(self, difficulty, data_path: Path, train_or_val=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if check_empty(data_path / "DAVIS"):
            self.download_dataset(data_path)

        path = data_path / "DAVIS" / "JPEGImages" / "480p"
        self.image_paths = get_img_paths(difficulty, path, train_or_val)
        self.num_path = len(self.image_paths)

        self.reset()

    def get_info(self):
        info = {}
        info["data_set"] = "DAVIS_2017"
        return info

    def download_dataset(self, path):
        path.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading DAVIS dataset.")
        r = requests.get(DAVIS_URL)
        zipfile = ZipFile(BytesIO(r.content))
        zipfile.extractall(path)
        logging.info("Download finished.")

    @lru_cache()
    def read_in_filepath(self, file_path):
        bg_array = []
        mask_array = []
        for fpath in sorted(file_path.glob("*.jpg")):
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            img = cv2.resize(img, (self.shape[1], self.shape[0]))
            fpath = str(fpath)
            mpath = fpath.replace("JPEGImages", "Annotations_unsupervised").replace(
                "jpg", "png"
            )
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.shape[1], self.shape[0]))
            mask = np.logical_and(mask, True)
            mask_array.append(mask)
            bg_array.append(img)

        return bg_array, mask_array

    def build_bg_arr(self):
        self.image_path = self.image_paths[self._loc]
        self.bg_arr, self.mask_arr = self.read_in_filepath(self.image_path)
        self.num_images = len(self.bg_arr)


class Kinetics400DataSource(RandomVideoSource):
    def __init__(self, *args, data_path: Path, train_or_val=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grayscale = False

        if check_empty(data_path / "kinetics400"):
            self.download_dataset(data_path)

        path = data_path / "kinetics400"
        self.image_paths = self.get_img_paths(self.difficulty, path, train_or_val)
        self.num_path = len(self.image_paths)

        self.reset()

    def get_img_paths(self, difficulty, data_path: Path, train_or_val=None):
        num_frames = DIFFICULTY_NUM_VIDEOS[difficulty]
        train_or_val = train_or_val if train_or_val is not None else "train"
        if train_or_val in ["train", "training"]:
            dataset_images = (data_path / "train").glob("*.mp4")
        elif train_or_val in ["val", "validation", "test"]:
            dataset_images = (data_path / "test").glob("*.mp4")
        else:
            raise GymDistractionsError(f"train_or_val {train_or_val} not defined.")

        image_paths = list(dataset_images)
        if num_frames is not None:
            if num_frames > len(image_paths) or num_frames < 0:
                raise ValueError(
                    f"`num_background_paths` is {num_frames} but should "
                    f"not be larger than the number of available "
                    f"background paths ({len(image_paths)}) and at least 0."
                )
            image_paths = image_paths[:num_frames]

        return image_paths

    def get_info(self):
        info = {}
        info["data_set"] = "KINETICS_400"
        return info

    def download_dataset(self, path):
        path.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading Kinetics400 dataset.")
        r = requests.get(KINETICS400_URL, stream=True)
        file = tarfile.open(fileobj=r.raw, mode="r|gz")
        file.extractall(path)
        datapath = path / "kinetics400"
        train_urls = self.get_url(datapath / "train.json")
        test_urls = self.get_url(datapath / "test.json")

        self.download(train_urls, datapath / "train")
        self.download(test_urls, datapath / "test")
        logging.info("Download finished.")

    def get_url(self, path):
        with Path(path).open() as f:
            data = json.load(f)
        urls = []
        for k in data:
            if data[k]["annotations"]["label"] == "driving car":
                urls.append(data[k]["url"])
        return urls

    def download(self, urls, dest_path: Path):
        dest_path.mkdir(exist_ok=True, parents=True)
        with tqdm.trange(len(urls)) as t:
            t.set_description("Downloading video")
            for i in t:
                try:
                    url = urls[i]
                    video = YouTube(url)
                    streams = video.streams.filter(file_extension="mp4")
                    for stream in streams:
                        if stream.resolution == "360p":
                            itag = stream.itag
                            break
                    video.streams.get_by_itag(itag).download(dest_path)
                except Exception:
                    continue

    def read_in_file(self, fname, grayscale=False):
        if grayscale:
            frames = skvideo.io.vread(str(fname), outputdict={"-pix_fmt": "gray"})
        else:
            frames = skvideo.io.vread(str(fname), num_frames=1000)
        img_arr = np.zeros(
            (frames.shape[0], self.shape[0], self.shape[1])
            + ((3,) if not self.grayscale else (1,))
        )
        for i in range(frames.shape[0]):
            img_arr[i] = cv2.resize(
                frames[i], (self.shape[1], self.shape[0])
            )  # THIS IS NOT A BUG! cv2 uses (width, height)

        return img_arr

    def build_bg_arr(self):
        fname = self.image_paths[self._loc]
        img_arr = self.read_in_file(fname, grayscale=self.grayscale)
        self.num_images = len(img_arr)
        self.bg_arr = img_arr
        self.mask_arr = np.full(img_arr.shape[:-1], True)
