from pathlib import Path
from typing import Optional, Type, Union

import gym
from gym.envs.registration import register

from .distractors import ImageSource
from .enums import DistractorLocations, ImageSourceEnum
from .merge_strategy import BaseStrategy

register(
    id="deepmind_control-v1",
    entry_point="distractor_dmc2gym.wrappers:DMCWrapper",
)


def make(
    domain_name,
    task_name,
    distraction_source: Optional[Union[str, Type[ImageSource]]] = None,
    distraction_location: Optional[Union[str, Type[BaseStrategy]]] = None,
    difficulty: Optional[str] = None,
    background_dataset_path: Optional[Path] = None,
    train_or_val: Optional[
        str
    ] = None,  # when use DAVIS Dataset, can divided it to train-set and validation-set
    visualize_reward=False,
    from_pixels=True,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    environment_kwargs=None,
    channels_first=True,
):

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    return gym.make(
        "deepmind_control-v1",
        **{
            "domain_name": domain_name,
            "task_name": task_name,
            "task_kwargs": {},
            "environment_kwargs": environment_kwargs,
            "visualize_reward": visualize_reward,
            "from_pixels": from_pixels,
            "height": height,
            "width": width,
            "camera_id": camera_id,
            "frame_skip": frame_skip,
            "channels_first": channels_first,
            "train_or_val": train_or_val,
            "ground": distraction_location,
            "distract_type": distraction_source,
            "difficulty": difficulty,
            "background_dataset_path": background_dataset_path,
        }
    )
