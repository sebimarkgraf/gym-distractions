from importlib.metadata import version
from pathlib import Path
from typing import Optional, Type, Union

import gymnasium as gym
from gymnasium.envs.registration import register

from .distractors import ImageSource
from .merge_strategy import BaseStrategy

__version__ = version(__package__ or __name__)

register(
    id="deepmind_control-v1",
    entry_point="gym_distractions.wrappers:DMCWrapper",
)


def make(
    domain_name,
    task_name,
    distraction_source: Optional[Union[str, Type[ImageSource]]] = None,
    distraction_location: Optional[Union[str, Type[BaseStrategy]]] = None,
    difficulty: Optional[str] = None,
    background_dataset_path: Optional[Path] = None,
    from_pixels=True,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    environment_kwargs=None,
    channels_first=True,
):

    return gym.make(
        "deepmind_control-v1",
        **{
            "domain_name": domain_name,
            "task_name": task_name,
            "task_kwargs": {},
            "environment_kwargs": environment_kwargs,
            "from_pixels": from_pixels,
            "height": height,
            "width": width,
            "camera_id": camera_id,
            "frame_skip": frame_skip,
            "channels_first": channels_first,
            "ground": distraction_location,
            "distract_type": distraction_source,
            "difficulty": difficulty,
            "background_dataset_path": background_dataset_path,
        }
    )
