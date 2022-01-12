from typing import Optional

import gym
from gym.envs.registration import register

from .enums import ImageSourceEnum, DistractorLocations


register(
    id="deepmind_control-v1",
    entry_point="distractor_dmc2gym.wrappers:DMCWrapper",
)


def make(
        domain_name,
        task_name,
        distract_type: Optional[ImageSourceEnum] = None,
        ground: Optional[DistractorLocations] = None,
        difficulty=None,
        intensity=1,
        background_dataset_path=None,
        train_or_val=None,  # when use DAVIS Dataset, can divided it to train-set and validation-set
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        time_limit=10,
        channels_first=True,
        *args,
        **kwargs
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
            "task_kwargs": {"random": seed},
            "environment_kwargs": environment_kwargs,
            "visualize_reward": visualize_reward,
            "from_pixels": from_pixels,
            "height": height,
            "width": width,
            "camera_id": camera_id,
            "frame_skip": frame_skip,
            "channels_first": channels_first,
            "train_or_val": train_or_val,
            "ground": ground,
            "distract_type": distract_type,
            "difficulty": difficulty,
            "intensity": intensity,
            "background_dataset_path": background_dataset_path
        }
    )
