import logging
from pathlib import Path
from typing import Optional, Type, Union

import gymnasium as gym
import numpy as np

from .distractors import ImageSource
from .merge_strategy import BaseStrategy
from .type_mappings import (
    ImageSourceEnum,
    MergeStrategies,
    map_distract_type_to_distractor,
    map_strategy_config,
)

logger = logging.getLogger(__name__)


class DistractionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        distract_type: Optional[Union[Type[ImageSource], ImageSourceEnum]] = None,
        merge_strategy: Optional[Union[MergeStrategies, Type[BaseStrategy]]] = None,
        difficulty: str = "easy",
        background_dataset_path: Path = Path("~/background-datasets"),
    ):
        super().__init__(env=env)
        shape2d = (
            self.env.observation_space.shape[0],
            self.env.observation_space.shape[1],
        )

        if distract_type is not None:
            self._bg_source = map_distract_type_to_distractor(distract_type)(
                shape2d=shape2d,
                difficulty=difficulty,
                background_dataset_path=background_dataset_path,
            )

            self.merger = map_strategy_config(merge_strategy)(self._bg_source)

        else:
            logger.info(
                "Distract type is None. Did not create a distraction. "
                "Wrapper is deactivated."
            )

    @property
    def deactivated(self):
        return self._bg_source is None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if self.deactivated:
            return obs, info

        self._distraction_source.reset(seed=seed)
        obs = self.merger.merge(obs=obs)
        info = {**info, "mask": self.merger.get_last_mask()}
        return obs, info

    def step(self, action: np.ndarray):
        obs, *others, info = self.env.step(action)

        if self.deactivated:
            return (obs, *others, info)

        obs = self.merger.merge(obs=obs)
        info = {**info, "mask": self.merger.get_last_mask()}

        return (obs, *others, info)
