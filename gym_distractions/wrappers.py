from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium import core, spaces

from . import BaseStrategy, ImageSource
from .type_mappings import (
    ImageSourceEnum,
    MergeStrategies,
    map_distract_type_to_distractor,
    map_strategy_config,
)


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int_(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        distract_type: Optional[Union[Type[ImageSource], ImageSourceEnum]] = None,
        ground: Optional[Union[MergeStrategies, Type[BaseStrategy]]] = None,
        difficulty: str = "easy",
        background_dataset_path: Optional[Path] = None,
        task_kwargs: Optional[dict] = None,
        from_pixels: bool = False,
        height: int = 84,
        width: int = 84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first: bool = True,
        render_mode: str = "rgb_array",
    ):
        self._render_mode = render_mode
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        self._env_args = dict(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
        )
        self._env = suite.load(**self._env_args)

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_space = _spec_to_box(self._env.observation_spec().values())

        self.current_state = None

        # background/foreground
        self._bg_source = None
        if distract_type is not None:
            shape2d = (height, width)
            self._bg_source = map_distract_type_to_distractor(distract_type)(
                shape2d=shape2d,
                difficulty=difficulty,
                background_dataset_path=background_dataset_path,
            )

            self.merger = map_strategy_config(ground)(self._bg_source)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step, action=None):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id,
                action=action,
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed=None):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def _getinfo(self, time_step):
        extra = {"internal_state": self._env.physics.get_state().copy()}
        extra["discount"] = time_step.discount
        if self._bg_source is not None and self.merger is not None:
            extra["mask"] = ~self.merger.get_last_mask()

        return extra

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step, action)
        self.current_state = _flatten_obs(time_step.observation)
        extra = self._getinfo(time_step)

        return obs, reward, done, False, extra

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env_args["task_kwargs"]["random"] = seed
            self._env = suite.load(**self._env_args)
            self.seed(seed)
        if self._bg_source:
            self._bg_source.reset(seed=seed)
        time_step = self._env.reset()

        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)

        info = self._getinfo(time_step)
        return obs, info

    def render(self, height=None, width=None, camera_id=0, action=None):
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        obs = self._env.physics.render(height=height, width=width, camera_id=camera_id)
        if self._bg_source:
            obs = self.merger.merge(obs)
        return obs
