import numpy as np
import pytest

from gym_distractions.distractors import NoiseSource
from gym_distractions.merge_strategy import (
    BackgroundMerge,
    FrontAndBackMerge,
    FrontMerge,
)


@pytest.mark.parametrize("strategy", [FrontMerge, BackgroundMerge, FrontAndBackMerge])
def test_merge_timeseries(strategy):
    distractor = NoiseSource(
        shape2d=(64, 64),
    )
    strategy = strategy(source=distractor)

    T = 50
    image_shape = (64, 64, 3)
    observations = np.random.default_rng().random(size=(T, *image_shape))
    original_obs = observations.copy()
    augmented_obs = strategy.merge_timeseries(observations)

    assert augmented_obs is not None
    assert augmented_obs.shape == (T, *image_shape)
    assert np.array_equal(
        original_obs, observations
    ), "Should not write on given observations"


@pytest.mark.parametrize("strategy", [FrontMerge, BackgroundMerge, FrontAndBackMerge])
def test_merge(strategy):
    distractor = NoiseSource(shape2d=(64, 64))
    strategy = strategy(source=distractor)

    image_shape = (64, 64, 3)
    observations = np.random.default_rng().random(size=image_shape)
    original_obs = observations.copy()
    augmented_obs = strategy.merge(observations)

    assert augmented_obs is not None
    assert augmented_obs.shape == image_shape
    assert np.array_equal(
        original_obs, observations
    ), "Should not write on given observations"
