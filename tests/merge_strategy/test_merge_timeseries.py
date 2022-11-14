import numpy as np
import pytest

from distractor_dmc2gym.distractors import RandomDotsSource
from distractor_dmc2gym.distractors.dots.dots_source import DotsSource
from distractor_dmc2gym.merge_strategy import BackgroundMerge, FrontMerge


@pytest.mark.parametrize("strategy", [FrontMerge, BackgroundMerge])
def test_merge_timeseries(strategy):
    distractor = DotsSource(
        shape2d=(64, 64), difficulty="easy", dots_behaviour=RandomDotsSource()
    )
    strategy = strategy(source=distractor)

    T = 50
    image_shape = (64, 64, 3)
    observations = np.random.randn(T, *image_shape)
    original_obs = observations.copy()
    augmented_obs = strategy.merge_timeseries(observations)

    assert augmented_obs is not None
    assert augmented_obs.shape == (T, *image_shape)
    assert np.array_equal(
        original_obs, observations
    ), "Should not write on given observations"
