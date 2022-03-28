import numpy as np

from distractor_dmc2gym.distractors import RandomDotsSource
from distractor_dmc2gym.merge_strategy import FrontMerge


def test_merge_timeseries():
    distractor = RandomDotsSource(shape=(64, 64), difficulty="easy")
    strategy = FrontMerge(source=distractor)

    T = 50
    image_shape = (64, 64, 3)
    observations = np.random.randn(T, *image_shape)
    augmented_obs = strategy.merge_timeseries(observations)

    assert augmented_obs is not None
    assert augmented_obs.shape == (T, *image_shape)
