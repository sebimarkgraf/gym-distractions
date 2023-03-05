import pytest

from gym_distractions.distractors import ImageSource, NoiseSource, RandomColorSource


@pytest.mark.parametrize("source_type", [NoiseSource, RandomColorSource])
def test_static_background_source(source_type):
    source: ImageSource = source_type(shape2d=(2, 2))

    source.reset()

    image, mask = source.get_image()

    assert image is not None
    assert image.shape[:2] == (2, 2)
    assert mask.dtype == bool, "Mask should have type bool"
    assert mask.shape[:2] == (2, 2)
