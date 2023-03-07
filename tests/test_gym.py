import pytest
from gymnasium.utils.env_checker import check_env

from gym_distractions import make


@pytest.mark.parametrize("distraction", [None, "dots", "noise", "color"])
def test_conformity_non_distracted(distraction):
    env = make(
        "cheetah",
        "run",
        distraction_source=distraction,
        distraction_location="background",
        from_pixels=True,
        height=128,
        width=256,
        channels_first=False,
    )
    check_env(env)
