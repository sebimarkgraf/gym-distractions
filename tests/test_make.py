import pytest
from gym.wrappers import TimeLimit

from distractor_dmc2gym import make


def test_make():
    make(
        "cheetah", "run", None, "color", from_pixels=True, visualize_reward=False,
    )


def test_image_size():
    env = make(
        "cheetah",
        "run",
        distraction_source=None,
        from_pixels=True,
        visualize_reward=False,
        height=128,
        width=256,
    )
    assert env.observation_space.shape[0] == 128
    assert env.observation_space.shape[1] == 256


@pytest.mark.parametrize("action_repeat", [1, 2, 4])
def test_action_repeat(action_repeat):
    env = make(
        "cheetah",
        "run",
        distraction_source=None,
        from_pixels=True,
        visualize_reward=False,
        height=128,
        width=256,
        frame_skip=action_repeat,
    )
    env = env
    done = False
    obs = env.reset()
    steps = 0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, _ = env.step(a)
        steps += 1

    assert steps == 1000 / action_repeat


@pytest.mark.parametrize("action_repeat", [2, 4])
@pytest.mark.parametrize("distraction_location", ["foreground", "background"])
def test_higher_action_repeat_distracted(action_repeat, distraction_location):
    env = make(
        "cheetah",
        "run",
        distraction_location=distraction_location,
        distraction_source="dots",
        from_pixels=True,
        visualize_reward=False,
        height=128,
        width=256,
        frame_skip=action_repeat,
    )
    env = env
    obs = env.reset()

    a = env.action_space.sample()
    obs, reward, done, _ = env.step(a)

    assert obs is not None
