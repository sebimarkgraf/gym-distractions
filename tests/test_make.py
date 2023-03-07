import pytest

from gym_distractions import make


def test_make():
    make(
        "cheetah",
        "run",
        None,
        "color",
        from_pixels=True,
    )


def test_image_size():
    env = make(
        "cheetah",
        "run",
        distraction_source=None,
        from_pixels=True,
        height=128,
        width=256,
        channels_first=False,
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
        height=128,
        width=256,
        frame_skip=action_repeat,
    )
    env = env
    done = False
    env.reset()
    steps = 0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(a)
        steps += 1

    assert done
    assert steps == 1000 / action_repeat


@pytest.mark.parametrize("action_repeat", [2, 4])
@pytest.mark.parametrize("distraction_location", ["foreground", "background"])
def test_higher_action_repeat_distracted(action_repeat, distraction_location):
    env = make(
        "cheetah",
        "run",
        distraction_location=distraction_location,
        distraction_source="dots_linear",
        from_pixels=True,
        height=128,
        width=256,
        frame_skip=action_repeat,
    )
    env = env
    env.reset()

    a = env.action_space.sample()
    obs, reward, done, _, _ = env.step(a)

    assert obs is not None


@pytest.mark.parametrize(
    "distraction",
    [
        "dots_linear",
        "dots_constant",
        "dots_episode",
        "dots_pendulum",
        "dots_random",
        "dots_quadlink",
    ],
)
def test_dots_sources(distraction):
    env = make(
        "cheetah",
        "run",
        distraction_location="background",
        distraction_source=distraction,
        from_pixels=True,
        height=128,
        width=256,
        frame_skip=2,
    )
    env = env
    env.reset()

    a = env.action_space.sample()
    obs, reward, done, _, _ = env.step(a)

    assert obs is not None
