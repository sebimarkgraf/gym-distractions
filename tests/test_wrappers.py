import numpy as np
import pytest
from PIL import Image

from gym_distractions.wrappers import DMCWrapper

default_args = {
    "domain_name": "cheetah",
    "task_name": "run",
    "task_kwargs": {"random": 1},
    "from_pixels": True,
    "channels_first": False,
}


def create_image(dir, width=1920, height=1080, num_of_images=100):
    width = int(width)
    height = int(height)
    num_of_images = int(num_of_images)

    for n in range(num_of_images):
        filename = dir / f"{n}.png"
        rgb_array = np.random.default_rng().normal(height, width, 4) * 255
        img = Image.fromarray(rgb_array.astype("uint8")).convert("RGBA")
        img.save(filename)


@pytest.fixture(scope="session")
def resource_files(tmpdir_factory):
    dir = tmpdir_factory.mktemp("data")
    create_image(dir, 84, 84, 5)

    return dir.join("*.png")


def test_render_foreground():
    wrapper = DMCWrapper(
        **default_args,
        distract_type="dots",
        ground="foreground",
    )
    obs = wrapper.render()
    assert obs is not None
    # Mujoco is always RGB
    assert obs.shape == (84, 84, 3)
    assert obs.shape == wrapper.observation_space.shape
    assert obs.dtype == np.dtype("uint8")


def test_render_background():
    wrapper = DMCWrapper(**default_args, distract_type="dots", ground="background")
    obs = wrapper.render()
    assert obs is not None
    # Mujoco is always RGB
    assert obs.shape == (84, 84, 3)
    assert obs.shape == wrapper.observation_space.shape
    assert obs.dtype == np.dtype("uint8")


def test_step():
    wrapper = DMCWrapper(
        **default_args,
    )
    a = wrapper.action_space.sample()
    obs, reward, done, truncated, _ = wrapper.step(a)

    assert obs is not None
    # Mujoco is always RGB
    assert obs.shape == (84, 84, 3)
    assert obs.shape == wrapper.observation_space.shape
    assert obs.dtype == np.dtype("uint8")
