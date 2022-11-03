from setuptools import find_packages, setup

setup(
    name="distractor_dmc2gym",
    version="1.5.1",
    author="Sebastian Markgraf",
    author_email="sebastian-markgraf@t-online.de",
    description="A gym like wrapper for dm_control with distractions.",
    packages=find_packages(),
    install_requires=[
        "gym",
        "dm_control >= 1.0.0",
        "opencv-python",
        "numpy",
        "imageio",
        "scikit-video",
        "pytube @ git+https://github.com/kinshuk-h/pytube",
        "n-link-simulator @ git+ssh://git@github.com/sebimarkgraf/n-link-simulator@main",
    ],
    extras_require={"dev": ["pytest", "pre-commit"]},
)
