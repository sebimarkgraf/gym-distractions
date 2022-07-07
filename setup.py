from setuptools import find_packages, setup

setup(
    name="distractor_dmc2gym",
    version="1.2.0",
    author="Sebastian Markgraf",
    author_email="sebastian-markgraf@t-online.de",
    description="A gym like wrapper for dm_control with distractions.",
    packages=find_packages(),
    install_requires=[
        "gym",
        "dm_control",
        "opencv-python",
        "numpy",
        "imageio",
        "scikit-video",
        "pytube @ git+https://github.com/kinshuk-h/pytube",
    ],
    extras_require={"dev": ["pytest", "pre-commit"]},
)
