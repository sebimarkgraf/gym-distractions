from setuptools import find_packages, setup

setup(
    name="distractor_dmc2gym",
    version="1.0.2",
    author="yitian_yang",
    author_email="ulisb@student.kit.edu",
    description="A gym like wrapper for dm_control with distractions.",
    packages=find_packages(),
    install_requires=[
        "gym",
        "dm_control",
        "opencv-python",
        "numpy",
        "imageio",
        "scikit-video",
        "pytube",
    ],
    extras_require={"dev": ["pytest", "pre-commit"]},
)
