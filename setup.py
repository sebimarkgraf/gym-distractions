from setuptools import setup, find_packages

setup(name='distractor_dmc2gym',
      version='1.0.0',
      author='yitian_yang',
      author_email='ulisb@student.kit.edu',
      description='A gym like wrapper for dm_control with distractions.',
      packages=find_packages(),
      install_requires=[
          'gym',
          'dm_control',
          'opencv-python',
          'numpy',
          'imageio'
      ],
      )
