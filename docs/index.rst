Gym-Distractions
==============================

.. toctree::
   :hidden:
   :glob:
   :maxdepth: 1

   license


Installation
------------

Gym-Distractions can be installed directly from PyPi.
Therefore run:

.. code-block:: console

   $ pip install gym-distractions

in your console. By fixing the version, you can ensure that
your different environments run the same version.


Usage
-----

Depending on your usage, Gym-Distractions
support wrapping environments with distractions or using
the inbuilt Mujoco support.

To create a Mujoco environment import the make function and
use it with your wanted domain and task:

.. code-block:: python

    from gym_distractions import make

    env = gym_distractions.make(domain_name,
                    task_name,
                    distract_type,
                    ground,
                    difficulty,
                    background_dataset_path,
                    train_or_val,
                    seed,
                    visualize_reward,
                    from_pixels,
                    height,
                    width,
                    camera_id,
                    frame_skip,
                    episode_length,
                    environment_kwargs,
                    time_limit,
                    channels_first
                    )
