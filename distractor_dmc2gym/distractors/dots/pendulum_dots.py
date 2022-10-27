import numpy as np

from distractor_dmc2gym.distractors.dots import GeneralDotsSource


class PendulumDotsSource(GeneralDotsSource):
    # def __init__(self, *args, dt, sim_dt, mass, length, g, friction, scale_factor, transition_noise_std):
    #    super(PendulumDotsSource, self).__init__(*args)
    #    self.dt = dt
    #    self.sim_dt = sim_dt
    #    self.mass = mass
    #    self.length = length
    #    self.inertia = self.mass * (self.length ** 2) / 3
    #    self.g = g
    #    self.friction = friction
    #    self.scale_factor = scale_factor
    #    self.transition_noise_std = transition_noise_std

    # def init_dots(self) -> dict:
    #    return {
    #        **super(PendulumDotsSource, self).init_dots(),
    #        "pendulum": np.array([self._np_random.uniform(-np.pi, np.pi), 0.0])
    #    }

    def update_positions(self):
        pass
