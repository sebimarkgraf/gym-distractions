from enum import IntEnum

import numpy as np

from distractor_dmc2gym.distractors.dots import GeneralDotsSource


class GravitationalConstant(IntEnum):
    PLANET = 1
    ELECTRONS = -1
    IDEAL_GAS = 0


class LinearDotsSource(GeneralDotsSource):
    def __init__(
        self,
        *args,
        base_velocity: float = 0.5,
        gravitation: GravitationalConstant = GravitationalConstant.IDEAL_GAS,
        **kwargs
    ):
        self.v = base_velocity
        super(LinearDotsSource, self).__init__(*args, **kwargs)
        self.gravitation = gravitation

    def init_dots(self):
        return {
            **super(LinearDotsSource, self).init_dots(),
            "velocities": (
                self._np_random.normal(0, 0.01, size=(self.num_sets, self.num_dots, 2))
                * self.v
            ),
        }

    def update_positions(self):
        def compute_acceleration(
            positions: np.array, sizes: np.array, gravitation: int
        ):
            accelerations = np.zeros(positions.shape)
            for i in range(len(positions)):
                relative_positions = positions - positions[i]
                distances = np.linalg.norm(relative_positions, axis=1, keepdims=True)
                distances[i] = 1

                force_vectors = (
                    relative_positions * gravitation * (sizes**2) / (distances**2)
                )
                accelerations[i] = 0.00001 * np.sum(force_vectors, axis=0)

            return accelerations

        accelerations = compute_acceleration(
            np.array(self.positions), np.array(self.sizes), self.gravitation
        )
        self.velocities += accelerations
        self.positions += self.velocities

        for i in range(self.positions.shape[0]):
            self._limit_position(i)

    def _limit_position(self, i):
        if not self.x_lim.high >= self.positions[i][0] >= self.x_lim.low:
            self.velocities[i][0] = -self.velocities[i][0]
        if not self.y_lim.high >= self.positions[i][1] >= self.y_lim.low:
            self.velocities[i][1] = -self.velocities[i][1]

    def reset_dots(self):
        idx = super(LinearDotsSource, self).reset_dots()
        self.velocities = self.dots_init["velocities"][idx].copy()
        return idx
