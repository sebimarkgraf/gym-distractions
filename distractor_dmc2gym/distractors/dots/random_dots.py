import numpy as np

from distractor_dmc2gym.distractors.dots import GeneralDotsSource


class RandomDotsSource(GeneralDotsSource):
    def update_positions(self):
        self.positions = np.concatenate(
            [
                self._np_random.uniform(*self.x_lim, size=(self.num_dots, 1)),
                self._np_random.uniform(*self.y_lim, size=(self.num_dots, 1)),
            ],
            axis=1,
        )
