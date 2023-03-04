import numpy as np

from .dots_source import DotsBehaviour, Limits, T


class EpisodeDotsSource(DotsBehaviour):
    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> T:
        return {
            "positions": np.concatenate(
                [
                    np_random.uniform(*x_lim, size=(num_dots, 1)),
                    np_random.uniform(*y_lim, size=(num_dots, 1)),
                ],
                axis=1,
            ),
        }

    def update_state(self, state: T) -> T:
        return state

    def get_positions(self, state: T) -> np.array:
        return state["positions"]
