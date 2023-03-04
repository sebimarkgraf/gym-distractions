import numpy as np

from .dots_source import DotsBehaviour, Limits, T


class ConstantDots(DotsBehaviour):
    def init_state(
        self,
        num_dots: int,
        x_lim: Limits,
        y_lim: Limits,
        np_random: np.random.Generator,
    ) -> T:
        return {
            # Fix always yield the same
            "positions": np.stack(
                [np.linspace(*x_lim, num=num_dots), np.linspace(*y_lim, num=num_dots)],
                axis=1,
            )
        }

    def update_state(self, state: T) -> T:
        return state

    def get_positions(self, state: T) -> np.array:
        return state["positions"]
