from distractor_dmc2gym.distractors.dots.dots_source import GeneralDotsSource


class ConstantDotsSource(GeneralDotsSource):
    def update_positions(self):
        """We do not update Positions here"""
        pass

    def reset_dots(self):
        """We do not update the Dots in new episodes"""
        set_idx = 0
        self.colors, self.positions, self.sizes = (
            self.dots_init["colors"][set_idx].copy(),
            self.dots_init["positions"][set_idx].copy(),
            self.dots_init["sizes"][set_idx].copy(),
        )
        return set_idx
