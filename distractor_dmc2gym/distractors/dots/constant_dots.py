from distractor_dmc2gym.distractors.dots.dots_source import GeneralDotsSource


class ConstantDotsSource(GeneralDotsSource):
    def update_positions(self):
        """We do not update Positions here"""
        pass

    def reset_dots(self):
        """We do not update the Dots in new episodes"""
        self.colors, self.positions, self.sizes = (
            self.dots_init["colors"].copy(),
            self.dots_init["positions"].copy(),
            self.dots_init["sizes"].copy(),
        )
