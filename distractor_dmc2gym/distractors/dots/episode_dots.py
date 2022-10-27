from distractor_dmc2gym.distractors.dots import GeneralDotsSource


class EpisodeDotsSource(GeneralDotsSource):
    def update_positions(self):
        """We do not update Positions here"""
