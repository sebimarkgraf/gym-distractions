from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import json


DIFFICULTY_SCALE = dict(easy=0.1, medium=0.2, hard=0.3)


class ImageSource(object, metaclass=ABCMeta):
    @abstractmethod
    def get_image(self) -> Tuple[np.array, np.array]:
        pass

    def reset(self):
        pass

    def get_info(self):
        info = {}
        return info

    def save_info(self, path: Path):
        info = {}
        info[self.__class__.__name__] = self.get_info()
        with open(path / 'distractors_info.json', "w") as f:
            json.dump(info, f, indent=4)

