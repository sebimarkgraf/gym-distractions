import warnings
from enum import Enum, EnumMeta
from functools import partial
from typing import Type, Union

from . import BaseStrategy
from .distractors import (
    ConstantDots,
    DAVISDataSource,
    DotsSource,
    EpisodeDotsSource,
    ImageSource,
    Kinetics400DataSource,
    LinearDotsSource,
    NoiseSource,
    PendulumDotsSource,
    QuadLinkDotsSource,
    RandomColorSource,
    RandomDotsSource,
    RandomVideoSource,
)
from .errors import GymDistractionsTypeError
from .merge_strategy import BackgroundMerge, FrontAndBackMerge, FrontMerge


class EnumContainsMeta(EnumMeta):
    def __contains__(self, item):
        return item in self.__members__.values()


class ImageSourceEnum(str, Enum, metaclass=EnumContainsMeta):
    VIDEO = "videos"
    IMAGE = "image"
    NOISE = "noise"
    COLOR = "color"
    DOTS = "dots"
    DOTS_CONSTANT = "dots_constant"
    DOTS_EPISODE = "dots_episode"
    DOTS_LINEAR = "dots_linear"
    DOTS_PENDULUM = "dots_pendulum"
    DOTS_QUADLINK = "dots_quadlink"
    DOTS_RANDOM = "dots_random"
    DAVIS = "davis"
    KINETICS = "kinetics400"


def map_distract_type_to_distractor(
    distract_type: Union[Type[ImageSource], ImageSourceEnum]
):
    if issubclass(type(distract_type), ImageSource):
        return distract_type

    if distract_type == ImageSourceEnum.DOTS:
        warnings.warn(
            "Dots were split in multiple classes. "
            "Please update your dots source to one of the new types.",
            DeprecationWarning,
        )
        distract_type = ImageSourceEnum.DOTS_LINEAR

    dot_types = {
        ImageSourceEnum.DOTS_LINEAR: LinearDotsSource,
        ImageSourceEnum.DOTS_CONSTANT: ConstantDots,
        ImageSourceEnum.DOTS_EPISODE: EpisodeDotsSource,
        ImageSourceEnum.DOTS_RANDOM: RandomDotsSource,
        ImageSourceEnum.DOTS_PENDULUM: PendulumDotsSource,
        ImageSourceEnum.DOTS_QUADLINK: QuadLinkDotsSource,
    }

    if distract_type in dot_types:
        behaviour = dot_types[distract_type]()
        return partial(DotsSource, dots_behaviour=behaviour)

    simple_types = {
        ImageSourceEnum.COLOR: RandomColorSource,
        ImageSourceEnum.NOISE: NoiseSource,
    }
    if distract_type in simple_types:
        return simple_types[distract_type]

    video_distractors = {
        ImageSourceEnum.VIDEO: RandomVideoSource,
        ImageSourceEnum.DAVIS: DAVISDataSource,
        ImageSourceEnum.KINETICS: Kinetics400DataSource,
    }
    if distract_type in video_distractors:
        return video_distractors[distract_type]

    raise GymDistractionsTypeError(f"Distractor of type {distract_type} not known.")


class MergeStrategies(str, Enum, metaclass=EnumContainsMeta):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    BOTH = "foreandbackground"


strategies = {
    MergeStrategies.FOREGROUND: FrontMerge,
    MergeStrategies.BACKGROUND: BackgroundMerge,
    MergeStrategies.BOTH: FrontAndBackMerge,
}


def map_strategy_config(strategy: Union[MergeStrategies, Type[BaseStrategy]]):
    if issubclass(type(strategy), BaseStrategy):
        return strategy

    try:
        return strategies[strategy]
    except KeyError as e:
        raise GymDistractionsTypeError(
            "Did not find matching strategy configuration"
        ) from e
