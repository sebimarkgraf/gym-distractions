from enum import Enum, EnumMeta


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


class DistractorLocations(str, Enum, metaclass=EnumContainsMeta):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    BOTH = "both"
