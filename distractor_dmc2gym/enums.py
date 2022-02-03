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
    DAVIS = "davis"


class DistractorLocations(str, Enum, metaclass=EnumContainsMeta):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    BOTH = "both"
