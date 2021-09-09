#! /usr/bin/env python3

from enum import Enum

__version__ = 0.4

class Platforms(str, Enum):
    """ A list of different supported platforms """
    DT2 = "detectron2"
    MMD = "mmdetection"
    PYT = "pytorch"
    PYT_PRIVACY = "pytorch_privacy"
