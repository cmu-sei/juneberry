#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM22-0856
#
# ======================================================================================================================

"""
Simple transformer for mirroring or shifting an image.  The JSON configuration accept two arguments for chance to flip.

"config": { "mirror_chance": 0.0, "flip_chance": 0.0 }

"""

import logging
import sys

import juneberry.image as jb_img_utils

logger = logging.getLogger(__name__)


class RandomMirrorFlip:
    def __init__(self, mirror_chance=None, flip_chance=None):
        self.mirror_chance = mirror_chance
        self.flip_chance = flip_chance

        if self.mirror_chance is None and self.flip_chance is None:
            logger.error(f"Neither 'mirror_chance' or 'flip_chance' found in specified. "
                         f"EXITING.")
            sys.exit(-1)

        if self.mirror_chance is not None and not 0 <= self.mirror_chance <= 1:
            logger.error(f"mirror_chance must be a value in range (0,1)."
                         f"mirror_chance value was {self.mirror_chance}"
                         f"EXITING.")
            sys.exit(-1)

        if self.flip_chance is not None and not 0 <= self.flip_chance <= 1:
            logger.error(f"flip_chance must be a value in range (0,1)."
                         f"flip_chance value was {self.flip_chance}"
                         f"EXITING.")
            sys.exit(-1)

    def __call__(self, image):
        """
        Transformation function that is provided a PIL image.
        :param image: The source PIL image.
        :return: The transformed PIL image.
        """
        return jb_img_utils.random_mirror_flip(image, self.mirror_chance, self.flip_chance)
