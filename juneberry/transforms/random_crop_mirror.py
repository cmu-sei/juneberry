#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - General Release
#
# Copyright 2021 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software subject to its own license.
#
# DM21-0884
#
# ======================================================================================================================

"""
Simple transformer to shift an image that may have been mirrored. The JSON configuration requires three arguments:
amount of horizontal shift allowed (in pixels), amount of vertical shift allowed (in pixels) and a boolean to control
whether or not the image should be mirrored.

"config": { "width_pixels": 0, "height_pixels": 0, "mirror": 0 }

"""

import juneberry.image as jb_img_utils


class RandomCropMirror:
    def __init__(self, width_pixels, height_pixels, mirror):
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
        self.mirror = mirror

    def __call__(self, image):
        """
        Transformation function that is provided a PIL image.
        :param image: The source PIL image.
        :return: The transformed PIL image.
        """
        return jb_img_utils.random_crop_mirror_image(image, self.mirror, self.width_pixels, self.height_pixels)
