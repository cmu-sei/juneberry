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
A set of general image conversions.
"""

import juneberry.image as iutils


class ConvertMode:
    """
    Converts the mode of the input image to the specified mode.
    "kwargs": { "mode": 'RGB' }
    """

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        if image.mode != self.mode:
            return image.convert(self.mode)

        return image


class ResizePad:
    """
    Resizes the image maintaining aspect ratio, padding with the specified color if necessary.

    NOTE: This uses Image.ANTIALIAS resampling.

    "kwargs": { "width": 224, "height": 224, "color": [ 0,0,0 ] }
    """

    def __init__(self, width, height, pad_color=(0, 0, 0)):
        self.width = width
        self.height = height
        self.color = pad_color

    def __call__(self, image):
        return iutils.resize_image(image, self.width, self.height, self.color)


class Watermark:
    def __init__(self, patch_path, size):
        # Load the patch and set size
        #
        pass

    def __call__(self, image):
        # Call the guts of the patch injector
        # iutils.watermark(image, self.patch)
        return image

