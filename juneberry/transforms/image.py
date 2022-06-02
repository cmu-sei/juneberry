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

from PIL import Image
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

class ChangeAllLabelsTo:
    def __init__(self, label):
        self.label = label

    def __call__(self, image, label):
        return( (image, self.label ))

class Watermark:
    def __init__(self, watermark_path, min_scale=1.0, max_scale=1.0, rotation=0, blur=0):
        # NOTE: Opening is lazy we need to force loading with load()
        self.watermark = Image.open(watermark_path).copy()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rotation = rotation
        self.blur = blur

    def __call__(self, image):
        # Copy the watermark so we can munge it        
        tmp_img: Image = self.watermark.copy()

        # Transform watermark
        tmp_img = iutils.transform_image(tmp_img, (self.min_scale, self.max_scale), self.rotation, self.blur)

        # Insert at a random location
        x, y = iutils.make_random_insert_position(tmp_img.size, image.size)
        image = iutils.insert_watermark_at_position(image, tmp_img, (x, y))

        return image
