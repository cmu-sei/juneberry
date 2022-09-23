#! /usr/bin/env python3

"""
Simple transformer for sharpening or blurring an image. Takes a float value based on a provided range
which determines whether the image is to be blurred or sharpened.
"""

import logging
import sys

import juneberry.image as iutils

logger = logging.getLogger(__name__)

class RandomSharpen:
	def __init__(self):
		#self.scale = scale
		#if self.scale < 0.0 or self.scale > 2.0:
		#	logger.error(f"Scale value out of range. Must be between 0.0 and 2.0"
		#			f"EXITING.")
		pass
	def __call__(self, image):
		"""
		Transformation function that is provided a PIL image.
		:param image: The source PIL image.
		:return: The transformed PIL image.
		"""
		return iutils.random_sharpen(image)