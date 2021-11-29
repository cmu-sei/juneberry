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
Simple transformer for unit testing. The config will specify what the transformer should expect for
size and mode and the transformer just checks.
"""

import logging
from numpy.random import default_rng
import random as python_random
import torch

logger = logging.getLogger(__name__)


class SizeCheckImageTransform:
    def __init__(self, width, height, mode):
        self.width = width
        self.height = height
        self.mode = mode

    def __call__(self, image):
        """
        Transformation function that is provided a Pillow image or PyTorch Tensor.
        :param image: The source PIL image.
        :return: The transformed PIL image.
        """
        if image.size[0] != self.width:
            logger.error(f"Image handed to transformer not of expected width. "
                         f"Expected {self.width} found {image.size[0]}")
        if image.size[1] != self.height:
            logger.error(f"Image handed to transformer not of expected height."
                         f"Expected {self.height} found {image.size[1]}")
        if image.mode != self.mode:
            logger.error(f"Image handed to transformer not of expected mode."
                         f"Expected {self.mode} found {image.mode}")
        return image


class NoOpTensorTransform:
    def __init__(self, name, **kwargs):
        self.name = name

    def __call__(self, arg=None, **kwargs):
        """
        Transform that does nothing.
        :param arg: A single source value.
        :return: The value unchanged.
        """
        return arg


class MinimumInOutBuilder:
    def __init__(self, **kwargs):
        self.base_dataset = "models/tabular_binary_sample/train_data_config.json"
        self.base_csv = "models/tabular_binary_sample/train_data.csv"
        self.training_file = kwargs["training_config_destination"]
        self.val_file = kwargs["val_config_destination"]
        self.test_file = kwargs["test_config_destination"]

    def __call__(self, **kwargs):
        from shutil import copyfile
        from pathlib import Path

        copyfile(self.base_dataset, self.training_file)
        copyfile(self.base_dataset, self.val_file)
        copyfile(self.base_dataset, self.test_file)

        csv_name = (Path(self.base_csv)).name
        copyfile(self.base_csv, Path(self.training_file).parent / csv_name)


class TypeLogTransform:
    def __init__(self, message):
        self.message = message

    def __call__(self, arg):
        """
        Transform that logs the supplied message and the type.
        :param arg: The input type.
        :return: The output type changed.
        """
        logger.info(f"{self.message} - {type(arg)}")
        return arg


class ShowRandomNumber:
    def __init__(self, message):
        self.msg = message

    def __call__(self, arg):
        logger.info(f"+++++ {self.msg} python={python_random.random()} {default_rng().random()} {torch.rand(1)}")
        return arg
