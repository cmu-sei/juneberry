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

import datetime as dt
import inspect
import logging
from PIL import Image
from typing import Tuple, List
import warnings

from torch import Tensor
import torchvision.transforms.functional as functional

from juneberry.pytorch.utils import EpochDataset

logger = logging.getLogger(__name__)


class ImageDataset(EpochDataset):
    """
    Loads our data set for PyTorch from a list of entries of filename and labels and also allows on the fly
    transformation. This does not change the order of the images.
    """
    def __init__(self, data_list: List[Tuple[str, int]], transforms=None, no_paging=False):
        """
        Initializes our data set loader.
        :param data_list: The data.
        :param transforms: A function to transform the data.  The function should take two arguments. One of
           the full path as a string and the second a numpy array of shape height, width, channels and should
           return an image of same shape.
        :param no_paging: Set to true to disable paging and load all source images
        """
        super().__init__()

        self.data = data_list
        self.transforms = transforms
        self.image_cache = {}
        self.no_paging = no_paging

        # If the transforms takes the extended set, use them all
        self.extended_signature = False
        if transforms is not None:
            params = inspect.signature(transforms).parameters.keys()
            self.extended_signature = set(params) == {'item', 'index', 'epoch'}

        # Filters out a warning associated with a known Pillow issue that occurs
        # when opening images.
        warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

        if no_paging:

            logger.info("Preloading all images...")
            start_time = dt.datetime.now()
            for file_path, label in self.data:
                # NOTE: Opening is lazy we need to force loading with load()
                image = Image.open(file_path)
                image.load()
                self.image_cache[file_path] = image
            elapsed = (dt.datetime.now() - start_time).total_seconds()

            logger.info(f"...preloading complete! {len(data_list)} images loaded in {elapsed} seconds. "
                        f"{len(data_list)/elapsed:0.2f} images per second.")

    def __len__(self):
        """
        :return: Total number of samples.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns one image with its label
        :param index: The index within the data set
        :return: PyTorch Tensor, integer label
        """
        file_path = self.data[index][0]
        label = self.data[index][1]

        # If we preloaded then use the cache.
        if self.no_paging:
            image = self.image_cache[file_path].copy()
        else:
            image = Image.open(file_path)

        if self.transforms is not None:
            if self.extended_signature:
                image = self.transforms(item=image, index=index, epoch=self.epoch)
            else:
                image = self.transforms(image)

        # We want to pass back a tensor, so convert if it wasn't already converted
        if not isinstance(image, Tensor):
            image = functional.to_tensor(image)

        return image, label
