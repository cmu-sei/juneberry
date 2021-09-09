#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

import datetime as dt
import logging
from PIL import Image
from torch import Tensor
from torch.utils import data
import torchvision.transforms.functional as functional
from typing import Tuple, List
import warnings

logger = logging.getLogger(__name__)


class ImageDataset(data.Dataset):
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
        self.data = data_list
        self.transforms = transforms
        self.image_cache = {}
        self.no_paging = no_paging

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
            image = self.transforms.transform(image)

        # We want to pass back a tensor, so convert if it wasn't already converted
        if not isinstance(image, Tensor):
            image = functional.to_tensor(image)

        return image, label
