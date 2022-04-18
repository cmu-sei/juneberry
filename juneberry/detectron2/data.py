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

import logging
import numpy as np
from pathlib import Path
import sys

from cornelius.detectron2.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms import Transform

from juneberry.config.dataset import DatasetConfig
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
from juneberry.transform_manager import TransformManager


logger = logging.getLogger(__name__)

TRAIN_DS_NAME = "juneberry_train"
VAL_DS_NAME = "juneberry_val"
EVAL_DS_NAME = "juneberry_eval"


def register_train_manifest_files(lab: Lab, model_manager: ModelManager) -> None:
    """
    Registers the training and validation datasets with DT2.
    :param lab: The lab
    :param model_manager: The model manager
    :return: None
    """
    train_path = str(model_manager.get_training_data_manifest_path().resolve())
    split_path = str(model_manager.get_validation_data_manifest_path().resolve())

    if not Path(train_path).exists() or not Path(split_path).exists():
        logger.error(f"Could not find the training OR validation manifest. {train_path}, {split_path}"
                     f"EXITING.")
        sys.exit(-1)

    register_coco_instances(TRAIN_DS_NAME, {}, train_path, lab.data_root())
    register_coco_instances(VAL_DS_NAME, {}, split_path, lab.data_root())


def register_eval_manifest_file(lab: Lab, model_manager: ModelManager, dataset_config: DatasetConfig) -> None:
    """
    Registers the eval dataset with DT2.
    :param lab: The lab
    :param model_manager: The model manager
    :param dataset_config: The dataset configuration
    :return: None
    """
    manifest_path = model_manager.get_eval_manifest_path(dataset_config.file_path).resolve()

    if not Path(manifest_path).exists():
        logger.error(f"Could not find the evaluation manifest. {manifest_path}"
                     f"EXITING.")
        sys.exit(-1)

    register_coco_instances(EVAL_DS_NAME, {}, str(manifest_path), lab.data_root())


class TransformAdapter(T.Transform):

    # Duck typing adapter to move Juneberry transforms into detectron2
    def __init__(self, xform):
        self.xform = xform

    def check_for_methods(self):
        if not hasattr(self.xform, "apply_image") and \
                not hasattr(self.xform, "apply_box") and \
                not hasattr(self.xform, "apply_coords") and \
                not hasattr(self.xform, "apply_polygons") and \
                not hasattr(self.xform, "apply_segmentation"):
            logger.error("A transform has been supplied that doesn't have any Detectron2 methods. EXITING")
            print(self.xform)
            sys.exit(-1)

    def apply_image(self, img: np.ndarray):
        if hasattr(self.xform, "apply_image"):
            return self.xform.apply_image(img)
        else:
            return img

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        if hasattr(self.xform, "apply_box"):
            return self.xform.apply_box(box)
        else:
            return box

    def apply_coords(self, coords: np.ndarray):
        if hasattr(self.xform, "apply_coords"):
            return self.xform.apply_coords(coords)
        else:
            return coords

    def apply_polygons(self, polygons: list) -> list:
        if hasattr(self.xform, "apply_polygons"):
            return self.xform.apply_polygons(polygons)
        else:
            return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if hasattr(self.xform, "apply_segmentation"):
            return self.xform.apply_segmentation(segmentation)
        else:
            return segmentation

    def inverse(self) -> "Transform":
        pass


def create_mapper(cfg, transforms, is_train: bool) -> DatasetMapper:
    """
    Creates a dataset mapper from the config including the provided transforms.
    :param cfg: The detectron2 config object
    :param transforms: A list of transform objects
    :param is_train: True if training, false for testing
    :return: A new dataset mapper from the config with the augmentations.
    """
    args = DatasetMapper.from_config(cfg, is_train)
    if transforms is not None and len(transforms) > 0:
        mgr = TransformManager(transforms)
        aug_list = args['augmentations']
        for transform in mgr.get_transforms():
            if isinstance(transform, Augmentation) or isinstance(transform, Transform):
                aug_list.append(transform)
            else:
                adapter = TransformAdapter(transform)
                adapter.check_for_methods()
                aug_list.append(adapter)

    return DatasetMapper(**args)
