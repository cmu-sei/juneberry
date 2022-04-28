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

import logging
import numpy as np
from pathlib import Path
import sys

from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms import Transform

from juneberry.config.dataset import DatasetConfig
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
import juneberry.mappers.utils as jb_mapper_utils
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


def create_mapper(cfg, dataset_mapper, transforms, is_train: bool) -> DatasetMapper:
    """
    Creates a dataset mapper from the config including the provided transforms.
    :param cfg: The detectron2 config object
    :param transforms: A list of transform objects
    :param is_train: True if training, false for testing
    :param dataset_mapper: A string for the custom dataset mapper to use
    :return: A new dataset mapper from the config with the augmentations.
    """

    # Conditional to test if a custom dataset mapper has been set in our cfg
    if dataset_mapper is not None:
        logger.info("Using custom dataset mapper to override Detectron2 dataset mapper")
        mapper_obj = jb_mapper_utils.construct_mapper(dataset_mapper.fqcn, dataset_mapper.kwargs)
        args = mapper_obj.from_config(cfg, is_train)
    else:
        logger.info("Using standard Detectron2 dataset mapper")
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
