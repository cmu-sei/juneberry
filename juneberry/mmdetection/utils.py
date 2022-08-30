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
import os
from pathlib import Path
import sys

import mmdet
from mmdet.apis import set_random_seed

from juneberry.config.model import ModelConfig
import juneberry.pytorch.utils as pyt_utils
from juneberry.pytorch.utils import PyTorchPlatformDefinitions

logger = logging.getLogger(__name__)


class MMDPlatformDefinitions(PyTorchPlatformDefinitions):
    def has_platform_config(self) -> bool:
        return True


def find_mmdetection():
    """
    Finds and returns the 'mmdetection' directory or exits if it could not be found.
    :return: The directory that contains a 'configs' directory.
    """

    # The environment variable is an override if they want to use a different one because they
    # may have modified configs, or some other reason.

    if 'MMDETECTION_HOME' in os.environ:
        logger.info("Locating 'mmdetection' directory via environment variable: MMDETECTION_HOME...")
        mm_home = Path(os.environ['MMDETECTION_HOME'])
    else:
        # The packager __init__.py file will be:
        # .../mmdetection/mmdet/__init__.py
        # Go up twice to get mmdetection
        # Get the package init file and then go up two levels
        logger.info("Locating mmdetection by installed package...")
        mm_home = Path(mmdet.__file__).parent.parent

    # Now make sure that directory exists.
    if not mm_home.exists():
        logger.error(f"Could not locate MMDetection directory. Please set MMDETECTION_HOME to point to the "
                     f"mmdetection directory. EXITING!")
        sys.exit(-1)

    # There should be a configs directory.
    if not (mm_home / 'configs').exists():
        logger.error(f"Could not find 'configs' directory in mmdetection path {mm_home}. EXITING!")
        sys.exit(-1)

    logger.info(f"...found mmdetection home: '{mm_home}'")

    return mm_home


def add_reproducibility_configuration(model_config: ModelConfig, cfg) -> None:
    """
    Add seeds and deterministic settings to the provided config.
    :param model_config: The config from which to get the seeds and determinism values.
    :param cfg: Where to apply the values.
    """
    # Set the basic seeds.
    pyt_utils.set_pytorch_seeds(model_config.seed)

    # Set the back end specific seeds.
    cfg.seed = model_config.seed
    if hasattr(model_config, 'pytorch'):
        if model_config.pytorch.get('deterministic', False):
            set_random_seed(model_config.seed, deterministic=True)
    else:
        set_random_seed(model_config.seed, deterministic=False)


def add_config_overrides(model_config: ModelConfig, cfg) -> None:
    """
    Adds everything from the "mmdetection.overrides" from the model config, if exists.
    :param model_config: The config from which to extract overrides.
    :param cfg: Where to apply the overrides.
    """
    if hasattr(model_config, "mmdetection"):
        if "overrides" in model_config.mmdetection:
            logger.info("Merging in overrides from config file.")
            cfg.merge_from_dict(model_config.mmdetection['overrides'])


#  _____                    __
# |_   _|                  / _|
#   | |_ __ __ _ _ __  ___| |_ ___  _ __ _ __ ___  ___
#   | | '__/ _` | '_ \/ __|  _/ _ \| '__| '_ ` _ \/ __|
#   | | | | (_| | | | \__ \ || (_) | |  | | | | | \__ \
#   \_/_|  \__,_|_| |_|___/_| \___/|_|  |_| |_| |_|___/

# Docs for how to customize
# https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html

# Existing transforms:
# https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/datasets/pipelines/transforms.html

"""
SAMPLE Pipeline for Training
pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
],
"""

"""
SAMPLE Pipeline for Testing
NOTE the MultiScaleFlipAug
pipeline=[
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
],
"""

# These are the json-style but THEY DON"T ALWAYS WORK because some of this things required tuples

"""
[
    {"type": "LoadImageFromFile"},
    {"type": "LoadAnnotations", "with_bbox": true},
    {"type": "Resize", "img_scale": [[1333, 640], [1333, 672], [1333, 704], [1333, 736], [1333, 768], [1333, 800]], 
    "multiscale_mode": "value", 
    "keep_ratio": true}, 
    {"type": "RandomFlip", "flip_ratio": 0.5}, 
    {"type": "Normalize", "mean": [103.53, 116.28, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": false}, 
    {"type": "Pad", "size_divisor": 32}, {"type": "DefaultFormatBundle"}, 
    {"type": "Collect", "keys": ["img", "gt_bboxes", "gt_labels"]}
]
[
    { "type": "LoadImageFromFile"},
    { "type": "MultiScaleFlipAug", 
      "img_scale": [1333, 800], 
      "flip": false, 
      "transforms": [
          {"type": "Resize", "keep_ratio": true}, 
          {"type": "RandomFlip"}, 
          {"type": "Normalize", "mean": [103.53, 116.28, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": false}, 
          {"type": "Pad", "size_divisor": 32}, 
          {"type": "ImageToTensor", "keys": ["img"]}, 
          {"type": "Collect", "keys": ["img"]}
        ]
    }
]

"""

# Here is what we'll add to the training config docs.

"""
"mmdetection": {
    "train_pipeline_stages": [
        {
        }
    ],
    "test_pipeline_stages": [
        {
        }
    ]
}
"""


def adjust_pipelines(model_config, cfg) -> None:
    """
    Adjust the "cfg.data" pipelines in the config based on the stanzas in the model config.
    :param model_config: The model config that might have pipelines.
    :param cfg: The config that has the existing pipelines.
    :return: None
    """
    if hasattr(model_config, "mmdetection"):
        new_stages = model_config.mmdetection.get('train_pipeline_stages', None)
        if new_stages:
            if len(cfg.data.train):
                logger.info("Adding in 'cfg.data.train.pipeline' stages...")
                add_stages(cfg.data.train.pipeline, new_stages)

        new_stages = model_config.mmdetection.get('val_pipeline_stages', None)
        if new_stages:
            if len(cfg.data.val):
                logger.info("Adding in 'cfg.data.val.pipeline' stages...")
                add_stages(cfg.data.val.pipeline, new_stages)

            # Make sure they aren't the same pipeline. We don't want to double modify
            if not cfg.data.val is cfg.data.test and len(cfg.data.test):
                logger.info("Adding in 'cfg.data.test.pipeline' stages...")
                add_stages(cfg.data.test.pipeline, new_stages)


def add_stages(pipeline, stage_list: list) -> None:
    """
    Injects the provided stages into the pipeline based on the criteria in the stage list.
    The provided pipeline is modified in place.
    The entries in the list are expected to be:
    {
        "name": < Name of existing stage. >,
        "stage": { <mmdetection stanza with "type" keyword> },
        "mode": < optional: [ before (default) | replace | after | delete ] How we insert relative to "name">,
        "tupleize": < optional: True to convert list values in stage to tuples before adding. Default is False. >
    }
    :param pipeline: Where to inject the new stages.
    :param stage_list: A description of the stage to inject.
    :return: None
    """
    for entry in stage_list:
        # Make sure we have a valid mode
        mode = entry.get('mode', 'before')
        if not mode in ['before', 'replace', 'update', 'after', 'delete']:
            logger.error(f"Unknown mode={mode} in entry={entry} EXITING!")
            sys.exit(-1)

        if 'name' not in entry:
            logger.info(f"No 'name' field in entry. entry={entry}. EXITING")
            sys.exit(-1)

        if mode != 'delete' and 'stage' not in entry:
            logger.info(f"No 'stage' field in entry. entry={entry}. EXITING")
            sys.exit(-1)

        stage = entry.get('stage', None)
        if stage is not None and entry.get('tupleize', False):
            stage = tupleize(stage)

        add_stage(pipeline, entry['name'], stage, mode)


def add_stage(pipeline, name, stage, mode) -> None:
    """
    This is used to inject the provided stage into the pipeline before or after
    the stage specified by the name. The pipeline is modified by this call.
    :param pipeline: Where to inject the stage.
    :param name: The name of the stage to find.
    :param stage: The new stage to inject.
    :param after: True to inject after the named stage.
    :return: None
    """

    for idx, row in enumerate(pipeline):
        if row['type'] == 'MultiScaleFlipAug':
            # This is a special case that even the MMD code recognizes. Basically
            # this is a wrapper and we need to add it there.

            # If they want to modify the MultiScaleFlipAug itself directly then
            # we can't directly edit it.  Basically we only support "replace"
            # which allows updating the fields.
            if name == "MultiScaleFlipAug":
                if mode == "update":
                    # Update the ones they provided us.
                    logger.info(f"...updating stage {name}. stage={stage}")
                    pipeline[idx].update(stage)
                else:
                    logger.error("We only allow 'update' on MultiScaleFlipAug. EXITING!")
                    sys.exit(-1)
            else:
                add_stage(row['transforms'], name, stage, mode)
            return
        elif row['type'] == name:
            # Create an options struct to inject.  This is following the sample code
            # in the api for merge_from_dict
            if mode == "before":
                logger.info(f"...inserting stage BEFORE {name}. stage={stage}")
                pipeline.insert(idx, stage)
            elif mode == "replace":
                logger.info(f"...replacing stage {name}. stage={stage}")
                pipeline[idx] = stage
            elif mode == "update":
                logger.info(f"...updating stage {name}. stage={stage}")
                pipeline[idx].update(stage)
            elif mode == "after":
                logger.info(f"...inserting stage AFTER {name}. stage={stage}")
                pipeline.insert(idx + 1, stage)
            elif mode == "delete":
                logger.info(f"...deleting stage {name}.")
                del pipeline[idx]
            else:
                logger.error(f"Unknown inertMode:{mode}, EXITING")
                sys.exit(-1)

            return

    # If we didn't get this far, then we have a problem.
    names = [r['type'] for r in pipeline]
    msg = f"Did not find stage name='{name}' in names='{','.join(names)}'"
    logger.error(f"{msg} EXITING")
    sys.exit(-1)


def tupleize(struct):
    """
    Creates a version of the structure where the lists are tuples.
    :param struct: The struct to convert.
    :return: A struct with all tuples instead of lists
    """
    if isinstance(struct, (list, tuple)):
        return tuple([tupleize(x) for x in struct])
    elif isinstance(struct, dict):
        return {k: tupleize(v) for k, v in struct.items()}
    return struct

#  _    _ ___________
# | |  | |_   _| ___ \
# | |  | | | | | |_/ /
# | |/\| | | | |  __/
# \  /\  /_| |_| |
#  \/  \/ \___/\_|

# In the future we want to be able to inject "NORMAL" python transforms into the
# pipeline.  We need to wrap it into our own module and then inject like we do
# other pipelines.

# from mmdet.datasets import PIPELINES

# @PIPELINES.register_module()
# class JuneberryTransformAdapter:
#     def __init__(self, transform_list):
#         self.transform_manager = TransformManager(transform_list)
#         pass
#
#     def __call__(self, results):
#         results['dummy'] = True
#         return results
#
