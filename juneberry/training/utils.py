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
Training utilities
"""

import logging
import sys

from juneberry.config.model import ModelConfig

logger = logging.getLogger(__name__)


def assemble_stanza_and_construct_trainer(model_config: ModelConfig) -> str:
    """
    This function enables backwards compatibility with older versions of Juneberry Model config
    files which do not contain a "trainer" stanza. The function interprets the "task" and "platform"
    fields in older model configs and determines which string to use for the "fqcn" field in the
    "trainer" stanza of the ModelConfig.
    :param model_config: A ModelConfig object that does not have a "trainer" field.
    :return: The "fqcn" to use in the "trainer" field, based on the "task" and "platform"
    listed in the ModelConfig.
    """
    # Backward compatible map of task and platform name to class to load
    task_platform_map = {
        "classification": {
            "pytorch": "juneberry.pytorch.classifier_trainer.ClassifierTrainer",
            "pytorch_privacy": "juneberry.pytorch.privacy.classifier_trainer.PrivacyTrainer",
            "tensorflow": "juneberry.tensorflow.trainer.ClassifierTrainer",
            "tfgloro": "juneberry.tensorflow.trainer.ClassifierTrainer"
        },
        "objectDetection": {
            "detectron2": "juneberry.detectron2.trainer.Detectron2Trainer",
            "mmdetection": "juneberry.mmdetection.trainer.MMDTrainer"
        }
    }

    # Dig through our map of tasks and platforms
    if model_config.task not in task_platform_map:
        logger.error(f"Juneberry training does not support the task '{model_config.task}'. "
                     f"Supported tasks: {list(task_platform_map.keys())}. Exiting.")
        sys.exit(-1)
    fqcn = task_platform_map[model_config.task][model_config.platform]
    logger.warning("Found deprecated platform/task configuration for loading trainer. "
                   "Consider updating the model config to use the trainer stanza.")
    logger.warning('"trainer": {')
    logger.warning(f'    "fqcn": "{fqcn}"')
    logger.warning('}')

    return fqcn
