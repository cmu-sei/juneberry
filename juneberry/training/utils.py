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

from argparse import Namespace
import logging
import sys

from prodict import Prodict

from juneberry.config.model import ModelConfig
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
import juneberry.loader as jb_loader
import juneberry.scripting as jb_scripting
from juneberry.training.trainer import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


def setup_training_logging_and_lab(args: Namespace, model_manager: ModelManager, dryrun: bool = False) -> Lab:
    # Use the config file to set up the workspace, data and logging
    log_prefix = "<<DRY_RUN>> " if dryrun else ""
    log_file = model_manager.get_training_dryrun_log_path() if dryrun else model_manager.get_training_log()

    return jb_scripting.setup_for_single_model(args, log_file=log_file, log_prefix=log_prefix,
                                               model_name=model_manager.model_name,
                                               banner_msg=">>> Juneberry Trainer <<<")


def setup_trainer_opt_args(args: Namespace):
    """
    Set up optional args from the command line.
    :param args: A Namespace object built from the attributes parsed out of the command line.
    """
    opt_args = {}
    opt_arg_keys = ['resume']
    for key in opt_arg_keys:
        if key in vars(args):
            opt_args[key] = vars(args)[key]

    return opt_args


def assemble_trainer_stanza(model_config: ModelConfig):
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
    model_config.trainer = Prodict(fqcn=task_platform_map[model_config.task][model_config.platform], kwargs={})
    logger.warning("Found deprecated platform/task configuration for loading trainer. "
                   "Consider updating the model config to use the trainer stanza.")
    logger.warning('"trainer": {')
    logger.warning(f'    "fqcn": "{model_config.trainer.fqcn}"')
    logger.warning('}')

    return model_config.trainer


def set_trainer_output_format(trainer: BaseTrainer, args: Namespace):
    # Set the output format for the trainer.
    trainer.onnx = args.onnx
    trainer.native = not args.skipNative

    # If no output format was set, choose the native format by default even if
    # the user requested it to be skipped.
    if not (trainer.onnx or trainer.native):
        logger.warning(f"An output format was not set. Choosing the native format.")
        trainer.native = True

    return trainer


def build_trainer(model_config: ModelConfig, args: Namespace, reqd_args_dict: dict):

    if model_config.trainer is None:
        model_config.trainer = assemble_trainer_stanza(model_config)
    else:
        if model_config.trainer.kwargs is None:
            model_config.trainer.kwargs = {}

    # If kw_args doesn't contain a required arg, substitute in the local variable for that kw_arg.
    reqd_args = ['lab', 'model_manager', 'model_config', 'dataset_config']
    for arg in reqd_args:
        if arg not in model_config.trainer.kwargs:
            model_config.trainer.kwargs[arg] = reqd_args_dict[arg]

    model_config.trainer.kwargs['log_level'] = logging.DEBUG if args.verbose else logging.INFO

    # Construct optional args that trainers may take.
    opt_args = setup_trainer_opt_args(args)

    # Construct the trainer
    logger.info(f"Instantiating trainer: {model_config.trainer.fqcn}")
    trainer = jb_loader.construct_instance(model_config.trainer.fqcn, model_config.trainer.kwargs, opt_args)

    if trainer is None:
        return trainer

    trainer = set_trainer_output_format(trainer, args)

    return trainer
