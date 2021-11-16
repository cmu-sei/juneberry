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
from types import SimpleNamespace

import torch
from tqdm import tqdm

import juneberry.config.coco_utils as coco_utils
from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.model import ModelConfig
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.loader as jb_loader
from juneberry.metrics.metrics import Metrics

logger = logging.getLogger(__name__)


def get_histogram(dataset_dicts, classes):
    """
    This function produces a histogram of the bounding box classes in a
    :param dataset_dicts: A list of dataset dictionaries, where each dictionary contains
    bounding box annotation data.
    :param classes: A zero-indexed list of the possible classes for the bounding boxes.
    :return: A dictionary containing the histogram data; i.e. how many bounding boxes
    appear in the dataset for each class.
    """

    # Convert class_names from dict into the expected list.
    class_names = [None] * (max(classes.keys()) + 1)
    for k, v in classes.items():
        class_names[int(k)] = v

    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int)
        if len(classes):
            assert classes.min(
            ) >= 0, f"Got an invalid category_id={classes.min()}"
            assert (classes.max() < num_classes), \
                f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    hist_dict = {}
    total = 0
    for i, v in enumerate(histogram):
        hist_dict[class_names[i]] = v
        total += v

    hist_dict['total'] = total

    return hist_dict


def create_evaluator(model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                     dataset: DatasetConfig, eval_options: SimpleNamespace, log_file: str):
    """
    This function is responsible for creating the correct evaluator object based on the contents of the ModelConfig.
    :param model_config: The loaded Juneberry ModelConfig which determines which Evaluator to build.
    :param lab: The Juneberry Lab that will be used to build the Evaluator.
    :param dataset: The Juneberry DatasetConfig that will be used to build the Evaluator.
    :param model_manager: The Juneberry ModelManager that will be used to build the Evaluator.
    :param eval_dir_mgr: The Juneberry EvalDirMgr that will be used to build the Evaluator.
    :param eval_options: A SimpleNamespace of different eval options that will be used to build the Evaluator.
    :param log_file: A string indicating the location of the current log file.
    """

    platform_map = {
        "pytorch": "juneberry.pytorch.evaluator.Evaluator",
        "pytorch_privacy": "juneberry.pytorch.evaluator.Evaluator",
        "detectron2": "juneberry.detectron2.evaluator.Evaluator",
        "mmdetection": "juneberry.mmdetection.evaluator.Evaluator",
        "tensorflow": "juneberry.tensorflow.evaluator.Evaluator",
        "tfgloro": "juneberry.tensorflow.gloro.evaluator.Evaluator"
    }

    # If the model config does not specify an evaluator to use, determine the default
    # evaluator using the platform map.
    if model_config.evaluator is None:

        # If the platform is not in the platform map, then there is no default evaluator for the
        # platform.
        if model_config.platform not in platform_map:
            logger.error(f"Evaluation not supported for the requested platform ({model_config.platform}). "
                         f"Supported platforms: {list(platform_map.keys())}. EXITING.")
            sys.exit(-1)

        # Obtain the fqcn from the platform map and warn the user.
        else:
            fqcn = platform_map[model_config.platform]

        kw_args = {}
        logger.warning("Found deprecated platform/task configuration for loading the evaluator. "
                       "Consider updating the model config to use the evaluator stanza.")
        logger.warning('"evaluator": {')
        logger.warning(f'    "fqcn": "{fqcn}"')
        logger.warning('}')

    # Handle the situation where the model config has an evaluator stanza.
    else:
        kw_args = model_config.evaluator.kwargs
        if kw_args is None:
            kw_args = {}
        fqcn = model_config.evaluator.fqcn

    reqd_args = ['lab', 'model_config', 'dataset',
                 'model_manager', 'eval_dir_mgr', 'eval_options', 'log_file']

    # If kw_args doesn't contain a required arg, substitute in the local variable for that kw_arg.
    for arg in reqd_args:
        if arg not in kw_args:
            kw_args[arg] = locals()[arg]

    # Create the Evaluator.
    logger.info(f"Instantiating evaluator: {fqcn}")
    evaluator = jb_loader.construct_instance(fqcn, kw_args)

    return evaluator


def continuous_predictions_to_class(y_pred, binary):
    """
    Convert a set of continuous predictions to numeric class.
    :param y_pred: The float predictions.
    :param binary: True if the data is binary
    :return: The classes
    """
    if binary:
        return np.round(y_pred).astype(int)
    else:
        return np.argmax(y_pred, axis=1)


def binary_to_classes(binary_predictions):
    """
    Expands the singular binary predictions to two classes
    :param binary_predictions:
    :return: The predictions broken into two probabilities.
    """
    return [[1.0 - x[0], x[0]] for x in binary_predictions]


def predict_classes(data_generator, model, device):
    """
    Generates predictions data for the provided data set via this model.
    :param data_generator: The data generator to provide data.
    :param model: The trained model.
    :param device: The device on which to do the predictions. The model should already be on the device.
    :return: A table of the predictions.
    """
    all_outputs = None
    for local_batch, local_labels in tqdm(data_generator):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(
            device), local_labels.to(device)

        # Model computations
        output = model(local_batch)

        with torch.set_grad_enabled(False):
            if all_outputs is None:
                all_outputs = output.detach().cpu().numpy()
            else:
                all_outputs = np.concatenate(
                    (all_outputs, output.detach().cpu().numpy()))

    return all_outputs.tolist()


def invoke_evaluator_method(evaluator, module_name: str):
    """
    This function is responsible for invoking methods during evaluation.
    :param evaluator: A Juneberry Evaluator object that is managing the evaluation.
    :param module_name: The module being invoked.
    :return: Nothing.
    """
    split_name = module_name.split(".")
    module_path = ".".join(split_name[:-1])
    class_name = split_name[-1]
    args = {"evaluator": evaluator}

    jb_loader.invoke_method(module_path=module_path, class_name=class_name,
                            method_name="__call__", method_args=args)


def populate_metrics(model_manager: ModelManager,
                     eval_dir_mgr: EvalDirMgr,
                     eval_output: EvaluationOutput) -> None:
    """
    Calculate metrics and populate the output results.
    :param model_manager: The Juneberry ModelManager that will be used to get data for metrics.
    :param eval_dir_mgr: The Juneberry EvalDirMgr that will be used to get data for metrics.
    :param eval_output: The Juneberry EvaluationOutput that will be populated with metrics.
    :return: None
    """
    anno_file = Path(eval_dir_mgr.get_manifest_path())
    num_annotations = coco_utils.count_annotations(anno_file)

    # Only populate metrics output if we have annotations.
    if num_annotations > 0:
        m = Metrics.create_with_filesystem_managers(model_manager,
                                                    eval_dir_mgr)
        eval_output.results.metrics.bbox = m.as_dict()
        eval_output.results.metrics.bbox_per_class = m.mAP_per_class

        for k, v in eval_output.results.metrics.bbox.items():
            logger.info(k + " = " + str(v))

        for k, v in eval_output.results.metrics.bbox_per_class.items():
            logger.info(k + " = " + str(v))
    else:
        logger.info(
            "There are no annotations; not using Metrics class to populate metrics output.")
