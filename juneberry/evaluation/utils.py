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
import sys
from types import SimpleNamespace

import torch
from tqdm import tqdm

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab

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
        classes = np.asarray([x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int)
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
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


def get_eval_procedure_class(procedure_name: str):
    """
    This function is responsible for retrieving the class indicated by the evaluation_procedure property of the
    model config.
    :param procedure_name: A string corresponding to the value of the evaluation_procedure property in the
    model config.
    :return: The class indicated by the evaluation_procedure property of the model config.
    """

    # Some string manipulation to separate the class name from the rest of the module path.
    split_name = procedure_name.split(".")
    class_name = split_name[-1]
    module_path = ".".join(split_name[:-1])

    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def create_evaluator(model_config: ModelConfig, lab: Lab, dataset: DatasetConfig, model_manager: ModelManager,
                     eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace):
    """
    This function is responsible for creating the correct evaluator object based on the contents of the ModelConfig.
    :param model_config: The loaded Juneberry ModelConfig which determines which Evaluator to build.
    :param lab: The Juneberry Lab that will be used to build the Evaluator.
    :param dataset: The Juneberry DatasetConfig that will be used to build the Evaluator.
    :param model_manager: The Juneberry ModelManager that will be used to build the Evaluator.
    :param eval_dir_mgr: The Juneberry EvalDirMgr that will be used to build the Evaluator.
    :param eval_options: A SimpleNamespace of different eval options that will be used to build the Evaluator.
    """
    # Return an evaluator for the PyTorch platform.
    if model_config.platform in ['pytorch', 'pytorch_privacy', 'tensorflow']:
    # if model_config.platform == "pytorch" or model_config.platform == "pytorch_privacy":
        # Fetch the desired evaluation procedure.
        eval_proc_name = model_config.evaluation_procedure

        # Import the class from the desired evaluation procedure.
        imported_class = get_eval_procedure_class(eval_proc_name)

        # Build the appropriate evaluator using the establish_evaluator method defined for the imported class.
        return imported_class.establish_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    # Return an evaluator for the detectron2 platform.
    elif model_config.platform == "detectron2":
        from juneberry.detectron2.evaluator import Detectron2Evaluator
        return Detectron2Evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    # Return an evaluator for the mmdetection platform.
    elif model_config.platform == "mmdetection":
        from juneberry.mmdetection.evaluator import MMDEvaluator
        return MMDEvaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    # Return an evaluator for the tensorflow platform.
    elif model_config.platform == "tensorflow":
        from juneberry.tensorflow.evaluator import TFEvaluator
        return TFEvaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    # Handle cases where there is no Evaluator for the requested platform.
    else:
        logger.error(f"Evaluation is not currently supported for the requested platform: {model_config.platform}")
        sys.exit(-1)


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
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        output = model(local_batch)

        with torch.set_grad_enabled(False):
            if all_outputs is None:
                all_outputs = output.detach().cpu().numpy()
            else:
                all_outputs = np.concatenate((all_outputs, output.detach().cpu().numpy()))

    return all_outputs.tolist()
