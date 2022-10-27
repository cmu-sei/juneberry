#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

import logging
import sys
from types import SimpleNamespace
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from tqdm import tqdm

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig, Plugin
from juneberry.config.training_output import TrainingOutput
from juneberry.evaluation.evaluator import EvaluatorBase as Evaluator
from juneberry.filesystem import EvalDirMgr, ModelManager
import juneberry.metrics.classification.metrics_manager as mm
from juneberry.lab import Lab
import juneberry.loader as jb_loader

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

    kw_args = model_config.evaluator.kwargs
    if kw_args is None:
        kw_args = {}
    fqcn = model_config.evaluator.fqcn

    # If kw_args doesn't contain a required arg, substitute in the local variable for that kw_arg.
    reqd_args = ['lab', 'model_config', 'dataset', 'model_manager', 'eval_dir_mgr', 'eval_options', 'log_file']
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
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        output = model(local_batch)

        with torch.set_grad_enabled(False):
            if all_outputs is None:
                all_outputs = output.detach().cpu().numpy()
            else:
                all_outputs = np.concatenate((all_outputs, output.detach().cpu().numpy()))

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

    jb_loader.invoke_method(module_path=module_path, class_name=class_name, method_name="__call__", method_args=args)


def prepare_classification_eval_output(evaluator: Evaluator):
    """
    This function is responsible for performing some of the common preparation tasks when working
    to format the evaluation output from a classifier.
    :param evaluator: A Juneberry Evaluator object.
    :return: Nothing.
    """

    # Add the predicted labels for each image to the output.
    labels = [item[1] for item in evaluator.eval_name_targets]
    evaluator.output.results.labels = labels

    is_binary = evaluator.eval_dataset_config.num_model_classes == 2

    # Calculate the metrics and add to output.
    evaluator.output.results.metrics.classification = {}
    metrics_mgr = mm.MetricsManager(evaluator.metrics_plugins)
    metrics = metrics_mgr(labels, evaluator.raw_output, is_binary)
    for k, v in metrics.items():
        evaluator.output.results.metrics.classification[k] = v
        logger.info(f"{k}: {v:.4f}")

    # Save these as two classes if binary so it's consistent with other outputs.
    if is_binary:
        evaluator.raw_output = binary_to_classes(evaluator.raw_output)

    # Add the raw prediction data to the output.
    evaluator.output.results.predictions = evaluator.raw_output

    # Add the dataset mapping and the number of classes the model is aware of to the output.
    evaluator.output.options.dataset.classes = evaluator.eval_dataset_config.label_names
    evaluator.output.options.model.num_classes = evaluator.eval_dataset_config.num_model_classes


def verify_model_hash(evaluator: Evaluator, evaluated_model_hash, onnx=False):
    """
    This function is responsible for checking the hash of the model being evaluated. When a model is
    trained in Juneberry, the hash of the model is stored in the training output.json. This function
    will compare the hash of the model being evaluated to the hash of the model that was trained, if
    the hash of the trained model can be retrieved from the training output.
    :param evaluator: A Juneberry Evaluator object.
    :param evaluated_model_hash: The hash of the model being evaluated.
    :param onnx: A boolean which controls whether to retrieve the hash of the ONNX model, or the
    non-ONNX model hash.
    returns: Nothing.
    """
    # If Juneberry was used to train the model, retrieve the hash from the training output file
    # and verify the hash matches the hash of the model used to evaluate the data.
    training_output_file_path = evaluator.model_manager.get_training_out_file()
    if training_output_file_path.exists():
        training_output = TrainingOutput.load(training_output_file_path)

        # Determine which hash to retrieve based on the ONNX boolean.
        if onnx:
            hash_from_output = training_output.results.onnx_model_hash
        else:
            hash_from_output = training_output.results.model_hash

        logger.info(f"Model hash retrieved from training output: {hash_from_output}")

        # Perform the hash comparison.
        if hash_from_output != evaluated_model_hash:
            logger.error(f"Hash of the model that was just evaluated: '{evaluated_model_hash}'")
            logger.error(f"The hash of the model used for evaluation does NOT match the hash in the training "
                         f"output file. Exiting.")
            sys.exit(-1)
        else:
            logger.info(f"Hashes match! Hash of the evaluated model: {evaluated_model_hash}")

    # Add the hash of the model used for evaluation to the Evaluator output.
    evaluator.output.options.model.hash = evaluated_model_hash


# TODO it would be better if this was in an OD-specific superclass of evaluator
#   as a more general get_default_metrics_config
def get_default_od_metrics_config() -> List[Plugin]:
    default_metrics_config = {
        "fqcn": "juneberry.metrics.objectdetection.brambox.metrics.Coco",
        "kwargs": {
            "iou_threshold": 0.5,
            "max_det": 100,
            "tqdm": False
        }
    }
    default_summary_config = {
        "fqcn": "juneberry.metrics.objectdetection.brambox.metrics.Summary",
        "kwargs": {
            "iou_threshold": 0.5,
            "tp_threshold": 0.8
        }
    }
    default_tide_config = {
        "fqcn": "juneberry.metrics.objectdetection.brambox.metrics.Tide",
        "kwargs": {
            "pos_thresh": 0.5,
            "bg_thresh": 0.5,
            "max_det": 100,
            "area_range_min": 0,
            "area_range_max": 100000,
            "tqdm": False
        }
    }

    return [
        Plugin.from_dict(default_metrics_config),
        Plugin.from_dict(default_summary_config),
        Plugin.from_dict(default_tide_config),
    ]


# TODO it would be better if this was in an OD-specific superclass of evaluator
#   as a more general get_default_metrics_formatter
def get_default_od_metrics_formatter() -> Plugin:
    default_metrics_formatter = {
        "fqcn": "juneberry.metrics.objectdetection.brambox.format.DefaultFormatter",
        "kwargs": {
        }
    }
    return Plugin.from_dict(default_metrics_formatter)
