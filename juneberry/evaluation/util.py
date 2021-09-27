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
import sys
from types import SimpleNamespace

import torch
from tqdm import tqdm

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evals.onnx import logger
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.pytorch.evaluation.util as jb_pytorch_eval_utils

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
    # Build an evaluator for the PyTorch platform.
    if model_config.platform == "pytorch" or model_config.platform == "pytorch_privacy":
        # Fetch the desired evaluation procedure.
        eval_proc_name = model_config.evaluation_procedure

        # Import the class from the desired evaluation procedure.
        imported_class = get_eval_procedure_class(eval_proc_name)

        # Build the appropriate evaluator using the establish_evaluator method defined for the imported class.
        evaluator = imported_class.establish_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr,
                                                       eval_options)

    # Build an evaluator for the detectron2 platform.
    elif model_config.platform == "detectron2":
        from juneberry.detectron2.dt2_evaluator import Detectron2Evaluator
        evaluator = Detectron2Evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    # Build an evaluator for the mmdetection platform.
    elif model_config.platform == "mmdetection":
        from juneberry.mmdetection.mmd_evaluator import MMDEvaluator
        evaluator = MMDEvaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    # Handle cases where there is no Evaluator for the requested platform.
    else:
        logger.error(f"Unsupported platform: {model_config.platform}")
        sys.exit(-1)

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


def top_k_classifications(evaluator, dataset_mapping):
    """
    This function is responsible for adding the top-K classification information to the
    evaluation output.
    :param evaluator: The Juneberry Evaluator object that is managing the evaluation.
    :param dataset_mapping: The label mapping of the dataset being evaluated.
    :return: Nothing.
    """
    # Retrieve the label mapping that the MODEL is aware of. Note that this dataset mapping might be
    # different than the label mapping that the dataset is aware of. For example, a dataset might
    # only contain labels from 10 different classes in its mapping, whereas the model might be
    # aware of 1000 different labels.
    model_mapping = evaluator.model_config.label_dict

    # A logging message indicating top-K classification will occur
    class_str = "class" if evaluator.top_k == 1 else f"{evaluator.top_k} classes"
    logger.info(f"Obtaining the top {class_str} predicted for each input.")

    # Add the top-K classification information to the output.
    evaluator.output.results.classifications = jb_pytorch_eval_utils.classify_inputs(evaluator.eval_name_targets,
                                                                                     evaluator.onnx_output,
                                                                                     evaluator.top_k,
                                                                                     dataset_mapping,
                                                                                     model_mapping)

    logger.info(f"Classified {len(evaluator.output.results.classifications)} inputs.")
