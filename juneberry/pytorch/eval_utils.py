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
import torch
from torch import FloatTensor
from torch.nn.functional import softmax

from juneberry.evaluation import utils as jb_eval_utils
from juneberry.evaluation.evaluator import Evaluator
from juneberry.evaluation.utils import logger


def classify_inputs(eval_name_targets, predictions, classify_topk, dataset_mapping, model_mapping):
    """
    Determines the top-K predicted classes for a list of inputs.
    :param eval_name_targets: The list of input files and their true labels.
    :param predictions: The predictions that were made for the inputs.
    :param classify_topk: How many classifications we would like to show.
    :param dataset_mapping: The mapping of class integers to human readable labels that the DATASET is aware of.
    :param model_mapping: The mapping of class integers to human readable labels that the MODEL is aware of.
    :return: A list of which classes were predicted for each input.
    """
    # Some tensor operations on the predictions; softmax converts the values to percentages.
    prediction_tensor = FloatTensor(predictions)
    predict = softmax(prediction_tensor, dim=1)
    values, indices = predict.topk(classify_topk)
    values = values.tolist()
    indices = indices.tolist()

    classification_list = []

    # Each input should have a contribution to the classification list.
    for i in range(len(eval_name_targets)):

        class_list = []
        for j in range(classify_topk):
            try:
                label_name = dataset_mapping[indices[i][j]]
            except KeyError:
                label_name = model_mapping[str(indices[i][j])] if model_mapping is not None else ""

            individual_dict = {'label': indices[i][j], 'labelName': label_name, 'confidence': values[i][j]}
            class_list.append(individual_dict)

        try:
            true_label_name = dataset_mapping[eval_name_targets[i][1]]
        except KeyError:
            true_label_name = model_mapping[str(eval_name_targets[i][1])] if model_mapping is not None else ""

        classification_dict = {'file': eval_name_targets[i][0], 'actualLabel': eval_name_targets[i][1],
                               'actualLabelName': true_label_name, 'predictedClasses': class_list}
        classification_list.append(classification_dict)

    return classification_list


def top_k_classifications(evaluator: Evaluator, dataset_mapping):
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
    evaluator.output.results.classifications = classify_inputs(evaluator.eval_name_targets, evaluator.raw_output,
                                                               evaluator.top_k, dataset_mapping, model_mapping)

    logger.info(f"Classified {len(evaluator.output.results.classifications)} inputs.")


def compute_accuracy(y_pred, y_true, accuracy_function, accuracy_args, binary):
    """
    Computes the accuracy from a set of predictions where the output is rows and the classes are the columns.
    :param y_pred: The output predictions to process.
    :param y_true: The correct labels.
    :param accuracy_function: The actual function that does the computation
    :param accuracy_args: Arguments that should be passed to the accuracy function
    :param binary: True if this a binary function.
    :return: Accuracy as a float.
    """
    with torch.set_grad_enabled(False):
        # The with clause should turn off grad, but for some reason I still get the error:
        # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
        # So I am including detach. :(
        if binary:
            np_y_pred = y_pred.type(torch.DoubleTensor).cpu().detach().numpy()
            np_y_true = y_true.type(torch.DoubleTensor).unsqueeze(1).cpu().numpy()
        else:
            np_y_pred = y_pred.cpu().detach().numpy()
            np_y_true = y_true.cpu().numpy()

        # Convert the continuous predictions to single class predictions
        singular_y_pred = jb_eval_utils.continuous_predictions_to_class(np_y_pred, binary)

        # Now call the function
        return accuracy_function(y_pred=singular_y_pred, y_true=np_y_true, **accuracy_args)
