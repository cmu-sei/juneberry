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
import torch
from torch import FloatTensor
from torch.nn.functional import softmax

from juneberry.evaluation import util as jb_eval_utils
from juneberry.evaluation.evaluator import Evaluator
from juneberry.evaluation.util import logger


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
