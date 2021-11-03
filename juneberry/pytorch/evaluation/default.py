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

import json
import logging
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sys
from torch import FloatTensor
from torch.nn.functional import softmax

from juneberry.config.training_output import TrainingOutput
import juneberry.filesystem as jbfs
from juneberry.pytorch.evaluator import PytorchEvaluator
import juneberry.pytorch.utils as pyt_utils

logger = logging.getLogger(__name__)


class DefaultEvaluationProcedure:
    """
    This is the default Pytorch evaluation class used for evaluating data in Juneberry.
    """
    def __call__(self, evaluator: PytorchEvaluator):
        """
        When called, this method uses the attributes of the evaluator to conduct the evaluation. The result
        of the process is raw evaluation data.
        :param evaluator: The PytorchEvaluator object managing the evaluation.
        :return: Nothing.
        """

        # Perform the evaluation; saving the raw data to the correct evaluator attribute.
        evaluator.raw_output = pyt_utils.predict_classes(evaluator.eval_loader, evaluator.model, evaluator.device)


class DefaultEvaluationOutput:
    """
    This is the default Pytorch evaluation class used for formatting raw evaluation data in Juneberry.
    """
    def __call__(self, evaluator: PytorchEvaluator):
        """
        When called, this method uses the attributes of the evaluator to format the raw evaluation data. The
        result of the process is the evaluator.output attribute will contain JSON-friendly data, which will
        then be written to a file.
        :param evaluator: The PytorchEvaluator object managing the evaluation.
        :return: Nothing.
        """

        # Add the predicted labels for each image to the output.
        labels = [item[1] for item in evaluator.eval_name_targets]
        evaluator.output.results.labels = labels

        # Diagnostic for accuracy
        # TODO: Switch to configurable and standard accuracy
        is_binary = evaluator.eval_dataset_config.num_model_classes == 2
        predicted_classes = pyt_utils.continuous_predictions_to_class(evaluator.raw_output, is_binary)

        # Calculate the accuracy and add it to the output.
        logger.info(f"Computing the accuracy.")
        accuracy = accuracy_score(labels, predicted_classes)
        evaluator.output.results.metrics.accuracy = accuracy

        # Calculate the balanced accuracy and add it to the output.
        logger.info(f"Computing the balanced accuracy.")
        balanced_acc = balanced_accuracy_score(labels, predicted_classes)
        evaluator.output.results.metrics.balanced_accuracy = balanced_acc

        # Log the the accuracy values.
        logger.info(f"******          Accuracy: {accuracy:.4f}")
        logger.info(f"****** Balanced Accuracy: {balanced_acc:.4f}")

        # Save these as two classes if binary so it's consistent with other outputs.
        if is_binary:
            evaluator.raw_output = pyt_utils.binary_to_classes(evaluator.raw_output)

        # Add the raw prediction data to the output.
        evaluator.output.results.predictions = evaluator.raw_output

        # Add the dataset mapping and the number of classes the model is aware of to the output.
        evaluator.output.options.dataset.classes = evaluator.eval_dataset_config.label_names
        evaluator.output.options.model.num_classes = evaluator.eval_dataset_config.num_model_classes

        # Calculate the hash of the model that was used to conduct the evaluation.
        evaluated_model_hash = jbfs.generate_file_hash(evaluator.model_manager.get_pytorch_model_path())

        # If Juneberry was used to train the model, we can retrieve the hash from the training output file
        # and verify that the hash matches the model we used to evaluate the data.
        training_output_file_path = evaluator.model_manager.get_training_out_file()
        if training_output_file_path.is_file():
            training_output = TrainingOutput.load(training_output_file_path)
            hash_from_output = training_output.results.model_hash
            if hash_from_output != evaluated_model_hash:
                logger.error(f"The hash of the model used for evaluation does NOT match the hash in the training "
                             f"output file. EXITING.")
                logger.error(f"Expected: '{hash_from_output}' Found: '{evaluated_model_hash}'")
                sys.exit(-1)

        # Add the hash of the model used for evaluation to the output.
        evaluator.output.options.model.hash = evaluated_model_hash

        # If requested, get the top K classes predicted for each input.
        if evaluator.top_k:
            top_k_classifications(evaluator, evaluator.eval_dataset_config.label_names)

        # Save the predictions portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())
        logger.info(f"Saving predictions to {evaluator.eval_dir_mgr.get_predictions_path()}")

        # Save the metrics portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
        logger.info(f"Saving metrics to {evaluator.eval_dir_mgr.get_metrics_path()}")


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
    evaluator.output.results.classifications = classify_inputs(evaluator.eval_name_targets,
                                                               evaluator.raw_output,
                                                               evaluator.top_k,
                                                               dataset_mapping,
                                                               model_mapping)

    logger.info(f"Classified {len(evaluator.output.results.classifications)} inputs.")


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
