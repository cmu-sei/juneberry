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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sys

from juneberry.config.training_output import TrainingOutput
import juneberry.evaluation.utils as jb_eval_utils
import juneberry.filesystem as jbfs
from juneberry.pytorch.evaluator import Evaluator
import juneberry.pytorch.eval_utils as jb_pytorch_eval_utils

logger = logging.getLogger(__name__)


class PyTorchEvaluationProcedure:
    """
    This is the default PyTorch evaluation class used for evaluating data in Juneberry.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to conduct the evaluation. The result
        of the process is raw evaluation data.
        :param evaluator: The Evaluator object managing the evaluation.
        :return: Nothing.
        """

        # Perform the evaluation; saving the raw data to the correct evaluator attribute.
        evaluator.raw_output = jb_eval_utils.predict_classes(evaluator.eval_loader, evaluator.model, evaluator.device)


class PyTorchEvaluationOutput:
    """
    This is the default PyTorch evaluation class used for formatting raw evaluation data in Juneberry.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to format the raw evaluation data. At the
        end of this call, the evaluator.output attribute will contain JSON-friendly data which will then be
        written to a file.
        :param evaluator: The Evaluator object managing the evaluation.
        :return: Nothing.
        """

        # Add the predicted labels for each image to the output.
        labels = [item[1] for item in evaluator.eval_name_targets]
        evaluator.output.results.labels = labels

        # Diagnostic for accuracy
        # TODO: Switch to configurable and standard accuracy
        is_binary = evaluator.eval_dataset_config.num_model_classes == 2
        predicted_classes = jb_eval_utils.continuous_predictions_to_class(evaluator.raw_output, is_binary)

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
            evaluator.raw_output = jb_eval_utils.binary_to_classes(evaluator.raw_output)

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
            jb_pytorch_eval_utils.top_k_classifications(evaluator, evaluator.eval_dataset_config.label_names)

        # Save the predictions portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())
        logger.info(f"Saving predictions to {evaluator.eval_dir_mgr.get_predictions_path()}")

        # Save the metrics portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
        logger.info(f"Saving metrics to {evaluator.eval_dir_mgr.get_metrics_path()}")