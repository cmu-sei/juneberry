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

import juneberry.evaluation.utils as jb_eval_utils
import juneberry.filesystem as jbfs
import juneberry.pytorch.evaluation.utils as jb_pytorch_eval_utils
from juneberry.pytorch.evaluation.evaluator import Evaluator
from juneberry.pytorch.utils import PyTorchPlatformDefinitions

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

        # Perform the common eval output processing steps for a classifier.
        jb_eval_utils.prepare_classification_eval_output(evaluator)

        # Calculate the hash of the model that was used to conduct the evaluation if we have a serialized one
        model_path = evaluator.model_manager.get_model_path(PyTorchPlatformDefinitions())
        if model_path.exists():
            evaluated_model_hash = jbfs.generate_file_hash(model_path)

            # If the Juneberry trained the model, a hash would have been calculated after training.
            # Compare that hash (if it exists) to the hash of the model being evaluated.
            jb_eval_utils.verify_model_hash(evaluator, evaluated_model_hash)

        # If requested, get the top K classes predicted for each input.
        if evaluator.top_k:
            jb_pytorch_eval_utils.top_k_classifications(evaluator, evaluator.eval_dataset_config.label_names)

        # Save the predictions portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())
        logger.info(f"Saving predictions to {evaluator.eval_dir_mgr.get_predictions_path()}")

        # Save the metrics portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
        logger.info(f"Saving metrics to {evaluator.eval_dir_mgr.get_metrics_path()}")
