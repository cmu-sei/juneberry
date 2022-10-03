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

import juneberry.evaluation.utils as jb_eval_utils
import juneberry.filesystem as jb_fs
from juneberry.onnx.evaluator import Evaluator
from juneberry.onnx.utils import ONNXPlatformDefinitions
import juneberry.pytorch.evaluation.utils as jb_pytorch_eval_utils

logger = logging.getLogger(__name__)


class OnnxEvaluationOutput:
    """
    This is the default ONNX evaluation class used for formatting raw classification evaluation data
    in Juneberry.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to format the raw evaluation data. The
        result of the process is the evaluator.output attribute will contain JSON-friendly data, which will
        then be written to a file.
        :param evaluator: The Evaluator object managing the evaluation.
        :return: Nothing.
        """

        # Perform the common eval output processing steps for a classifier.
        jb_eval_utils.prepare_classification_eval_output(evaluator)

        # Calculate the hash of the model that was used to conduct the evaluation.
        model_path = evaluator.model_manager.get_model_path(ONNXPlatformDefinitions())
        evaluated_model_hash = jb_fs.generate_file_hash(model_path)

        # If the model Juneberry trained the model, a hash would have been calculated after training.
        # Compare that hash (if it exists) to the hash of the model being evaluated.
        jb_eval_utils.verify_model_hash(evaluator, evaluated_model_hash, onnx=True)

        # If requested, get the top K classes predicted for each input.
        if evaluator.top_k:
            jb_pytorch_eval_utils.top_k_classifications(evaluator, evaluator.eval_dataset_config.label_names)

        # Save the predictions portion of the evaluation output to the appropriate file.
        logger.info(f"Saving predictions to {evaluator.eval_dir_mgr.get_predictions_path()}")
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())

        # Save the metrics portion of the evaluation output to the appropriate file.
        logger.info(f"Saving metrics to {evaluator.eval_dir_mgr.get_metrics_path()}")
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
