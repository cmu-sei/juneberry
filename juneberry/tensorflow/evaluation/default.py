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
import juneberry.filesystem as jb_fs
from juneberry.tensorflow.evaluation.evaluator import Evaluator
from juneberry.tensorflow.utils import TensorFlowPlatformDefinitions

logger = logging.getLogger(__name__)


class TFEvaluationProcedure:
    """
    Attempt at a TensorFlow eval procedure.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to conduct the evaluation. The result
        of the process is raw evaluation data.
        :param evaluator: The Evaluator object managing the evaluation.
        :return: Nothing.
        """

        # loss, acc
        logger.info("Evaluating...")
        evaluator.eval_results = evaluator.model.evaluate(evaluator.eval_loader)

        metrics_results = "  "
        for idx, name in enumerate(evaluator.model.metrics_names):
            metrics_results = metrics_results + f"{name}={evaluator.eval_results[idx]}, "
        metrics_results = metrics_results[:-2]  # remove trailing comma from output
        logger.info(metrics_results)

        logger.info(f"...generating predictions...")
        evaluator.predictions = evaluator.model.predict(evaluator.eval_loader)
        logger.info(f"...evaluation complete.")


class TFEvaluationOutput:
    """
    This is the default TensorFlow evaluation class used for formatting raw evaluation data in Juneberry.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to format the raw evaluation data. The
        result of the process is the evaluator.output attribute will contain JSON-friendly data, which will
        then be written to a file.
        :param evaluator: The Evaluator object managing the evaluation.
        :return: Nothing.
        """

        # Calculate the hash of the model that was used to conduct the evaluation.
        model_path = evaluator.model_manager.get_model_path(TensorFlowPlatformDefinitions())
        evaluated_model_hash = jb_fs.generate_file_hash(model_path)

        # If the model Juneberry trained the model, a hash would have been calculated after training.
        # Compare that hash (if it exists) to the hash of the model being evaluated.
        jb_eval_utils.verify_model_hash(evaluator, evaluated_model_hash)

        # Add the dataset mapping and the number of classes the model is aware of to the output.
        evaluator.output.options.dataset.classes = evaluator.eval_dataset_config.label_names
        evaluator.output.options.model.num_classes = evaluator.eval_dataset_config.num_model_classes

        evaluator.output.results.labels = evaluator.eval_labels

        if evaluator.output.results.metrics.classification == None:
            evaluator.output.results.metrics.classification = {}

        for idx, name in enumerate(evaluator.model.metrics_names):
            evaluator.output.results.metrics.classification[name] = evaluator.eval_results[idx]

        evaluator.output.results.predictions = evaluator.predictions.tolist()
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
