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

import onnx
import onnxruntime as ort

from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evaluator import EvaluatorBase
import juneberry.evaluation.utils as jb_eval_utils
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
from juneberry.onnx.utils import ONNXPlatformDefinitions
import juneberry.utils as jb_utils

logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    """
    This subclass is the ONNX-specific version of the Evaluator.
    """

    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, **kwargs):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, **kwargs)

        self.onnx_model = None
        self.ort_session = None
        self.raw_output = []
        self.eval_loader = None
        self.graph_model = eval_options.onnx_graph

    # ==========================================================================

    @classmethod
    def get_platform_defs(cls):
        return ONNXPlatformDefinitions()

    # ==========================================================================

    @classmethod
    def get_default_metric_value(cls, eval_data: EvaluationOutput):
        """ :return: The value of the Evaluator's default metric as found in the results structure """
        return eval_data.results.metrics.classification["accuracy"], "accuracy"

    # ==========================================================================

    def setup(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for setting up the Evaluator.
        :return: Nothing.
        """

        # Set the seeds using the value from the ModelConfig.
        jb_utils.set_seeds(self.model_config.seed)

        # Check that the model config defines the classes required to perform the ONNX evaluation.
        # Log an error for those that are not defined.
        error = False

        if self.eval_data_loader_method is None:
            logger.error(f"ONNX evaluations expect a 'loader' kwarg in the 'evaluator' stanza of the model "
                         f"config to define which data loader to use, but no 'loader' was found.")
            error = True
        if self.eval_method is None:
            logger.error(f"ONNX evaluations expect a 'procedure' kwarg in the 'evaluator' stanza of the model "
                         f"config to define which ONNX eval class to use, but no 'procedure' was found.")
            error = True
        if self.eval_output_method is None:
            logger.error(f"ONNX evaluations expect an 'output' kwarg in the 'evaluator' stanza of the model "
                         f"config to define which ONNX eval output formatting class to use, but no 'output' was "
                         f"found.")
            error = True

        # Check if the model directory contains an ONNX model.
        if not self.model_manager.get_model_path(ONNXPlatformDefinitions()).exists():
            logger.error(f"An ONNX evaluation was requested for model '{self.model_manager.model_name}', however "
                         f"the model directory does not contain a 'model.onnx' file.")
            error = True

        # Exit if an error was encountered.
        if error:
            logger.error(f"Exiting.")
            sys.exit(-1)

        logger.info(f"ONNX Evaluator setup steps are complete.")

    def obtain_dataset(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the
        dataset to be evaluated.
        :return: Nothing.
        """

        # Invoke the desired data loader class.
        logger.info(f"Creating a custom EVALUATION data loader according to {self.eval_data_loader_method}")
        jb_eval_utils.invoke_evaluator_method(self, self.eval_data_loader_method)
        logger.info(f"EVALUATION data loader has been created.")

    def obtain_model(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the model
        to be evaluated.
        :return: Nothing.
        """

        # Load the ONNX model.
        self.onnx_model = onnx.load(self.model_manager.get_model_path(ONNXPlatformDefinitions()))

        # Check that the ONNX model is well formed.
        onnx.checker.check_model(self.onnx_model)

        if self.graph_model:
            logger.info(f"Graph of the ONNX model:\n{onnx.helper.printable_graph(self.onnx_model.graph)}")

    def evaluate_data(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for feeding the evaluation
        dataset into the model and obtaining the raw evaluation data. That process is usually defined in some
        external method, usually found in juneberry.evaluation.evals.
        :return: Nothing.
        """

        # Establish an onnxruntime inference session.
        self.ort_session = ort.InferenceSession(str(self.model_manager.get_model_path(ONNXPlatformDefinitions())))

        # At this point the dry run should terminate since the next action is to conduct the evaluation.
        if self.dryrun:
            logger.info(f"Dry run complete.")
            sys.exit(0)

        # Invoke the desired evaluation class.
        logger.info(f"Will evaluate model '{self.model_manager.model_name}' using {self.eval_dataset_config_path}")
        logger.info(f"Generating EVALUATION data according to {self.eval_method}")
        jb_eval_utils.invoke_evaluator_method(self, self.eval_method)
        logger.info(f"EVALUATION COMPLETE.")

    def format_evaluation(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for converting the raw
        evaluation data into the format the user wants. Much like evaluate_data, the actual process is
        usually defined in some external method, typically found in juneberry.pytorch.evaluation.
        :return: Nothing.
        """

        # Invoke the desired evaluation output formatting class.
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")
        jb_eval_utils.invoke_evaluator_method(self, self.eval_output_method)
        logger.info(f"EVALUATION data has been formatted.")
