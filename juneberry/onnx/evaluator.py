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
import onnx
import onnxruntime as ort
import sys
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evaluator import EvaluatorBase
import juneberry.evaluation.utils as jb_eval_utils
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.utils as jb_utils


logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    """
        This subclass is the ONNX-specific version of the Evaluator.
        """

    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, **kwargs):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, **kwargs)

        self.input_data = []
        self.onnx_model = None
        self.ort_session = None
        self.raw_output = []
        self.eval_loader = None

    def setup(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for setting up the Evaluator.
        :return: Nothing.
        """

        # TODO: Shouldn't this be done in the lab??

        if self.model_config.hints is not None and 'num_workers' in self.model_config.hints.keys():
            num_workers = self.model_config.hints.num_workers
            logger.warning(f"Overriding number of workers. Found {num_workers} in ModelConfig")
            self.lab.num_workers = num_workers

        # Set the seeds using the value from the ModelConfig.
        jb_utils.set_seeds(self.model_config.seed)

        # Use default values if they were not provided in the model config.
        if self.eval_method is None:
            self.eval_method = "juneberry.onnx.default.OnnxEvaluationProcedure"
        if self.eval_output_method is None:
            self.eval_output_method = "juneberry.onnx.default.OnnxEvaluationOutput"

        logger.info(f"ONNX Evaluator setup steps are complete.")

    def obtain_dataset(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the
        dataset to be evaluated. The input_data is expected to be a list of individual tensors,
        where each tensor will be fed in to the evaluation procedure, one at a time.
        :return: Nothing.
        """

        # TODO: I think there's a risk here if the datasets are too large to fit in memory.
        #  self.input_data could end up being very large.

        # If a PyTorch model is being evaluated, create a separate PyTorch specific evaluator
        # and use it to construct a PyTorch dataloader. Once the dataloader exists, convert it
        # into the format that ONNX expects.
        if self.model_config.platform == "pytorch":
            from juneberry.pytorch.evaluation.evaluator import Evaluator

            # Create a PytorchEvaluator and use it to build a PyTorch dataloader for the input data.
            evaluator = Evaluator(self.lab, self.model_config, self.model_manager, self.eval_dir_mgr,
                                  self.eval_dataset_config, None)
            evaluator.obtain_dataset()
            self.eval_loader = evaluator.eval_loader

            # Retrieve the labels for the input data.
            self.eval_name_targets = evaluator.eval_name_targets.copy()

        # This bit will be responsible for converting the TensorFlow input data into the format ONNX expects.
        elif self.model_config.platform == "tensorflow":
            from juneberry.tensorflow.evaluation.evaluator import Evaluator
            evaluator = Evaluator(self.model_config, self.lab, self.model_manager, self.eval_dir_mgr,
                                  self.eval_dataset_config, None)
            evaluator.obtain_dataset()
            self.eval_loader = evaluator.eval_loader

            self.eval_name_targets = evaluator.eval_labels
            self.eval_name_targets = [('', x) for x in self.eval_name_targets]

        # Handle cases where the model platform does not support an ONNX evaluation.
        else:
            logger.info(f"ONNX evaluations are currently NOT supported for the {self.model_config.platform} platform.")
            sys.exit(-1)

    def obtain_model(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the model
        to be evaluated.
        :return: Nothing.
        """
        # Load the ONNX model.
        self.onnx_model = onnx.load(self.model_manager.get_onnx_model_path())

        # Check that the ONNX model is well formed.
        onnx.checker.check_model(self.onnx_model)

        # TODO: Decide if this graph adds any value to the evaluation process.
        # logger.info(f"Graph of the ONNX model:\n{onnx.helper.printable_graph(self.onnx_model.graph)}")

    def evaluate_data(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for feeding the evaluation
        dataset into the model and obtaining the raw evaluation data. That process is usually defined in some
        external method, usually found in juneberry.evaluation.evals.
        :return: Nothing.
        """

        self.ort_session = ort.InferenceSession(str(self.model_manager.get_onnx_model_path()))

        if self.dryrun:
            logger.info(f"Dry run complete.")
            sys.exit(0)

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
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")

        jb_eval_utils.invoke_evaluator_method(self, self.eval_output_method)

        logger.info(f"EVALUATION data has been formatted.")
