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
import numpy as np
from torch import split
from tqdm import tqdm
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
from juneberry.onnx.evaluator import Evaluator as OnnxEvaluatorBase
from juneberry.pytorch.evaluation.evaluator import Evaluator as PyTorchEvaluator

logger = logging.getLogger(__name__)


class Evaluator(OnnxEvaluatorBase):
    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, **kwargs):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, **kwargs)

        # If the user did not specify any classes in the evaluator kwargs, use the default classes
        # for this platform.

        # The default value for the data loader.
        if self.eval_data_loader_method is None:
            self.eval_data_loader_method = "juneberry.onnx.pytorch.DataLoader"

        # The default value for the formatting of the evaluation output.
        if self.eval_output_method is None:
            self.eval_output_method = "juneberry.onnx.default.OnnxEvaluationOutput"

        # The default value for the evaluation procedure.
        if self.eval_method is None:
            self.eval_method = "juneberry.onnx.pytorch.EvaluationProcedure"


class DataLoader:
    """
    This class is responsible for creating a data loader that can be used to evaluate an ONNX
    model that was created using a Juneberry PyTorch trainer.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the same process as a non-ONNX PyTorch evaluation to
        construct the evaluation dataloader.
        """
        # Create a PytorchEvaluator and use it to build a PyTorch dataloader for the input data.
        pyt_evaluator = PyTorchEvaluator(evaluator.model_config, evaluator.lab, evaluator.model_manager,
                                         evaluator.eval_dir_mgr, evaluator.eval_dataset_config, None)
        pyt_evaluator.obtain_dataset()
        evaluator.eval_loader = pyt_evaluator.eval_loader

        # Retrieve the labels for the input data.
        evaluator.eval_name_targets = pyt_evaluator.eval_name_targets.copy()


class EvaluationProcedure:
    """
    This class is responsible for sending the data from a PyTorch evaluation data loader into
    an onnxruntime inference session and producing the raw data for the detected objects.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method will loop through all batches in the eval data loader and send
        each image into the onnxruntime inference session. The resulting output will then be stored.
        """

        # Retrieve the proper input name to use for the inference session and the eval data loader.
        input_name = evaluator.ort_session.get_inputs()[0].name
        data_loader = evaluator.eval_loader

        # Loop through every batch in the data loader.
        for i, (batch, target) in enumerate(tqdm(data_loader)):

            # Convert each batch into a list of input data.
            input_list = []
            for item in split(batch, 1):
                input_list.append(item.data.numpy())

            # Send each item from the input list into the onnxruntime inference session and store
            # the result in evaluator.raw_output.
            for item in input_list:
                ort_out = evaluator.ort_session.run([], {input_name: item})
                ort_out = np.array(ort_out[0]).tolist()
                evaluator.raw_output.append(ort_out[0])
