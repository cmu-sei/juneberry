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

"""
Unit tests for operations related to model evaluation.
"""

from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.workspace import LabProfile
from juneberry.detectron2.evaluator import Evaluator as Detectron2Evaluator
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.evaluation.utils import create_evaluator
from juneberry.lab import Lab
from juneberry.mmdetection.evaluator import Evaluator as MMDEvaluator
from juneberry.pytorch.evaluation.evaluator import Evaluator as PyTorchEvaluator
from juneberry.tensorflow.evaluation.evaluator import Evaluator as TFEvaluator
import utils


class EvalTestHelper:
    def __init__(self, tmp_path, platform: str):
        self.model_config = ModelConfig()

        # The TensorFlow evaluator needs these to be defined.
        self.model_config.model_architecture = {'kwargs': {'img_height': 0, 'img_width': 0, 'channels': 0}}

        platform_map = {
            "pytorch": "juneberry.pytorch.evaluation.evaluator.Evaluator",
            "pytorch_privacy": "juneberry.pytorch.evaluation.evaluator.Evaluator",
            "detectron2": "juneberry.detectron2.evaluator.Evaluator",
            "mmdetection": "juneberry.mmdetection.evaluator.Evaluator",
            "tensorflow": "juneberry.tensorflow.evaluation.evaluator.Evaluator",
        }
        self.model_config.evaluator = {'fqcn': platform_map[platform]}

        self.lab = Lab(workspace=tmp_path / 'workspace', data_root=tmp_path / 'data_root')
        self.lab_profile = LabProfile()
        self.dataset = DatasetConfig()
        self.dataset.file_path = ""
        self.model_manager = self.lab.model_manager("test")
        self.eval_dir_mgr = self.model_manager.get_eval_dir_mgr("test_dataset")
        self.eval_options = SimpleNamespace()
        self.log_file = ""

    def build_evaluator(self):
        return create_evaluator(self.model_config, self.lab, self.model_manager, self.eval_dir_mgr, self.dataset,
                                self.eval_options, self.log_file)


class EvaluatorHarness(EvaluatorBase):
    def __init__(self, evaluator, eval_options):
        super().__init__(evaluator.model_config, evaluator.lab, evaluator.model_manager, evaluator.eval_dir_mgr,
                         evaluator.eval_dataset_config, eval_options)

        self.setup_calls = []
        self.obtain_dataset_calls = []
        self.obtain_model_calls = []
        self.evaluate_data_calls = []
        self.format_evaluation_calls = []

        self.step = 0

    # TODO: This really isn't checking much, just that the correct calls are made.

    @utils.log_func
    def check_gpu_availability(self):
        self.step += 1

    @utils.log_func
    def setup(self):
        self.setup_calls.append(self.step)
        self.step += 1

    @utils.log_func
    def obtain_dataset(self):
        self.obtain_dataset_calls.append(self.step)
        self.step += 1

    @utils.log_func
    def obtain_model(self):
        self.obtain_model_calls.append(self.step)
        self.step += 1

    @utils.log_func
    def evaluate_data(self):
        self.evaluate_data_calls.append(self.step)
        self.step += 1

    @utils.log_func
    def format_evaluation(self):
        self.format_evaluation_calls.append(self.step)
        self.step += 1

    def check_calls(self):
        assert self.setup_calls == [0]
        assert self.obtain_dataset_calls == [1]
        assert self.obtain_model_calls == [2]
        assert self.evaluate_data_calls == [3]
        assert self.format_evaluation_calls == [4]


def test_pytorch_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path, "pytorch")
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, PyTorchEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_detectron2_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path, "detectron2")
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, Detectron2Evaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_mmdetection_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path, "mmdetection")
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, MMDEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_tensorflow_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path, "tensorflow")
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, TFEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()
