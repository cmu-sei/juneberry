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

"""
Unit tests for operations related to model evaluation.
"""
import functools
import inspect
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.detectron2.evaluator import Evaluator as Detectron2Evaluator
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.evaluation.utils import create_evaluator
from juneberry.lab import Lab
from juneberry.mmdetection.evaluator import Evaluator as MMDEvaluator
from juneberry.pytorch.evaluator import Evaluator as PyTorchEvaluator
from juneberry.tensorflow.evaluator import Evaluator as TFEvaluator


class EvalTestHelper:
    def __init__(self, tmp_path):
        self.model_config = ModelConfig()

        # The TensorFlow evaluator needs these to be defined.
        self.model_config.model_architecture = {'args': {'img_height': 0, 'img_width': 0, 'channels': 0}}

        self.lab = Lab(workspace=tmp_path / 'workspace', data_root=tmp_path / 'data_root')
        self.dataset = DatasetConfig()
        self.dataset.file_path = ""
        self.model_manager = self.lab.model_manager("test")
        self.eval_dir_mgr = self.model_manager.get_eval_dir_mgr("test_dataset")
        self.eval_dir_mgr.setup()
        self.eval_options = SimpleNamespace()

    def build_evaluator(self):
        return create_evaluator(self.model_config, self.lab, self.model_manager, self.eval_dir_mgr, self.dataset,
                                self.eval_options)


def get_fn_name(fn):
    for k, v in inspect.getmembers(fn):
        if k == "__name__":
            return v
    return "Unknown"


log_step = 0


def log_func(func):
    func_name = get_fn_name(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global log_step
        # Use this to get a list of all calls in order
        # print(f">> {log_step} {func_name}")
        log_step += 1
        return func(*args, **kwargs)

    return wrapper


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

    @log_func
    def check_gpu_availability(self):
        self.step += 1

    @log_func
    def setup(self):
        self.setup_calls.append(self.step)
        self.step += 1

    @log_func
    def obtain_dataset(self):
        self.obtain_dataset_calls.append(self.step)
        self.step += 1

    @log_func
    def obtain_model(self):
        self.obtain_model_calls.append(self.step)
        self.step += 1

    @log_func
    def evaluate_data(self):
        self.evaluate_data_calls.append(self.step)
        self.step += 1

    @log_func
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
    helper = EvalTestHelper(tmp_path)
    helper.model_config.platform = "pytorch"
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, PyTorchEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_detectron2_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)
    helper.model_config.platform = "detectron2"
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, Detectron2Evaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_mmdetection_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)
    helper.model_config.platform = "mmdetection"
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, MMDEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_tensorflow_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)
    helper.model_config.platform = "tensorflow"
    evaluator = helper.build_evaluator()

    assert isinstance(evaluator, TFEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()
