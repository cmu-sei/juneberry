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

"""
Unit tests for operations related to model evaluation.
"""
import functools
import inspect
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.detectron2.dt2_evaluator import Detectron2Evaluator
from juneberry.evaluation.evaluator import Evaluator
from juneberry.evaluation.onnx_evaluator import OnnxEvaluator
from juneberry.evaluation.util import create_evaluator
from juneberry.lab import Lab
from juneberry.mmdetection.mmd_evaluator import MMDEvaluator
from juneberry.pytorch.evaluation.pytorch_evaluator import PytorchEvaluator
from juneberry.tensorflow.evaluator import TFEvaluator


class EvalTestHelper:
    def __init__(self, tmp_path):
        self.model_config = ModelConfig()

        # The TensorFlow evaluator needs these to be defined.
        self.model_config.model_architecture = {'args': {'img_height': 0, 'img_width': 0, 'channels': 0}}

        self.lab = Lab(workspace=tmp_path / 'workspace', data_root=tmp_path / 'data_root')
        self.dataset = DatasetConfig()
        self.model_manager = self.lab.model_manager("test")
        self.eval_dir_mgr = self.model_manager.get_eval_dir_mgr("test_dataset")
        self.eval_dir_mgr.setup()
        self.eval_options = SimpleNamespace()

    def build_evaluator(self, platform: str, procedure: str):
        self.model_config.platform = platform
        self.model_config.evaluation_procedure = procedure

        return create_evaluator(self.model_config, self.lab, self.dataset, self.model_manager, self.eval_dir_mgr,
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


class EvaluatorHarness(Evaluator):
    def __init__(self, evaluator, eval_options):
        super().__init__(evaluator.model_config, evaluator.lab, evaluator.eval_dataset_config, evaluator.model_manager,
                         evaluator.eval_dir_mgr, eval_options)

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


def test_get_eval_procedure_class():
    import inspect
    from juneberry.evaluation.util import get_eval_procedure_class

    eval_proc_str = "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationProcedure"
    eval_class = get_eval_procedure_class(eval_proc_str)

    assert inspect.isclass(eval_class)
    assert hasattr(eval_class, 'establish_evaluator')


def test_pytorch_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)

    platform = "pytorch"
    procedure = "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationProcedure"
    evaluator = helper.build_evaluator(platform, procedure)

    assert isinstance(evaluator, PytorchEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_onnx_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)

    platform = "pytorch"
    procedure = "juneberry.evaluation.evals.onnx.OnnxEvaluationProcedure"
    evaluator = helper.build_evaluator(platform, procedure)

    assert isinstance(evaluator, OnnxEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_detectron2_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)

    platform = "detectron2"
    evaluator = helper.build_evaluator(platform, "")

    assert isinstance(evaluator, Detectron2Evaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_mmdetection_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)

    platform = "mmdetection"
    evaluator = helper.build_evaluator(platform, "")

    assert isinstance(evaluator, MMDEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()


def test_tensorflow_evaluator(tmp_path):
    helper = EvalTestHelper(tmp_path)

    platform = "tensorflow"
    evaluator = helper.build_evaluator(platform, "")

    assert isinstance(evaluator, TFEvaluator)

    eval_harness = EvaluatorHarness(evaluator, helper.eval_options)
    eval_harness.perform_evaluation()
    eval_harness.check_calls()
