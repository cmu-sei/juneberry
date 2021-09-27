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

from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.detectron2.dt2_evaluator import Detectron2Evaluator
from juneberry.evaluation.onnx_evaluator import OnnxEvaluator
from juneberry.evaluation.util import create_evaluator
from juneberry.lab import Lab
from juneberry.mmdetection.mmd_evaluator import MMDEvaluator
from juneberry.pytorch.evaluation.pytorch_evaluator import PytorchEvaluator


def test_get_eval_procedure_class():
    import inspect
    from juneberry.evaluation.util import get_eval_procedure_class

    eval_proc_str = "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationProcedure"
    eval_class = get_eval_procedure_class(eval_proc_str)

    assert inspect.isclass(eval_class)
    assert hasattr(eval_class, 'establish_evaluator')


def test_build_correct_evaluator(tmp_path):
    model_config = ModelConfig()
    lab = Lab(workspace=tmp_path / 'workspace', data_root=tmp_path / 'data_root')
    dataset = DatasetConfig()
    model_manager = lab.model_manager("test")
    eval_dir_mgr = model_manager.get_eval_dir_mgr("test_dataset")
    eval_dir_mgr.setup()
    eval_options = SimpleNamespace()

    model_config.platform = "pytorch"
    model_config.evaluation_procedure = "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationProcedure"
    evaluator = create_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)
    assert isinstance(evaluator, PytorchEvaluator)

    model_config.evaluation_procedure = "juneberry.evaluation.evals.onnx.OnnxEvaluationProcedure"
    evaluator = create_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)
    assert isinstance(evaluator, OnnxEvaluator)

    model_config.platform = "detectron2"
    evaluator = create_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)
    assert isinstance(evaluator, Detectron2Evaluator)

    model_config.platform = "mmdetection"
    evaluator = create_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)
    assert isinstance(evaluator, MMDEvaluator)


def test_pytorch_evaluator():
    assert 1 == 1


def test_onnx_evaluator():
    assert 1 == 1


def test_detectron2_evaluator():
    assert 1 == 1


def test_mmdetection_evaluator():
    assert 1 == 1
