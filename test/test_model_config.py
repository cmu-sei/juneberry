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
Unit tests for the training config.
"""

import json
from pathlib import Path

from juneberry.config.model import ModelConfig
import juneberry.filesystem as jbfs


def make_basic_config_old():
    return {
        "batchSize": 16,
        "trainingDatasetConfigPath": "path/to/data/set",
        "epochs": 50,
        "formatVersion": "0.1.0",
        "platform": "pytorch",
        "modelArchitecture": {
            "module": "sample.module",
            "args": {"num_classes": 1000}
        },
        "seed": 6789,
        "timestamp": "optional ISO time stamp for when this was generated generated",
        "validation": {
            "algorithm": "randomFraction",
            "arguments": {
                "seed": 1234,
                "fraction": 0.5
            }
        }
    }


def make_basic_config():
    return {
        "batch_size": 16,
        "training_dataset_config_path": "path/to/data/set",
        "epochs": 50,
        "format_version": ModelConfig.FORMAT_VERSION,
        "platform": "pytorch",
        "model_architecture": {
            "module": "sample.module",
            "args": {"num_classes": 1000}
        },
        "seed": 1234,
        "timestamp": "optional ISO time stamp for when this was generated generated",
        "validation": {
            "algorithm": "random_fraction",
            "arguments": {
                "seed": 1234,
                "fraction": 0.5
            }
        }
    }


def add_optional_args(config):
    config["description"] = "Test description"
    config["model_architecture"][
        'previous_model'] = "OPTIONAL: name of model directory from which to load weights before training."
    config["model_architecture"][
        'previous_model_version'] = "OPTIONAL: version of the model directory from which to load weights."


def make_pytorch_args():
    return {
        "deterministic": True,
        "loss_fn": "torch.nn.CrossEntropyLoss",
        "loss_args": {},
        "optimizer": "torch.optim.SGD",
        "optimizer_args": {"lr": 0.01},
        "lr_schedule": "MultiStepLR",
        "lr_schedule_args": {
            "milestones": [3, 5],
            "gamma": 0.5
        },
        "accuracy_fn": "sklearn.metrics.accuracy_score",
        "accuracy_args": {"normalize": True},
    }


def make_transforms():
    return {
        "training_transforms": [
            {
                "fqcn": "my.fqg",
                "kwargs": {"arg1": "hello"}
            }
        ],
        "evaluation_transforms": [
            {
                "fqcn": "my.fqg",
                "kwargs": {"arg1": "hello"}
            }
        ]
    }


def assert_is_subset(expected, test):
    if isinstance(expected, (list, tuple)):
        for i1, i2 in zip(expected, test):
            assert_is_subset(i1, i2)
    elif isinstance(expected, dict):
        for k in expected.keys():
            assert k in test
            assert_is_subset(expected[k], test[k])
    else:
        assert expected == test


def test_basic_loading(tmp_path):
    # Do a basic construction and see that all the keys we identified
    config = make_basic_config()
    add_optional_args(config)
    config.update(make_transforms())
    config['pytorch'] = make_pytorch_args()

    config_path = Path(tmp_path, "config.json")

    with open(config_path, 'w') as out_file:
        json.dump(config, out_file, indent=4)

    data = jbfs.load_json(str(config_path))
    mc = ModelConfig.construct(data=data, file_path=str(config_path))

    mc2 = ModelConfig.load(str(config_path))

    assert mc == mc2

    assert_is_subset(config, mc)


# def test_model_upgrade(tmp_path):
#     config = make_basic_config_old()
#
#     config_path = Path(tmp_path, "old_config.json")
#     with open(config_path, 'w') as out_file:
#         json.dump(config, out_file, indent=4)
#
#     mc = ModelConfig.load_config(config_path)
#     conf_util.rekey(config, jb_model.get_camel_to_snake())
#     config['format_version'] = ModelConfig.FORMAT_VERSION
#
#     assert_is_subset(config, mc)


# def is_int(val):
#     try:
#         num = int(val)
#     except ValueError:
#         return False
#     return True
#
# class TestFormatErrors(unittest.TestCase):
#
#     # Helper method for analyzing error
#     def assert_error(self, cm, message):
#         self.assertEqual(len(cm.output), 2)
#         self.assertIn("ERROR:juneberry", cm.output[0])
#         self.assertIn(message, cm.output[0])
#
#     def test_data_type_error(self):
#         config = make_basic_config()
#         config['imageData']['sources'] = {"label": 0}
#
#         with self.assertLogs(level='ERROR') as cm:
#             tc = TrainingConfig('modelName', config)
#         self.assert_error(cm, "ValueError")


# todo: Remove testDataSet
# todo: seed is not optional
# todo: Remove convenience args
# todo: Check that get wraps Nones properly
# todo: Training initializer should
