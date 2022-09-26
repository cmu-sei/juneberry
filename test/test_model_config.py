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
Unit tests for the training config.
"""

import json
from pathlib import Path

from juneberry.config.model import ModelConfig
import juneberry.filesystem as jb_fs
from utils import make_basic_model_config


def add_optional_args(config):
    config["description"] = "Test description"
    config["model_architecture"][
        'previous_model'] = "OPTIONAL: name of model directory from which to load weights before training."


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
    config = make_basic_model_config(add_transforms=True)
    add_optional_args(config)
    config['pytorch'] = make_pytorch_args()

    config_path = Path(tmp_path, "config.json")

    with open(config_path, 'w') as out_file:
        json.dump(config, out_file, indent=4)

    data = jb_fs.load_json(str(config_path))
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
#         config = make_basic_model_config()
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
