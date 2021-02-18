#! /usr/bin/env python3

"""
Unit tests for the training config.
"""

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

from juneberry.config.training import TrainingConfig
import juneberry.config.training as jb_training


def make_basic_config():
    return {
        "batchSize": 16,
        "dataSetConfigPath": "path/to/data/set",
        "epochs": 50,
        "formatVersion": jb_training.FORMAT_VERSION,
        "platform": "pytorch",
        "modelArchitecture": {
            "module": "sample.module",
            "args": {"num_classes": 1000}
        },
        "seed": "optional random seed",
        "timestamp": "optional ISO time stamp for when this was generated generated",
        "validation": {
            "algorithm": "randomFraction",
            "arguments": {
                "seed": 1234,
                "fraction": 0.5
            }
        }
    }


def make_validation_fraction(seed=1234, fraction=0.5):
    return {
        "validation": {
            "algorithm": "randomFraction",
            "arguments": {
                "seed": seed,
                "fraction": fraction
            }
        }
    }


def make_optional_args():
    return {
        "description": "Test description",
        "modelArchitecture": {
            "previousModel": "OPTIONAL: name of model directory from which to load weights before training.",
            "previousModelVersion": "OPTIONAL: version of the model directory from which to load weights.",
        }
    }


def make_pytorch_args():
    return {
        "deterministic": True,
        "lossFunction": "torch.nn.CrossEntropyLoss",
        "lossArgs": {},
        "optimizer": "torch.optim.SGD",
        "optimizerArgs": {"lr": 0.01},
        "lrSchedule": "MultiStepLR",
        "lrScheduleArgs": {
            "milestones": [3, 5],
            "gamma": 0.5
        },
        "accuracyFunction": "sklearn.metrics.accuracy_score",
        "accuracyArgs": {"normalize": True},
    }


def make_transforms():
    return {
        "trainingTransforms": [
            {
                "fullyQualifiedClass": "my.fqg",
                "kwargs": {"arg1": "hello"}
            }
        ],
        "predictionTransforms": [
            {
                "fullyQualifiedClass": "my.fqg",
                "kwargs": {"arg1": "hello"}
            }
        ]
    }


def test_basic_loading():
    # Do a basic construction and see that all the keys we identified
    config = make_basic_config()
    config.update(make_optional_args())
    config['pytorch'] = make_pytorch_args()
    config['transforms'] = make_transforms()

    tc = TrainingConfig('modelName', config)

    # For basic construction (since we don't have anything weird) everything should be passed through.
    for key in config.keys():
        if key in config:
            assert tc[key] == config[key]

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
#         self.assertIn("ERROR:root", cm.output[0])
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
