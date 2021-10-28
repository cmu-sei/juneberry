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

import unittest

from juneberry.config.experiment import ExperimentConfig


def make_basic_config():
    return {
        "description": "simple description",
        "models": [
            {
                "name": "imagenette_160x160_rgb_unit_test_pyt_resnet18",
                "tests": [
                    {
                        "tag": "pyt50",
                        "dataset_path": "data_sets/imagenette_unit_test_def.json",
                        "classify": 3
                    }
                ]
            }
        ],
        "reports": [
            {
                "description": "basic description",
                "type": 'plot_roc',
                "tests": [
                    {
                        "tag": "pyt50",
                        "classes": "0"
                    }
                ],
                "output_name": "sample_roc_1.png",
                "csv_name": "sample_roc_1.csv",
                "plot_title": "Sample ROC Plot"
            }
        ],
        "format_version": "1.5.0"
    }
    # NOTE: We provide the formatVersion manually to force an update of the unit test when
    # the version changes.


def test_config_basics():
    config = make_basic_config()

    # Most of the real functionality is in the checks
    exp_conf = ExperimentConfig.construct(config)
    assert len(config['models']) == len(exp_conf['models'])
    assert len(config['reports']) == len(exp_conf['reports'])


class TestFormatErrors(unittest.TestCase):

    def assert_error(self, cm, message, count=2):
        # Normally we have 3 errors. One for the actual and then 2 complaining about the invalid
        # print(cm.output)
        self.assertEqual(len(cm.output), count)
        self.assertIn("ERROR:juneberry", cm.output[0])
        self.assertIn(message, cm.output[0])

    # def test_version(self):
    #     config = make_basic_config()
    #     config['formatVersion'] = "0.0.0"
    #
    #     with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
    #         ExperimentConfig.construct(config)
    #     self.assert_error(log, "does not match latest version", 1)

    def test_models_missing(self):
        config = make_basic_config()
        del config['models']

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "'models' is a required property", 3)

    def test_models_non_zero(self):
        config = make_basic_config()
        config['models'] = []

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "is too short at ['models']", 3)

    def test_model_bad_name(self):
        config = make_basic_config()
        config['models'][0]['name'] = "bad name"

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "Model not found")

    def test_model_duplicate_tag(self):
        config = make_basic_config()
        config['models'][0]['tests'].append({
            "tag": "pyt50",
            "dataset_path": "data_sets/imagenette_unit_test_def.json",
        })

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "Found duplicate tag")

    def test_model_duplicate_tag_2(self):
        config = make_basic_config()
        config['models'].append({
            "name": "tabular_binary_sample",
            "tests": [
                {
                    "tag": "pyt50",
                    "dataset_path": "data_sets/imagenette_unit_test_def.json",
                }
            ]
        })

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "Found duplicate tag")

    def test_model_bad_dataset_path(self):
        config = make_basic_config()
        config['models'][0]['tests'][0]['dataset_path'] = "bad name"

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "Dataset not found")

    def test_report_bad_type(self):
        config = make_basic_config()
        config['reports'][0]['type'] = "bad type"

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "bad type", 4)

    def test_report_bad_tag(self):
        config = make_basic_config()
        config['reports'][0]['tests'][0]['tag'] = "wrong tag"

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ExperimentConfig.construct(config)
        self.assert_error(log, "Unknown report tag")
