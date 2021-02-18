#! /usr/bin/env python3

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

from pathlib import Path
import unittest

import juneberry
from juneberry.config.dataset import DatasetConfig
from juneberry.config.dataset import DataType, TaskType


def make_basic_config(image_data=True, classification=True):
    config = {
        "numModelClasses": 4,
        "description": "Unit test",
        "timestamp": "never",
        "formatVersion": "3.2.0",
        "labelNames": {"0": "frodo", "1": "sam"},
    }

    if image_data:
        config['dataType'] = 'image'
        if classification:
            config['imageData'] = {
                "taskType": "classification",
                "sources": [{"directory": "some/path", "label": 0}]
            }
        else:
            config['imageData'] = {
                "taskType": "objectDetection",
                "sources": [{"directory": "some/path"}]
            }
    else:
        config['dataType'] = 'tabular'
        config['tabularData'] = {
            "sources": [{"path": "some/path"}],
            "labelIndex": 0
        }

    return config


def make_sample_data(tmp_path):
    data_dir = Path("data")

    data_path = Path(tmp_path) / data_dir
    data_path.mkdir(parents=True)

    file_names = ["file_one.txt", "file_two.txt"]
    for file_name in file_names:
        file_path = data_path / file_name
        if not file_path.exists():
            with open(file_path, "w") as out_file:
                out_file.write("Hello world\n")

    return data_dir, file_names


def test_config_basics():
    config = make_basic_config()

    ds = DatasetConfig(config, Path("."))
    assert ds.is_image_type() is True
    assert ds.data_type == DataType.IMAGE
    assert ds.task_type == TaskType.CLASSIFICATION
    assert ds.num_model_classes == 4
    assert ds.description == "Unit test"
    assert ds.timestamp == "never"
    assert ds.format_version == "3.2.0"

    assert len(ds.label_names) == 2
    assert ds.label_names[0] == "frodo"
    assert ds.label_names[1] == "sam"

    config = make_basic_config(True, False)
    ds = DatasetConfig(config, Path("."))
    assert ds.data_type == DataType.IMAGE
    assert ds.task_type == TaskType.OBJECTDETECTION

    config = make_basic_config(False)
    ds = DatasetConfig(config, Path("."))
    assert ds.data_type == DataType.TABULAR


def test_image_data():
    config = make_basic_config()
    image_data = {
        "imageData": {
            "taskType": "classification",
            "sources": [{
                "directory": "some/path",
                "label": 4
            }],
        },
    }

    config.update(image_data)

    ds = DatasetConfig(config, Path('.'))
    assert ds.is_image_type()
    assert len(ds.get_image_sources()) == 1


def test_tabular_data_data_root(tmp_path):
    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["dataType"] = "tabular"

    csv_data = {
        "tabularData": {
            "sources": [{
                "root": "dataroot",
                "path": data_dir / file_names[0]
            }],
            "labelIndex": 20
        },
    }

    config.update(csv_data)

    juneberry.DATA_ROOT = tmp_path
    ds = DatasetConfig(config, Path('.'))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels()

    assert len(paths) == 1
    assert paths[0] == Path(juneberry.DATA_ROOT) / data_dir / file_names[0]
    assert index == 20


def test_csv_data_workspace(tmp_path):
    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["dataType"] = "tabular"

    csv_data = {
        "tabularData": {
            "sources": [
                {
                    "root": "workspace",
                    "path": data_dir / file_names[0]
                },
                {
                    "root": "workspace",
                    "path": data_dir / file_names[1]
                }
            ],
            "labelIndex": 20
        },
    }

    config.update(csv_data)

    juneberry.WORKSPACE_ROOT = tmp_path
    ds = DatasetConfig(config, Path('.'))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels()

    assert len(paths) == len(file_names)
    for i in range(len(file_names)):
        assert paths[i] == Path(juneberry.WORKSPACE_ROOT) / data_dir / file_names[i]

    assert index == 20


def test_csv_data_relative(tmp_path):
    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["dataType"] = "tabular"

    csv_data = {
        "tabularData": {
            "sources": [{
                "root": "relative",
                "path": data_dir / file_names[0]
            }],
            "labelIndex": 20
        },
    }

    config.update(csv_data)

    ds = DatasetConfig(config, Path(tmp_path))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels()

    assert len(paths) == 1
    assert paths[0] == Path(tmp_path) / data_dir / file_names[0]
    assert index == 20


def test_csv_glob(tmp_path):
    # The path structure should support globbing
    # So, use the tmp_path as the dataroot and slap in some files

    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["dataType"] = "tabular"

    csv_data = {
        "tabularData": {
            "sources": [{
                "root": "dataroot",
                "path": data_dir / "file_*.txt"
            }],
            "labelIndex": 42
        },
    }

    config.update(csv_data)

    # Now construct the object and see what we have
    juneberry.DATA_ROOT = tmp_path
    ds = DatasetConfig(config, Path('.'))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels()

    assert len(paths) == len(file_names)

    # The glob operator may provide them in a different order so we just need
    # to make sure it is in there somewhere
    for file_name in file_names:
        assert Path(tmp_path) / data_dir / file_name in paths

    assert index == 42


class TestFormatErrors(unittest.TestCase):

    # Helper method for analyzing error
    def assert_error(self, cm, message):
        self.assertEqual(len(cm.output), 2)
        self.assertIn("ERROR:root", cm.output[0])
        self.assertIn(message, cm.output[0])

    def test_image_data_missing_directory(self):
        config = make_basic_config()
        config['imageData']['sources'] = {"label": 0}

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig(config, Path('.'))
        self.assert_error(log, "directory")

    def test_image_data_missing_label(self):
        config = make_basic_config()
        config['imageData']['sources'] = {"directory": "some/path"}

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig(config, Path('.'))
        self.assert_error(log, "label")

    def test_csv_data_missing_path(self):
        config = make_basic_config(False)
        config['tabularData']['sources'] = [{"root": "dataroot"}]

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig(config, Path('.'))
        self.assert_error(log, "path")

    def test_csv_data_bad_root(self):
        config = make_basic_config(False)
        config['tabularData']['sources'] = [{"root": "foobar", "path": "dataroot"}]

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig(config, Path('.'))
        self.assert_error(log, "foobar")

    def test_csv_bad_label_index(self):
        config = make_basic_config(False)
        del config['tabularData']['labelIndex']

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig(config, Path('.'))
        self.assert_error(log, "labelIndex")


def test_sampling():
    config = make_basic_config()
    sampling_data = {
        "sampling": {
            "algorithm": "foo",
            "args": "none"
        },
    }

    config.update(sampling_data)

    ds = DatasetConfig(config, Path('.'))
    assert ds.has_sampling()
    assert ds.sampling['algorithm'] == "foo"
    assert ds.sampling['args'] == "none"
