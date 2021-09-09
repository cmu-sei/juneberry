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

from pathlib import Path
import unittest

from juneberry.config.dataset import DatasetConfigBuilder
from juneberry.config.dataset import DatasetConfig, DataType, TaskType, SamplingAlgo
from juneberry.lab import Lab


def make_basic_config(image_data=True, classification=True):
    config = {
        "num_model_classes": 4,
        "description": "Unit test",
        "timestamp": "never",
        "format_version": "3.2.0",
        "label_names": {"0": "frodo", "1": "sam"},
    }

    if image_data:
        config['data_type'] = 'image'
        if classification:
            config['image_data'] = {
                "task_type": "classification",
                "sources": [{"directory": "some/path", "label": 0}]
            }
        else:
            config['image_data'] = {
                "task_type": "object_detection",
                "sources": [{"directory": "some/path"}]
            }
    else:
        config['data_type'] = 'tabular'
        config['tabular_data'] = {
            "sources": [{"path": "some/path"}],
            "label_index": 0
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

    ds = DatasetConfig.construct(config, Path("."))
    assert ds.is_image_type() is True
    assert ds.data_type == DataType.IMAGE
    assert ds.image_data.task_type == TaskType.CLASSIFICATION
    assert ds.num_model_classes == 4
    assert ds.description == "Unit test"
    assert ds.timestamp == "never"
    assert ds.format_version == "3.2.0"

    assert len(ds.label_names) == 2
    assert ds.label_names['0'] == "frodo"
    assert ds.label_names['1'] == "sam"

    config = make_basic_config(True, False)
    ds = DatasetConfig.construct(config, Path("."))
    assert ds.data_type == DataType.IMAGE
    assert ds.image_data.task_type == TaskType.OBJECT_DETECTION

    config = make_basic_config(False)
    ds = DatasetConfig.construct(config, Path("."))
    assert ds.data_type == DataType.TABULAR


def test_config_builder_basic(tmp_path):
    # Test to json
    # Test load from json as dataset
    lab = Lab(workspace=str(Path(tmp_path) / 'workspace'), data_root=str(Path(tmp_path) / 'data_root'))
    data_dir, file_names = make_sample_data(lab.data_root())
    builder = DatasetConfigBuilder(DataType.IMAGE, TaskType.OBJECT_DETECTION, 'Unit test')
    builder.add_source(lab, directory=Path(data_dir / file_names[0]), label=0, remove_image_ids=[0, 1, 2],
                       sampling_fraction=1)
    builder.add_source(lab, directory=Path(data_dir / file_names[1]), label=1)
    builder.set_sampling(SamplingAlgo.RANDOM_FRACTION, {'fraction': 0.5, 'seed': 1234})

    config_path = lab.workspace() / 'data_sets' / 'unit_test_builder.json'

    assert len(builder.config.image_data['sources']) == 2
    assert builder.config.image_data.sources[0].remove_image_ids == [0, 1, 2]
    assert 'label_names' in builder.errors.keys()
    assert builder.config.sampling.algorithm == SamplingAlgo.RANDOM_FRACTION

    path = lab.save_dataset_config(str(config_path), builder.config)
    assert path.exists()
    config = lab.load_dataset_config(config_path)
    print(vars(config))
    assert len(config.image_data.sources) == 2
    assert config.image_data.sources[0].remove_image_ids == [0, 1, 2]
    # TODO: What is the whole error mechanism for??
    # assert 'label_names' in builder.config['errors']
    assert builder.config.sampling['algorithm'] == SamplingAlgo.RANDOM_FRACTION


def test_image_data():
    config = make_basic_config()
    image_data = {
        "image_data": {
            "task_type": "classification",
            "sources": [{
                "directory": "some/path",
                "label": 4
            }],
        },
    }

    config.update(image_data)

    ds = DatasetConfig.construct(config, Path('.'))
    assert ds.is_image_type()
    assert len(ds.get_image_sources()) == 1


def test_obj_detection_data():
    config = make_basic_config()
    image_data = {
        "image_data": {
            "task_type": "object_detection",
            "sources": [
                {
                    "directory": "some/path/*.json",
                },
                {
                    "directory": "another/path/*.json"
                }
            ],
        },
    }
    config.update(image_data)
    ds = DatasetConfig.construct(config, Path('.'))
    assert ds.is_image_type()
    assert ds.is_object_detection_task()
    assert len(ds.get_image_sources()) == 2


def test_tabular_data_data_root(tmp_path):
    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["data_type"] = "tabular"

    csv_data = {
        "tabular_data": {
            "sources": [{
                "root": "dataroot",
                "path": str(data_dir / file_names[0])
            }],
            "label_index": 20
        },
    }

    config.update(csv_data)

    lab = Lab(workspace='.', data_root=tmp_path)
    ds = DatasetConfig.construct(config, Path('.'))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels(lab)

    assert len(paths) == 1
    assert paths[0] == Path(tmp_path) / data_dir / file_names[0]
    assert index == 20


def test_csv_data_workspace(tmp_path):
    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["data_type"] = "tabular"

    csv_data = {
        "tabular_data": {
            "sources": [
                {
                    "root": "workspace",
                    "path": str(data_dir / file_names[0])
                },
                {
                    "root": "workspace",
                    "path": str(data_dir / file_names[1])
                }
            ],
            "label_index": 20
        },
    }

    config.update(csv_data)

    lab = Lab(workspace=tmp_path, data_root='.')
    ds = DatasetConfig.construct(config, Path('.'))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels(lab)

    assert len(paths) == len(file_names)
    for i in range(len(file_names)):
        assert paths[i] == Path(tmp_path) / data_dir / file_names[i]

    assert index == 20


def test_csv_data_relative(tmp_path):
    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["data_type"] = "tabular"

    csv_data = {
        "tabular_data": {
            "sources": [{
                "root": "relative",
                "path": str(data_dir / file_names[0])
            }],
            "label_index": 20
        },
    }

    config.update(csv_data)

    lab = Lab(workspace=tmp_path, data_root=tmp_path)
    ds = DatasetConfig.construct(config, Path(tmp_path))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels(lab)

    assert len(paths) == 1
    assert paths[0] == Path(tmp_path) / data_dir / file_names[0]
    assert index == 20


def test_csv_glob(tmp_path):
    # The path structure should support globbing
    # So, use the tmp_path as the dataroot and slap in some files

    config = make_basic_config()
    data_dir, file_names = make_sample_data(tmp_path)

    config["data_type"] = "tabular"

    csv_data = {
        "tabular_data": {
            "sources": [{
                "root": "dataroot",
                "path": str(data_dir / "file_*.txt")
            }],
            "label_index": 42
        },
    }

    config.update(csv_data)

    # Now construct the object and see what we have
    lab = Lab(workspace=tmp_path, data_root=tmp_path)
    ds = DatasetConfig.construct(config, Path('.'))
    assert ds.is_tabular_type()

    paths, index = ds.get_resolved_tabular_source_paths_and_labels(lab)

    assert len(paths) == len(file_names)

    # The glob operator may provide them in a different order so we just need
    # to make sure it is in there somewhere
    for file_name in file_names:
        assert Path(tmp_path) / data_dir / file_name in paths

    assert index == 42


class TestFormatErrors(unittest.TestCase):

    # Helper method for analyzing error
    def assert_error(self, cm, message):
        # print(f"OUTPUT: {cm.output}")
        self.assertEqual(len(cm.output), 3)
        self.assertIn("ERROR:juneberry", cm.output[0])
        self.assertIn(message, cm.output[0])

    def test_image_data_missing_directory(self):
        config = make_basic_config()
        config['image_data']['sources'] = [{"label": 0}]

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig.construct(config, Path('.'))
        self.assert_error(log, "directory")

    # TODO: This is only valid if the task type is classification.
    # def test_image_data_missing_label(self):
    #     config = make_basic_config()
    #     config['image_data']['sources'] = [{"directory": "some/path"}]
    #
    #     with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
    #         ds = DatasetConfig.construct(config, Path('.'))
    #     self.assert_error(log, "label")

    def test_csv_data_missing_path(self):
        config = make_basic_config(False)
        config['tabular_data']['sources'] = [{"root": "dataroot"}]

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig.construct(config, Path('.'))
        self.assert_error(log, "path")

    def test_csv_data_bad_root(self):
        config = make_basic_config(False)
        config['tabular_data']['sources'] = [{"root": "foobar", "path": "dataroot"}]

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig.construct(config, Path('.'))
        self.assert_error(log, "foobar")

    def test_csv_bad_label_index(self):
        config = make_basic_config(False)
        del config['tabular_data']['label_index']

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            ds = DatasetConfig.construct(config, Path('.'))
        self.assert_error(log, "label_index")


def test_sampling():
    config = make_basic_config()
    sampling_data = {
        "sampling": {
            "algorithm": "random_fraction",
            "args": "none"
        },
    }

    config.update(sampling_data)

    ds = DatasetConfig.construct(config, Path('.'))
    assert ds.has_sampling()
    assert ds.sampling['algorithm'] == "random_fraction"
    assert ds.sampling['args'] == "none"
