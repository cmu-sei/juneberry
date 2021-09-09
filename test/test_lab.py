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

import datetime
import json
from pathlib import Path

from juneberry.lab import Lab
from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig

import test_model_config
import test_data_set


def create_tmp_model(lab: Lab, tmp_path, model_name) -> dict:
    mm = lab.model_manager(model_name)
    model_path = Path(tmp_path) / mm.get_model_dir()
    model_path.mkdir(parents=True)

    data = test_model_config.make_basic_config()
    with open(Path(tmp_path) / mm.get_model_config(), "w") as json_file:
        json.dump(data, json_file)

    return data


def create_tmp_dataset(tmp_path, dataset_name) -> dict:
    ds_path = Path(tmp_path) / dataset_name
    ds_path.parent.mkdir(parents=True)

    data = test_data_set.make_basic_config(image_data=True, classification=True)
    with open(ds_path, "w") as json_file:
        json.dump(data, json_file)

    return data


def test_default_construction():
    lab = Lab(workspace='ws', data_root='dr')
    assert lab.workspace() == Path('ws')
    assert lab.data_root() == Path('dr')
    assert lab.tensorboard is None
    assert lab.num_gpus == 0


def test_construction():
    lab = Lab(workspace='ws', data_root='dr')
    assert lab.workspace() == Path('ws')
    assert lab.data_root() == Path('dr')


def test_model_loading(tmp_path):
    model_name = "pytest_model"

    # Setup up a lab object and paths
    lab = Lab(workspace=tmp_path, data_root=tmp_path)
    data = create_tmp_model(lab, tmp_path, model_name)

    # Now make sure we can load it
    model_config = lab.load_model_config(model_name)
    assert model_config.seed == data['seed']


def test_dataset_loading(tmp_path):
    ds_name = "data_sets/pytest_dataset.json"

    # Setup up a lab object and paths
    lab = Lab(workspace=tmp_path, data_root=tmp_path)
    data = create_tmp_dataset(tmp_path, ds_name)

    # Now make sure we can load it
    ds_config = lab.load_dataset_config(lab.workspace() / ds_name)
    assert ds_config.num_model_classes == data['num_model_classes']


def test_model_config_saving(tmp_path):
    model_data = ModelConfig.construct(test_model_config.make_basic_config())
    model_data.task = 'object_detection'
    model_data.timestamp = datetime.datetime.now
    
    lab = Lab(workspace=tmp_path / 'workspace', data_root=tmp_path / 'data_root')
    save_path = lab.save_model_config(model_data, 'test_model', model_version='v1')

    assert save_path.exists()
    assert save_path == tmp_path / 'workspace' / 'models' / 'test_model' / 'v1' / 'config.json'


def test_dataset_config_saving(tmp_path):
    data = DatasetConfig.construct(test_data_set.make_basic_config(image_data=True, classification=False))
    data.timestamp = datetime.datetime.now

    lab = Lab(workspace=tmp_path / 'workspace', data_root=tmp_path / 'data_root')
    config_path = lab.workspace() / 'test_dataset.json'
    save_path = lab.save_dataset_config(str(config_path), data)

    assert save_path.exists()
    assert save_path == tmp_path / 'workspace' / 'test_dataset.json'


def test_str(tmp_path):
    lab = Lab(workspace=tmp_path, data_root=tmp_path)
    print(lab)
