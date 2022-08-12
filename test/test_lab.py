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


def test_construction():
    lab = Lab(workspace='ws', data_root='dr')
    assert lab.workspace() == Path('ws')
    assert lab.data_root() == Path('dr')


def test_model_zoo():
    lab = Lab(model_zoo="some_url")
    assert lab.model_zoo == "some_url"


def test_cache_path():
    lab = Lab(cache='/path/to/cache')
    assert lab.cache == Path("/path/to/cache")


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
    save_path = lab.save_model_config(model_data, 'test_model')

    assert save_path.exists()
    assert save_path == tmp_path / 'workspace' / 'models' / 'test_model' / 'config.json'


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
