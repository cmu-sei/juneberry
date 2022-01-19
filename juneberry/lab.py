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
from pathlib import Path

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.filesystem as jbfs


class Lab:
    """
    Class which represents a "Laboratory" for performing experiments.  It contains local execution
    specific values as paths to workspaces, data roots, output directories, and numbers of gpus.
    """

    def __init__(self, *, workspace='.', data_root='.', tensorboard=None, num_gpus=0, num_workers=4,
                 machine_class=None):
        # We expose these as direct attributes
        self.tensorboard = Path(tensorboard) if tensorboard is not None else None
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.machine_class = machine_class

        # We store multiple workspaces and data_roots so we can search them.  The first one is
        # always the default.
        self._workspaces = {'default': Path(workspace)} if workspace is not None else {}
        self._data_roots = {'default': Path(data_root)} if data_root is not None else {}

    def create_copy_from_keys(self, ws_key, dr_key):
        return Lab(workspace=self.workspace(ws_key), data_root=self.data_root(dr_key),
                   tensorboard=self.tensorboard, num_gpus=self.num_gpus, num_workers=self.num_workers,
                   machine_class=self.machine_class)

    def workspace(self, ws_key='default') -> Path:
        """ :return: The path to the default workspace. """
        return self._workspaces[ws_key]

    def add_workspace(self, workspace, ws_key):
        self._workspaces[ws_key] = workspace

    def data_root(self, dr_key='default') -> Path:
        """ :return: The path to the default dataroot. """
        return self._data_roots[dr_key]

    def add_data_root(self, data_root, dr_key):
        self._data_roots[dr_key] = data_root

    def model_manager(self, model_name: str, model_version=None) -> jbfs.ModelManager:
        """ :return: The ModelManager for this model. """
        # return jbfs.ModelManager(self.workspace, model_name)
        return jbfs.ModelManager(model_name, model_version)

    # Convenience loaders

    def load_model_config(self, model_name, ws_key='default') -> ModelConfig:
        """
        Loads the config from the model and returns a model config object.
        :param model_name: The model name.
        :param ws_key: The workspace to use to load.
        :return: The model config object.
        """
        mm = self.model_manager(model_name)
        model_config_path = self.workspace(ws_key) / mm.get_model_config()
        return ModelConfig.load(model_config_path)

    def load_dataset_config(self, config_path, ws_key='default') -> DatasetConfig:
        """
        Loads the config object for the associated config path within the workspace.
        :param config_path: The config path within a workspace.
        :param ws_key: The workspace to use to load.
        :return: The config object.
        """
        full_path = Path(self.workspace(ws_key)) / config_path
        return DatasetConfig.load(str(full_path), relative_path=full_path.parent)

    def save_model_config(self, model_config, model_name, model_version=None, ws_key='default'):
        mm = self.model_manager(model_name, model_version)
        ws = self.workspace(ws_key)

        model_dir_path = ws / mm.get_model_dir()
        model_dir_path.mkdir(parents=True, exist_ok=True)

        model_config_path = ws / mm.get_model_config()

        model_config.timestamp = str(datetime.datetime.now())
        content = model_config.to_json()
        jbfs.save_json(content, model_config_path)
        return model_config_path

    def save_dataset_config(self, dataset_config_path: str, dataset_config: DatasetConfig, ws_key='default'):
        ws = self.workspace(ws_key)
        dataset_config_path = ws / dataset_config_path
        dataset_config_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_config.timestamp = str(datetime.datetime.now())
        dataset_config.save(str(dataset_config_path))
        return dataset_config_path

    def __str__(self):
        return f'{{"workspace"="{self.workspace()}", ' \
               f'"data_root"="{self.data_root()}", ' \
               f'"tensorboard"="{self.tensorboard}", ' \
               f'"num_gpus"={self.num_gpus}, ' \
               f'"num_workers"={self.num_workers}' \
               f'"machine_class"={self.machine_class}}}'
