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
import logging
from pathlib import Path
import sys

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.machine_specs import MachineSpecs
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


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

    @staticmethod
    def check_path(path, label):
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"The requested {label} at {path_obj.absolute()} does not exist.")
            return 1
        else:
            return 0

    @staticmethod
    def validate_args(workspace: str, data_root: str, tensorboard: str, machine_class: str) -> None:
        """
        Checks to see that the four lab arguments are valid and dies if they aren't. We do NOT do this
        automatically on lab construction because there are cases where we want to construct a lab
        without everything existing because might create them.
        :param workspace: The workspace
        :param data_root: The data root
        :param tensorboard: OPTIONAL: tensorboard directory
        :param machine_class: OPTIONAL: Machine class.
        :return:
        """
        errors = 0
        errors += Lab.check_path(workspace, "workspace")
        errors += Lab.check_path(data_root, "data root")
        if tensorboard is not None:
            errors += Lab.check_path(tensorboard, "tensorboard directory")

        # Try to load the machine config
        # TODO:

        if errors > 0:
            logger.error(f"Identified {errors} configuration errors. See log for details. Exiting.")
            sys.exit(-1)

    def create_copy_from_keys(self, ws_key, dr_key):
        return Lab(workspace=str(self.workspace(ws_key)), data_root=str(self.data_root(dr_key)),
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

    def load_machine_specs(self, model_name, ws_key='default') -> MachineSpecs:
        """
        Finds the relevant execution specs based on machine and model class.
        :param model_name: The model name.
        :param ws_key: The workspace to use to load.
        :return: A MachineSpecs object containing execution specifications.
        """
        machine_class = self.machine_class
        machine_config_path = Path(self.workspace(ws_key)) / 'config.json'
        return MachineSpecs.load(data_path=str(machine_config_path), machine_class=machine_class, model_name=model_name)

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
               f'"num_workers"={self.num_workers}, ' \
               f'"machine_class"={self.machine_class}}}'
