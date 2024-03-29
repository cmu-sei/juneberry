#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

import datetime
import logging
from pathlib import Path
import sys

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.workspace import LabProfile, WorkspaceConfig
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Lab:
    """
    Class which represents a "Laboratory" for performing experiments. It contains local execution
    specific values, such as paths to workspaces, data roots, output directories, and other host-specific
    information.
    """

    def __init__(self, *, workspace='.', data_root='.', tensorboard=None,
                 profile_name: str = "default", model_zoo: str = None, cache=None):
        """
        Used to initialize the environment in which to run experiments.
        :param workspace: The path to the default workspace. By default this is the current working directory.
        :param data_root: The path to where the data is stored. By default this is the current working directory.
        :param tensorboard: The path to where the tensorboard information is stored.
        :param profile_name: The name of the lab profile.
        :param model_zoo: Optional url to the model zoo.
        :param cache: Optional path to where data is cached.
        """

        # We expose these as direct attributes.
        self.tensorboard = Path(tensorboard) if tensorboard is not None else None

        # Where we store host specific information.
        self.profile_name = profile_name

        # A place to store the workspace config and the profile.
        self.ws_config = WorkspaceConfig.load()
        self.profile: LabProfile
        self.profile = LabProfile()

        # Where models in the zoo can be found
        self.model_zoo = model_zoo

        # Where we cache items like model zoo models
        self.cache = Path(cache) if cache is not None else None

        # We store multiple workspaces and data_roots so we can search them.  The first one is
        # always the default and where we write things.
        # NOTE. We don't want people using these. They should access the workspace and data root
        # via workspace() and data_root()
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
    def validate_args(workspace: str, data_root: str, tensorboard: str, profile_name: str,
                      model_zoo: str = None, cache: str = None) -> None:
        """
        Checks to see that the four lab arguments are valid and exits if they aren't. We do NOT do this
        automatically on lab construction because there are cases where we want to construct a lab
        without everything existing because a script might create them.
        :param workspace: The workspace
        :param data_root: The data root
        :param tensorboard: OPTIONAL: tensorboard directory
        :param profile_name: OPTIONAL: Name of the profile to use.
        :param model_zoo: OPTIONAL: Model zoo url.
        :param cache: OPTIONAL: Path to the cache directory.
        :return:
        """
        errors = 0
        errors += Lab.check_path(workspace, "workspace")
        errors += Lab.check_path(data_root, "data root")
        if tensorboard is not None:
            errors += Lab.check_path(tensorboard, "tensorboard directory")

        # Try to load the lab profile from the workspace config
        ws_config = WorkspaceConfig.load()
        if not profile_name == "default" and not ws_config.has_profile(profile_name):
            logger.error(f"Profile '{profile_name}' does not exist in the workspace config file.")
            errors += 1

        if errors > 0:
            logger.error(f"Identified {errors} configuration errors. See log for details. Exiting.")
            sys.exit(-1)

    def create_copy_from_keys(self, ws_key, dr_key):
        # Construct a new one and copy over the profile
        lab = Lab(workspace=str(self.workspace(ws_key)), data_root=str(self.data_root(dr_key)),
                  tensorboard=self.tensorboard, profile_name=self.profile_name,
                  model_zoo=self.model_zoo, cache=self.cache)
        lab.profile = self.profile
        return lab

    def workspace(self, ws_key='default') -> Path:
        """ :return: The path to the default workspace. """
        return self._workspaces[ws_key]

    def add_workspace(self, workspace, ws_key):
        self._workspaces[ws_key] = workspace

    def data_root(self, dr_key='default') -> Path:
        """
        ":param dr_key: Which dataroot we want. "default" by default.
        :return: The path to the dataroot.
        """
        return self._data_roots[dr_key]

    def add_data_root(self, data_root, dr_key):
        self._data_roots[dr_key] = data_root

    @staticmethod
    def model_manager(model_name: str) -> jb_fs.ModelManager:
        """ :return: The ModelManager for this model. """
        return jb_fs.ModelManager(model_name)

    @staticmethod
    def experiment_manager(experiment_name: str) -> jb_fs.ExperimentManager:
        """ :return: The ExperimentManager for this experiment. """
        return jb_fs.ExperimentManager(experiment_name)

    @staticmethod
    def experiment_creator(experiment_name: str) -> jb_fs.ExperimentCreator:
        """ :return: The ExperimentCreator for this experiment. """
        return jb_fs.ExperimentCreator(experiment_name)

    # Setup commands
    def setup_lab_profile(self, *, model_name: str = None, model_config: ModelConfig = None) -> None:
        """
        When the lab is used for a specific model/model config, this adjusts the values in the
        lab based on that model.
        :param model_name: The name of the model.
        :param model_config: Optional loaded model config with potential overrides.
        :return: None
        """
        self.profile = self.load_lab_profile(model_name)
        if model_config is not None and model_config.lab_profile is not None:
            for k, v in model_config.lab_profile.items():
                if v is not None:
                    self.profile[k] = model_config.lab_profile[k]

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
        return DatasetConfig.load(config_path, relative_path=full_path.parent)

    def load_lab_profile(self, model_name: str = None, ws_key='default') -> LabProfile:
        """
        Finds the relevant lab profile based on lab's profile name and optional model name class.
        :param model_name: The model name.
        :param ws_key: The workspace to use to load.
        :return: A LabProfile object containing execution specifications.
        """
        return self.ws_config.get_profile(self.profile_name, model_name)

    def save_model_config(self, model_config, model_name, ws_key='default'):
        mm = self.model_manager(model_name)
        ws = self.workspace(ws_key)

        model_dir_path = ws / mm.get_model_dir()
        model_dir_path.mkdir(parents=True, exist_ok=True)

        model_config_path = ws / mm.get_model_config()

        model_config.timestamp = str(datetime.datetime.now())
        content = model_config.to_json()
        jb_fs.save_json(content, model_config_path)
        return model_config_path

    def save_dataset_config(self, dataset_config_path: str, dataset_config: DatasetConfig, ws_key='default'):
        ws = self.workspace(ws_key)
        dataset_config_path = ws / dataset_config_path
        dataset_config_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_config.timestamp = str(datetime.datetime.now())
        dataset_config.save(str(dataset_config_path))
        return dataset_config_path

    def __str__(self):
        return str({
            'workspace': str(self.workspace()),
            'data_root': str(self.data_root()),
            'tensorboard': str(self.tensorboard),
            'profile_name': self.profile_name,
            'profile': self.profile,
            'model_zoo': self.model_zoo,
            'cache_path': str(self.cache)
        })
