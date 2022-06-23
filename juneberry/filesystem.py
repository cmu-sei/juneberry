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
Script that supports the model layout.

NOTE: These paths are all relative to the Juneberry workspace root.
"""

import datetime
import hashlib
import hjson
import json
import logging
import numpy as np
import os
from pathlib import Path
import socket
import sys

from juneberry import Platforms

logger = logging.getLogger(__name__)


def clean_if_exists(files_to_clean, dry_run=False):
    """
    Removes file or folder if it exists
    :param files_to_clean: Array of Path objects to remove
    :param dry_run: Flag to commit cleaning
    """
    # Paths are sorted in reverse order so that when directories are removed they will already be empty
    files = sorted(files_to_clean, reverse=True)
    msg = "Removing"
    if dry_run:
        msg = "Would remove"
    for file in files:
        if file.exists():
            logger.info(f"{msg}: {file}")
            if not dry_run:
                if file.is_dir():
                    is_empty = not any(file.iterdir())
                    if is_empty:
                        file.rmdir()
                    else:
                        logger.error(f"\tDirectory {file} is not empty so removal failed!")
                else:
                    file.unlink()


def load_json(json_path: str, attribute_map=None):
    """
    Responsible for reading data from a JSON file and (if necessary) converting JSON
    keys to their Python equivalent attributes.
    :param json_path: The Path to the JSON file to be loaded.
    :param attribute_map: An optional dictionary that translates between JSON fields
    and their corresponding Python attribute names.
    :return data: A dictionary of data that was read from the JSON file.
    """
    # Attempt to open up the file at the indicated Path and read the data.
    try:
        with open(json_path) as json_file:
            data = hjson.load(json_file)

    # If the file was not found, log an error and exit.
    except FileNotFoundError:
        logger.error(f"Failed to load {json_path}. The file could not be found. EXITING.")
        sys.exit(-1)

    # If an attribute map was provided, check if any of the JSON fields have a different
    # attribute name in Python and perform the conversion.
    if attribute_map is not None:

        # Loop through every key in what was read from the JSON file.
        for key in data.copy().keys():

            # If the key is in the attribute map, then it has a different name in Python.
            if key in attribute_map.keys():
                # Replace the current key with its Python equivalent name and delete the old key.
                data[attribute_map[key]] = data[key]
                del data[key]

    return data


def save_hjson(data, json_path, *, indent: int = 4) -> None:
    """
    Save the data to the specified path (string or Path) in HJSON format, applying the specified indent (int),
    and converting all the traditional non encoding bits (e.g. Path, numpy) to
    the appropriate data structure.
    :param data: The data to save.
    :param json_path: The path to save the data to.
    :param indent: The indent spacing to use; with a default of 4.
    :return: None
    """
    with open(json_path, "w") as json_file:
        hjson.dump(data, json_file, indent=indent, default=json_cleaner, sort_keys=True)


def save_json(data, json_path, *, indent: int = 4) -> None:
    """
    Save the data to the specified path (string or Path) in JSON format, applying the specified indent (int),
    and converting all the traditional non encoding bits (e.g. Path, numpy) to
    the appropriate data structure.
    :param data: The data to save.
    :param json_path: The path to save the data to.
    :param indent: The indent spacing to use; with a default of 4.
    :return: None
    """
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=indent, default=json_cleaner, sort_keys=True)


def save_file(data, path: str, *, indent: int = 4) -> None:
    """
    Generic file saver that chooses the file format based on the extension of the path.
    :param path: 
    :param indent: The indent spacing to use; with a default of 4.
    :return: None
    """
    ext = Path(path).suffix.lower()
    if ext == '.json':
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=indent, default=json_cleaner, sort_keys=True)
    elif ext == '.hjson':
        with open(path, 'w') as hjson_file:
            hjson.dump(data, hjson_file, indent=indent, default=json_cleaner, sort_keys=True)

    # TODO: implement file handlers for these formats.
    elif ext in {'.gzip', '.gz'}:
        logger.error('TODO: handle gzip extensions.')
        sys.exit(-1)
    elif ext in {'.yaml', '.yml'}:
        logger.error('TODO: handle YAML extensions.')
        sys.exit(-1)
    elif ext in {'.toml', '.tml'}:
        logger.error('TODO: handle TOML extensions.')
        sys.exit(-1)
    else:
        logger.error(f'Unsupported file extension {ext}')
        sys.exit(-1)


def load_file(path: str):
    """
    Loads the file from the specified file path.
    :param path: The path to the file to load.
    :return:
    """
    if Path(path).exists():
        ext = Path(path).suffix.lower()
        if ext in {'.json', '.hjson'}:
            # HJSON is a superset of JSON, so the HJSON parser can handle both cases.
            with open(path, 'rb') as in_file:
                return hjson.load(in_file)

        # TODO: implement file handlers for these formats.
        elif ext in {'.gzip', '.gz'}:
            logger.error('TODO: handle gzip extensions.')
            sys.exit(-1)
        elif ext in {'.yaml', '.yml'}:
            logger.error('TODO: handle YAML extensions.')
            sys.exit(-1)
        elif ext in {'.toml', '.tml'}:
            logger.error('TODO: handle TOML extensions.')
            sys.exit(-1)
        else:
            logger.error(f'Unsupported file extension {ext}')
            sys.exit(-1)

    else:
        logger.error(f'Failed to load {path}. The file could not be found. Exiting.')
        sys.exit(-1)
    

class ExperimentManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_dir_path = Path('experiments') / self.experiment_name
        self.experiment_model_dir_path = Path('models') / self.experiment_name
        self.experiment_log_dir_path = self.experiment_dir_path / "logs"
        self.experiment_reports_dir_path = self.experiment_dir_path / "report_json_files"

        if not self.experiment_log_dir_path.exists():
            self.experiment_log_dir_path.mkdir(parents=True)

    def get_experiment_name(self):
        """ :return: The name of the experiment. """
        return self.experiment_name

    def _get_experiment_dir(self):
        """ :return: The experiment directory. """
        return self.experiment_dir_path

    def get_experiment_model_dir(self):
        """ :return: The path to the directory containing all of the experiment model configs. """
        return self.experiment_model_dir_path

    def get_experiment_log_dir(self):
        """ :return: The path to the directory containing some of the logs generated during the experiment. """
        return self.experiment_log_dir_path

    def get_experiment_reports_dir(self):
        """ :return: The path to the directory containing the individual report configs for the experiment. """
        return self.experiment_reports_dir_path

    def get_experiment_report_file(self, idx: int):
        """ :return: A path to a particular report file inside the experiment reports directory."""
        return self.experiment_reports_dir_path / f"report_{idx}.json"

    def get_experiment_config(self):
        """ :return: The relative path to the experiment's config file. """
        return self.experiment_dir_path / 'config.json'

    def get_experiment_rules(self):
        """ :return: The relative path to the experiment's rules file. """
        return self.experiment_dir_path / 'rules.json'

    def get_experiment_dodo(self, workflow: str):
        """ :return: The relative path to the experiment's dodo.py pydoit file for the specified workflow. """
        file_name = workflow + "_dodo.py"
        return self.experiment_dir_path / file_name

    def get_experiment_file(self, file_name):
        """
        :param file_name: Name of file
        :return: The relative path to the experiment's file
        """
        return self.experiment_dir_path / file_name

    def get_experiment_db_file(self):
        """ :return: The relative path to the experiment's DB-file (pydoit). """
        return self.experiment_dir_path / '.doit.db'

    def get_experiment_log_path(self):
        """ :return: The path to the experiment log file. """
        return self.experiment_log_dir_path / "log_experiment.txt"

    def get_experiment_dryrun_log_path(self):
        """ :return: The path to the experiment log file. """
        return self.experiment_log_dir_path / "log_experiment_dryrun.txt"

    def get_output_list(self):
        """
        :return: Returns a list of files or glob patterns of outputs.
        """
        return [self.get_experiment_dryrun_log_path()]

    def clean(self, dry_run=False):
        """
        Cleans all files and folders in model directory that were generated by the framework except for
        'log_experiment.txt' and 'config.json'.
        :param dry_run: Flag to commit cleaning
        """
        clean_if_exists(self.get_output_list(), dry_run)

    def ensure_experiment_directory(self) -> None:
        """ Checks for the existence of the experiment directory and exits if the directory doesn't exist. """
        if not os.path.exists(self.experiment_dir_path):
            logger.error(f"Experiment directory '{self.experiment_dir_path}' does not exist!! EXITING.")
            exit(-1)


class ExperimentCreator(ExperimentManager):
    def __init__(self, experiment_name):
        """
        :param experiment_name: Name of the experiment directory.
        """
        super().__init__(experiment_name)

    def get_experiment_creation_log(self):
        """ :return: The path to the experiment creation log. """
        return self.experiment_dir_path / 'log_experiment_creation.txt'

    def get_experiment_creation_dryrun_log(self):
        """ :return: The path to the dry run of the experiment creation log. """
        return self.experiment_dir_path / 'log_experiment_creation_dryrun.txt'

    def get_experiment_outline(self):
        """ :return: The path to the model's config file. """
        return self.experiment_dir_path / 'experiment_outline.json'


class AttackManager(ExperimentManager):
    def __init__(self, experiment_name):
        """
        :param experiment_name: Name of the experiment directory for the attack.
        """
        super().__init__(experiment_name)

    def setup(self):
        """
        This method is responsible for performing the setup tasks associated with the Attack Manager.
        """
        # A list of directories, mostly in the model directory, that the attack will use.
        directories = [
            self.experiment_model_dir_path,
            self.get_private_dir(),
            self.get_shadow_dir(),
            self.get_meta_dir(),
            self.get_private_subdir(disjoint=False),
            self.get_private_subdir(disjoint=True),
            self.get_meta_subdir(disjoint=False),
            self.get_meta_subdir(disjoint=True)
        ]

        # If the attack directories don't exist, create them.
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

    def get_attack_setup_log(self):
        """ :return: The path to the attack setup log. """
        return self.experiment_log_dir_path / 'log_attack_setup.txt'

    def get_attack_setup_dryrun_log(self):
        """ :return: The path to the dry run of the attack setup log. """
        return self.experiment_log_dir_path / 'log_attack_setup_dryrun.txt'

    def get_experiment_attack_file(self):
        """ :return: The path to the experiment's attack config file. """
        return self.experiment_dir_path / 'attack.json'

    def get_experiment_dataset_dir(self):
        """ :return: A directory for storing experiment-specific dataset configs. """
        dataset_dir = self._get_experiment_dir() / 'datasets'
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def get_experiment_dataset_path(self, dataset_num: int):
        """
        :param dataset_num: An integer corresponding to a particular dataset config file.
        :return: The relative path to a particular experiment-specific dataset config file.
        """
        return self.get_experiment_dataset_dir() / f"{dataset_num}.json"

    def get_experiment_inout_dataset_path(self, dataset_type_str: str, meta=False, disjoint=False):
        """
        :param dataset_type_str: A string (usually "training" | "val" | "test") that will be
        used to construct the name of the in_out dataset config file.
        :param meta: Boolean that controls whether to return the 'meta' (default) or
        'private' version of the inout dataset.
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        :return: The relative path to a particular in_out dataset config file within the
        experiment-specific dataset config files.
        """
        type_str = 'disjoint' if disjoint else 'superset'
        mod_str = 'meta' if meta else 'private'
        return self.get_experiment_dataset_dir() / f'in_out_{type_str}_{mod_str}_{dataset_type_str}_dataset_config.json'

    def get_private_dir(self):
        """ :return: The path to the 'private' subdirectory within the attack's model directory. """
        return self.experiment_model_dir_path / 'private'

    def get_private_subdir(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        :return: The path to the 'disjoint' or 'superset' subdirectory within the
        attack's private model subdirectory. """
        sub_dir = 'disjoint' if disjoint else 'superset'
        return self.get_private_dir() / sub_dir

    def get_private_model_name(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        :return: A string of the model name which corresponds to the 'disjoint' or 'superset'
        private model for the attack."""
        model_dir_parts = self.get_private_subdir(disjoint=disjoint).parts[1:]
        return Path(*model_dir_parts)

    def get_private_model_config(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        :return: The path to the model config file for either the 'superset' or 'disjoint' private model.
        """
        return self.get_private_subdir(disjoint=disjoint) / 'config.json'

    def get_private_model_query_dataset_config_path(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        "return: The path to a query dataset config for a particular private model.
        """
        return self.get_private_subdir(disjoint=disjoint) / 'query_dataset_config.json'

    def get_shadow_dir(self):
        """ :return: The path to the 'shadow' subdirectory within the attack's model directory. """
        return self.experiment_model_dir_path / 'shadow'

    def get_shadow_subdir(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        : return: The path to the 'disjoint' or 'superset' subdirectory within the attack's
        shadow model subdirectory.
        """
        sub_dir = 'disjoint' if disjoint else 'superset'
        return self.get_shadow_dir() / sub_dir

    def get_shadow_model_dir(self, model_num: int, disjoint=False):
        """
        :param model_num: Integer that identifies a particular shadow model subdirectory.
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        :return: The path to a particular shadow model subdirectory within either the
        'superset' (default) or 'disjoint' shadow model subdirectory.
        """
        return self.get_shadow_subdir(disjoint=disjoint) / str(model_num)

    def get_shadow_model_training_config_path(self, model_num: int, disjoint=False):
        """
        :param model_num: Integer that identifies a particular shadow model subdirectory.
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        "return: The path to a training dataset config for a particular shadow model.
        """
        return self.get_shadow_model_dir(model_num, disjoint=disjoint) / 'training_dataset_config.json'

    def get_shadow_model_query_dataset_config_path(self, model_num: int, disjoint=False):
        """
        :param model_num: Integer that identifies a particular shadow model subdirectory.
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        "return: The path to a query dataset config for a particular shadow model.
        """
        return self.get_shadow_model_dir(model_num, disjoint=disjoint) / 'query_dataset_config.json'

    def get_shadow_model_name(self, model_num: int, disjoint=False):
        """
        :param model_num: Integer that identifies a particular shadow model subdirectory.
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        "return: The model name for a particular shadow model.
        """
        model_dir_parts = self.get_shadow_model_dir(model_num, disjoint=disjoint).parts[1:]
        return Path(*model_dir_parts)

    def get_meta_dir(self):
        """ :return: The path to the 'meta' subdirectory within the attack's model directory. """
        return self.experiment_model_dir_path / 'meta'

    def get_meta_subdir(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        : return: The path to the 'disjoint' or 'superset' subdirectory within the attack's
        meta model subdirectory.
        """
        sub_dir = 'disjoint' if disjoint else 'superset'
        return self.get_meta_dir() / sub_dir

    def get_plugin_file(self, meta=False, disjoint=False):
        """
        :param meta: Boolean that controls whether to return the 'meta' version of the
        in_out_plugin or the 'private' directory version.
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        : return: The path to the Plugin file used to build the in_out datasets for the
        'disjoint' or 'superset' meta model.
        """
        if meta:
            return self.get_meta_subdir(disjoint=disjoint) / f'in_out_plugin.json'
        else:
            return self.get_private_subdir(disjoint=disjoint) / f'in_out_plugin.json'

    def get_meta_model_name(self, disjoint=False):
        """
        :param disjoint: Boolean that controls whether to return the 'superset' (default) or
        'disjoint' version of the desired property.
        "return: The model name for a particular meta model.
        """
        model_dir_parts = self.get_meta_subdir(disjoint=disjoint).parts[1:]
        return Path(*model_dir_parts)


class EvalDirMgr:
    def __init__(self, root: str, platform: str, dataset_name: str = None) -> None:
        """
        Constructs an EvalDirMgr object rooted at the root path for the specified platform
        with the given name.
        NOTE: These are usually created via the ModelManager's "get_eval_dir_mgr" method.
        :param root: The root directory.
        :param platform: The platform.
        :param dataset_name: The name of the evaluation dataset.
        """
        self.root = Path(root) / 'eval' / dataset_name if dataset_name else Path(root) / 'eval'
        self.platform = platform

    def setup(self):
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def get_dir(self):
        return self.root

    def get_scratch_dir(self):
        return self.root / "scratch"

    def get_dryrun_imgs_dir(self):
        return str(self.root / "dryrun_imgs")

    def get_platform_config(self):
        return str(self.root / "platform_config.json")

    def get_manifest_path(self):
        return str(self.root / "eval_manifest.json")

    def get_detections_path(self):
        return str(self.root / "detections.json")

    def get_detections_anno_path(self):
        return str(self.root / "detections_anno.json")

    def get_log_path(self, tool_name=None):
        if tool_name:
            return str(self.root / f"log_{tool_name}.txt")
        return str(self.root / "log.txt")

    def get_log_dryrun_path(self, tool_name=None):
        if tool_name:
            return str(self.root / f"log_dryrun_{tool_name}.txt")
        return str(self.root / "log_dryrun.txt")

    def get_metrics_path(self):
        return str(self.root / "metrics.json")

    def get_predictions_path(self):
        return str(self.root / "predictions.json")

    def get_sample_detections_dir(self):
        return str(self.root / "sample_detections")


class ModelManager:
    # TODO: See https://wiki-int.sei.cmu.edu/confluence/display/CYBINV/Juneberry+Refactoring+Sketchboard
    def __init__(self, model_name, model_version=None, *, platform: str = None):
        """
        :param model_name: Name of model directory
        :param model_version: The version of the model.
        :param platform: The platform of the model. Useful when creating in-memory or new models.
        """
        if model_version is None:
            model_version = ""
        self.model_name = model_name
        self.model_version = model_version
        self.model_dir_path = Path('models') / self.model_name / self.model_version
        self.ensure_model_directory()
        if platform is not None:
            self.model_platform = platform
        else:
            self.model_platform = self.set_model_platform()
        self.model_task = self.set_model_task()

    def setup(self):
        """ Prepares a directory for use. """
        self.model_dir_path.mkdir(parents=True, exist_ok=True)
        for name in ['train', 'eval']:
            (self.model_dir_path / name).mkdir(parents=True, exist_ok=True)

    def get_model_name(self):
        """ :return: The name of the model. """
        return self.model_name

    def get_model_dir(self):
        """ :return: The path to the model directory within the workspace root. """
        return self.model_dir_path

    def get_model_platform(self):
        """ : return: The platform of the model. """
        return self.model_platform

    def get_plots_dir(self):
        """ :return: The path to the plots directory within the model directory. """
        return self.model_dir_path / 'plots'

    def get_model_path(self):
        """ :return: The path to a pytorch-compatible model file. """
        if self.model_platform in ['tensorflow']:
            return self.get_tensorflow_model_path()
        else:
            return self.get_pytorch_model_path()

    def get_pytorch_model_path(self):
        """ :return: The path to a pytorch-compatible model file. """
        return self.model_dir_path / 'model.pt'

    def get_onnx_model_path(self):
        """ :return: The path to an ONNX compatible model file. """
        return self.model_dir_path / 'model.onnx'

    def get_tensorflow_model_path(self):
        """ :return: The path to a tensorflow-compatible model file. """
        return self.model_dir_path / 'model.h5'

    def get_detectron2_model_path(self):
        """ :return: The path to a detectron2-compatible model file. """
        return self.get_train_scratch_path() / 'model_final.pth'

    def get_pytorch_model_summary_path(self):
        """ :return: The path to model summary file. """
        return self.model_dir_path / 'model_summary.txt'

    def get_model_config(self):
        """ :return: The path to the model's config file. """
        return self.model_dir_path / 'config.json'

    def get_known_results(self):
        """ :return: The path to the model's known results file. """
        return self.model_dir_path / 'known_ut_results.json'

    def get_latest_results(self):
        """ :return: The path to the model's latest results file. """
        return self.model_dir_path / 'latest_ut_results.json'

    # ============ TRAINING ============

    def get_train_root_dir(self):
        """ :return: The path to the training directory within the model directory. """
        return self.model_dir_path / "train"

    def get_dryrun_imgs_dir(self):
        """ :return: The path to the dryrun_imgs directory. """
        return self.get_train_root_dir() / 'dryrun_imgs'

    def get_training_out_file(self):
        """ :return: The path to the model's training output file. """
        return self.get_train_root_dir() / 'output.json'

    def get_training_summary_plot(self):
        """ :return: The path to the model's training plot. """
        return self.get_train_root_dir() / 'output.png'

    def get_training_log(self):
        """ :return: The path to the model's training log. """
        return self.get_train_root_dir() / 'log.txt'

    def get_training_dryrun_log_path(self):
        """ :return: The path to the model's training dryrun log. """
        return self.get_train_root_dir() / 'log_dryrun.txt'

    def get_platform_training_config(self, extension: str = 'json') -> Path:
        """ :return: Path to the config (after modification) that was actually used to train. """
        # TODO: Add in sensitivity to platform for extension, for now we take it as
        #  an optional argument as a compromise.
        if self.model_platform == Platforms.DT2:
            return self.get_train_root_dir() / "platform_config.yaml"
        return self.get_train_root_dir() / ("platform_config." + extension)

    def get_training_data_manifest_path(self) -> Path:
        """ :return: Path to a file that contains a manifest of the data to use for training. """
        return self.get_train_root_dir() / "training_manifest.json"

    def get_validation_data_manifest_path(self) -> Path:
        """ :return: Path to a file that contains a manifest of the data to use for validation. """
        return self.get_train_root_dir() / "validation_manifest.json"

    def get_train_scratch_path(self) -> Path:
        """ :return: Path to a directory for 'scratch' outputs from training. """
        return self.get_train_root_dir() / "scratch"

    def get_tuning_dir(self) -> Path:
        """ :return: Path to a directory containing files related to hyperparameter tuning. """
        return self.get_train_root_dir() / "tuning"

    def get_tuning_checkpoint_dir(self) -> Path:
        """ :return: Path to a directory containing model checkpoints from hyperparameter tuning. """
        return self.get_tuning_dir() / "checkpoints"

    # ============ Evaluation ============

    def get_eval_dir_mgr(self, dataset_path: str = None) -> EvalDirMgr:
        """
        Construct an EvalDirMgr object based on the current model and the provided dataset path.
        :param dataset_path: The path to the dataset file.
        :return: An EvalDirMgr object.
        """
        dataset_arg = Path(dataset_path).stem if dataset_path else None
        return EvalDirMgr(self.model_dir_path, self.model_platform, dataset_arg)

    def iter_eval_dirs(self):
        """
        :return: A generator over the eval directory return eval dir managers.
        """
        eval_dir = Path(self.get_eval_root_dir())
        for item in eval_dir.iterdir():
            if item.is_dir():
                yield EvalDirMgr(self.model_dir_path, self.model_platform, item.name)

    # ============ Misc ============

    def set_model_platform(self):
        """ :return: The platform of the model, e.g. pytorch, detectron2, others."""
        if self.get_model_config().exists():
            # We need to know the "platform" because it changes the layout.
            # We get the platform from the file but we don't want to do a full ModelConfig instantiation
            # so we just peek into the data as a struct.
            data = load_file(self.get_model_config())
            platform = data.get('platform', None)
            if platform not in ['pytorch', 'detectron2', 'mmdetection', 'pytorch_privacy', 'tensorflow']:
                # TODO: Should this be an error? We need to try it and run full tests
                logger.warning(f"Unknown platform '{platform}' found in: {self.get_model_config()}")
            return platform
        else:
            return None

    def set_model_task(self):
        """: return: The task the model is for, e.g. classification, objectDetection."""
        # Load the model config (if it exists).
        if self.get_model_config().exists():
            data = load_file(self.get_model_config())

            # Get the value for the task property (if it exists).
            task = data.get('task', None)

            # Log a warning if a task type wasn't found and return the task.
            if task is None:
                logger.warning(f"Task was not found in {self.get_model_config()}")
            return task

    # ============ MMDetection ============

    def get_mmd_latest_model_path(self):
        """ :return: The path to the latest mmdetection model."""
        return self.get_train_scratch_path() / 'latest.pth'

    # ============ New Style Evaluation Output ============
    # TODO: Replace these with the eval dir stuff when we get there

    def get_eval_root_dir(self):
        """ :return: The path to the eval root directory."""
        return self.model_dir_path / 'eval'

    def get_eval_dir(self, dataset_path):
        """
        :param dataset_path: The Path of the dataset being evaluated.
        :return: The path to the location within the eval root directory
         corresponding to the dataset being evaluated.
        """
        return self.get_eval_root_dir() / Path(dataset_path).stem

    def get_platform_eval_config(self, dataset_path, extension: str = 'json') -> Path:
        """ :return: Path the the config (after modification) that was actually used to evaluate."""
        # TODO: Add in sensitivity to platform for extension, for now we take it as
        #  an optional argument as a compromise.
        return self.get_eval_dir(dataset_path) / ("platform_config." + extension)

    def get_eval_manifest_path(self, dataset_path) -> Path:
        """
        :param dataset_path: The name of the dataset being evaluated.
        :return: The path to the manifest that lists the contents to be evaluated.
        """
        return self.get_eval_dir(dataset_path) / "eval_manifest.json"

    # ============ Misc ============

    def get_model_diagram(self):
        """ :return: The path to the model's diagram file """
        return self.model_dir_path / 'model_diagram.png'

    # ============ OLD STYLE EVAL - TRAINING ============

    def get_dryrun_imgs_path(self, image_name):
        """ :return: The path to image in the dry run directory """
        return self.get_dryrun_imgs_dir() / f'{image_name}.png'

    def ensure_model_directory(self) -> None:
        """ Checks for the existence of the model directory and exits if the model directory doesn't exist. """
        if not self.model_dir_path.exists():
            logger.error(f"Model directory '{self.model_dir_path}' does not exist. Exiting.")
            exit(-1)
        else:
            self.setup()

    def get_training_output_list(self):
        """
        :return: A list of files or glob patterns of files generated during training.
        """
        files = [self.get_model_path(),
                 self.get_training_out_file(),
                 self.get_training_log(),
                 self.get_train_root_dir(),
                 self.get_training_summary_plot()]

        if self.model_platform in ['detectron2', 'mmdetection']:
            ext = "py" if self.model_platform == Platforms.MMD else "yaml"
            more_files = [self.get_platform_training_config(ext),
                          self.get_training_data_manifest_path(),
                          self.get_validation_data_manifest_path()]
            files.extend(more_files)

        elif self.model_platform in ['tensorflow']:
            more_files = [self.get_training_data_manifest_path(),
                          self.get_validation_data_manifest_path()]
            files.extend(more_files)

        return files

    @staticmethod
    def get_training_clean_extras_list() -> list:
        """
        :return: A list of additional things to clean.
        """
        return []

    def get_dry_run_output_list(self):
        """
        :return: A list of files or glob patterns of files generated during a dry run.
        """

        files = [self.get_pytorch_model_summary_path(),
                 self.get_training_dryrun_log_path(),
                 self.get_dryrun_imgs_dir()]

        if self.model_platform in ['tensorflow']:
            files.extend([self.get_training_data_manifest_path(),
                          self.get_validation_data_manifest_path()])
        return files

    def get_dry_run_clean_extras_list(self) -> list:
        """
        :return: A list of additional things to clean.
        """
        return [self.get_dryrun_imgs_path("*")]

    def get_evaluation_output_list(self, data_set):
        """
        :return: A list of files or glob patterns of files that are produced during an evaluation.
        """
        eval_dir_mgr = self.get_eval_dir_mgr(data_set)
        return [
            eval_dir_mgr.root,
            eval_dir_mgr.get_predictions_path(),
            eval_dir_mgr.get_log_path(),
            eval_dir_mgr.get_metrics_path()
        ]

    @staticmethod
    def get_predictions_clean_extras_list() -> list:
        """
        :return: A list of additional things to clean.
        """
        return []

    def clean(self, dry_run=False):
        files_to_clean = []

        cwd = Path('.')
        for data_list in (self.get_training_output_list(),
                          self.get_dry_run_output_list(),
                          self.get_evaluation_output_list("*")):
            for item in data_list:
                if Path(item).is_dir():
                    files_to_clean.append(item)
                    item = Path(item, '*')
                if "*" in str(item):
                    files_to_clean.extend(list(cwd.glob(str(item))))
                else:
                    files_to_clean.append(item)

        clean_if_exists(files_to_clean, dry_run)

    def get_tuning_output_list(self) -> list:
        """
        :return: A list of files or glob patterns of files generated during tuning.
        """
        files = []

        return files

    def get_tuning_clean_extras_list(self) -> list:
        """
        :return: A list of additional things to clean after tuning.
        """
        return []

    def get_tuning_dryrun_output_list(self) -> list:
        """
        :return: A list of files or glob patterns of files generated during tuning.
        """
        files = []

        return files

    def get_tuning_dryrun_clean_extras_list(self) -> list:
        """
        :return: A list of additional things to clean after tuning.
        """
        return []

    def create_tensorboard_directory_name(self, tensorboard_root):
        time_str = datetime.datetime.now().strftime('%m%d_%H%M') + '_'
        hostname_str = socket.gethostname() + '_'
        return os.path.join(tensorboard_root, time_str + hostname_str + self.model_name)


class DataManager:
    def __init__(self, data_config, version=None):
        # TODO: This should be dataGroupName
        # TODO: This needs to be added to docs.
        # TODO: Should we accept the data root or data roots?
        if 'dataSetPath' in data_config.keys():
            self.name = data_config["dataSetPath"]
        else:
            self.name = ''

        # version is passed in separately from data_config to allow for for the same data conf
        # to be used in different versions of the experiment.
        self.version = version
        if self.version is None:
            self.version = ''

        self.version_path = Path(self.name) / self.version

    def get_directory_path(self, directory):
        """
        Generates and returns the Path of the directory holding desired files.
        :param directory: Data directory for files
        :return: Path object for directory holding files
        """
        return self.version_path / directory

    def get_file_path(self, directory, relative_path):
        """
        Generates and returns the Path of the file under the directory and path given
        :param directory: Data directory for files
        :param relative_path: Path from directory to cached file
        :return: Path object for file
        """
        return self.get_directory_path(directory) / relative_path


def generate_file_hash(filepath):
    """
    Generates a standard hash string for the file.
    :param filepath: The path to the file.
    :return: The hash string
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as in_file:
        buf = in_file.read()
        if sys.platform == "win32":
            # remove \r's not present in Unix
            buf = buf.replace("\r".encode(), "".encode())
        hasher.update(buf)

    return hasher.hexdigest()


def json_cleaner(obj):
    """
    Converts an object to a version that's JSON-serializable.
    :param obj: The object to be converted.
    :return: A version of the object that can be written to JSON.
    """
    if isinstance(obj, Path):
        return str(obj)
    elif type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # This unpacks floats, etc.
            return obj.item()

    # Throw a type error
    raise TypeError('json_cleaner unknown type:', type(obj))
