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

from contextlib import contextmanager
import datetime
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys

import hjson
import numpy as np
import toml
import yaml
from yaml import Loader, Dumper

from juneberry.platform import PlatformDefinitions

logger = logging.getLogger(__name__)


def clean_if_exists(files_to_clean, dry_run=False):
    """
    Removes file or folder if it exists
    :param files_to_clean: Array of Path objects to remove
    :param dry_run: Flag to commit cleaning
    """
    # Paths are sorted in reverse order so that when directories are removed they will already be empty
    files = sorted(files_to_clean, reverse=True)
    msg = "  Removing"
    if dry_run:
        msg = "  Would remove"
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
    Save the data to the specified path (string or Path) in HJSON format, applying the specified
    indent (int), and converting all the traditional non encoding bits (e.g. Path, numpy) to
    the appropriate data structure.
    :param data: The data to save.
    :param json_path: The path to save the data to.
    :param indent: The indent spacing to use; with a default of 4.
    :return: None
    """
    with open_file(json_path, "wt") as json_file:
        hjson.dump(data, json_file, indent=indent, default=json_cleaner, sort_keys=True)


def save_json(data, json_path, *, indent: int = 4) -> None:
    """
    Save the data to the specified path (string or Path) in JSON format, applying the specified
    indent (int), and converting all the traditional non encoding bits (e.g. Path, numpy) to
    the appropriate data structure.
    :param data: The data to save.
    :param json_path: The path to save the data to.
    :param indent: The indent spacing to use; with a default of 4.
    :return: None
    """
    with open_file(json_path, "wt") as json_file:
        json.dump(data, json_file, indent=indent, default=json_cleaner, sort_keys=True)


def save_yaml(data, yaml_path) -> None:
    """
    Save the data to the specified path (string or Path) in YAML format.
    :param data: The data to save.
    :param yaml_path: The path to save the data to.
    :return: None
    """
    with open_file(yaml_path, 'wt') as yaml_file:
        yaml.dump(data, yaml_file, Dumper=Dumper)


def save_toml(data, toml_path) -> None:
    """
    Save the data to the specified path (string or Path) in TOML format.
    :param data: The data to save.
    :param toml_path: The path to save the data to.
    :return: None
    """
    with open_file(toml_path, 'wt') as toml_file:
        toml.dump(data, toml_file)


@contextmanager
def open_file(path, mode='r') -> str:
    """
    Opens files using file handlers according to the file extension. The base case is to use
    the python default 'open'.
    :param path: Path to the file to be opened
    :param mode: Mode to open the file
    :yield: File object
    """
    ext = Path(path).suffix.lower()
    if ext in {'.gzip', '.gz'}:
        with gzip.open(path, mode) as file:
            yield file
    else:
        with open(path, mode, encoding="utf8") as file:
            yield file


def save_file(data, path: str, *, indent: int = 4) -> None:
    """
    Generic file saver that chooses the file format based on the extension of the path.
    :param data:
    :param path:
    :param indent: The indent spacing to use; with a default of 4.
    :return: None
    """
    exts = Path(path).suffixes
    ext = exts[0].lower()  # the file type should be the left most of the suffixes
    if ext == '.json':
        save_json(data, path, indent=indent)
    elif ext == '.hjson':
        save_hjson(data, path, indent=indent)
    elif ext in {'.yaml', '.yml'}:
        save_yaml(data, path)
    elif ext in {'.toml', '.tml'}:
        save_toml(data, path)
    else:
        logger.error(f'Unsupported file extension {ext}')
        sys.exit(-1)


def load_file(path: str):
    """
    Loads the file from the specified file path via the specific loader.
    :param path: The path to the file to load.
    :return: File contents as a dict
    """
    if Path(path).exists():
        exts = Path(path).suffixes
        ext = exts[0].lower()  # the file type should be the left most of the suffixes
        with open_file(path, 'rt') as file:
            if ext in {'.json', '.hjson'}:
                # HJSON is a superset of JSON, so the HJSON parser can handle both cases.
                return hjson.load(file)
            if ext in {'.yaml', '.yml'}:
                return yaml.load(file, Loader=Loader)
            if ext in {'.toml', '.tml'}:
                return toml.load(file)
            else:
                logger.error(f'Unsupported file extension {ext}')
                sys.exit(-1)
    else:
        gz_path = Path(path)
        gz_path = gz_path.with_suffix(gz_path.suffix + '.gz')
        if gz_path.exists():
            logger.info(f"Could not find '{path}'. Using GZIP of '{path}'.")
            return load_file(str(gz_path))
        else:
            logger.error(f'Failed to load {path}. The file could not be found. Exiting.')
            sys.exit(-1)


def load_json_lines(path: str) -> list:
    """
    The purpose of this function is to load content from the specified file path, when the
    target file contains data in JSON Lines format.
    :param path: The path to the JSON Lines file to load.
    :return: A list of data, where each element in the list is a valid JSON object.
    """
    with open(path, 'r') as file:
        return [json.loads(line) for line in file]


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

    def get_experiment_datasets_dir(self):
        """ :return: The path to the directory containing datasets specific to the experiment. """
        return self.experiment_dir_path / "data_sets"

    def get_experiment_base_dataset(self):
        """ :return: The path to the experiment's base dataset config. """
        return self.experiment_dir_path / "base_data_set.json"

    def get_experiment_dataset_path(self, dataset_name: str):
        """ :return: The path to a dataset file inside the experiment's data_sets directory. """
        return self.get_experiment_datasets_dir() / f"{dataset_name}.json"

    def get_experiment_watermarks_dir(self):
        """ :return: The path to the directory containing the experiment's watermarks. """
        return self.experiment_dir_path / "watermarks"

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

    def get_log_path(self, dryrun=False):
        """ :return: The path to the experiment log file. """
        if dryrun:
            return self.get_dryrun_log_path()
        return self.experiment_log_dir_path / "log_experiment.txt"

    def get_dryrun_log_path(self):
        """ :return: The path to the experiment log file. """
        return self.experiment_log_dir_path / "log_experiment_dryrun.txt"

    def get_output_list(self):
        """
        :return: Returns a list of files or glob patterns of outputs.
        """
        return [self.get_experiment_db_file(),
                self.get_experiment_rules(),
                self.get_experiment_dodo(workflow='main'),
                self.get_experiment_dodo(workflow='dryrun'),
                self.get_experiment_reports_dir()]

    def clean(self, dry_run=False):
        """
        Cleans all files and folders in the experiment directory that were generated by the framework except for
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

    def get_log_path(self, dryrun=False):
        """ :return: The path to the experiment creation log. """
        if dryrun:
            return self.get_dryrun_log_path()
        return self.experiment_dir_path / 'log_experiment_creation.txt'

    def get_dryrun_log_path(self):
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
    @staticmethod
    def get_base_path(root):
        return Path(root) / 'eval'

    @staticmethod
    def get_base_log_path(root):
        return Path(root) / 'logs' / 'eval'

    @staticmethod
    def get_path(root, dataset_name):
        return EvalDirMgr.get_base_path(root) / dataset_name if dataset_name else Path(root) / 'eval'

    @staticmethod
    def get_log_dir_path(root, dataset_name):
        return EvalDirMgr.get_base_log_path(root) / dataset_name if dataset_name else Path(root) / 'logs' / 'eval'

    def __init__(self, root: str, dataset_name: str = None) -> None:
        """
        Constructs an EvalDirMgr object rooted at the root path for the specified platform
        with the given name.
        NOTE: These are usually created via the ModelManager's "get_eval_dir_mgr" method.
        :param root: The root directory.
        :param dataset_name: The name of the evaluation dataset.
        """
        # TODO: Why should the eval dir manager point to the root eval directory?
        #  This seems like an error somewhere else
        self.root = EvalDirMgr.get_path(root, dataset_name)
        self.log_dir = EvalDirMgr.get_log_dir_path(root, dataset_name)

    def setup(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_dir(self):
        return self.root

    def get_scratch_dir(self):
        return self.root / "scratch"

    def get_dryrun_imgs_dir(self):
        return str(self.root / "dryrun_imgs")

    def get_platform_config(self, suffix: str = '.json'):
        return str(self.root / f"platform_config{suffix}")

    def get_manifest_path(self):
        return str(self.root / "eval_manifest.json")

    def get_detections_path(self):
        return str(self.root / "detections.json")

    def get_detections_anno_path(self):
        return str(self.root / "detections_anno.json")

    def get_log_path(self, tool_name=None, dryrun=False):
        if dryrun:
            return self.get_dryrun_log_path(tool_name)
        if tool_name:
            return str(self.log_dir / f"log_{tool_name}.txt")
        return str(self.log_dir / "log.txt")

    def get_dryrun_log_path(self, tool_name=None):
        if tool_name:
            return str(self.log_dir / f"log_dryrun_{tool_name}.txt")
        return str(self.log_dir / "log_dryrun.txt")

    def get_metrics_path(self):
        return str(self.root / "metrics.json")

    def get_predictions_path(self):
        return str(self.root / "predictions.json")

    def predictions_exists(self) -> bool:
        """ :return: True if predictions exists as json or json.gz """
        pred_path = Path(self.get_predictions_path())
        if pred_path.exists():
            return True
        pred_path = Path(self.get_predictions_path() + ".gz")
        if pred_path.exists():
            return True
        return False

    def get_sample_detections_dir(self):
        return str(self.root / "sample_detections")

    def get_instances_predictions_file(self):
        """ :return: The location to the instances_predictions file generated during a
        Detectron2 evaluation. """
        return str(self.root / "instances_predictions.pth")


class ModelManager:
    def __init__(self, model_name):
        """
        :param model_name: Name of model directory
        """
        self.model_name = model_name
        self.model_dir_path = Path('models') / self.model_name
        self.model_log_dir_path = self.model_dir_path / 'logs'

    def setup(self):
        """ Prepares a directory for use. """
        self.model_dir_path.mkdir(parents=True, exist_ok=True)

    def setup_training(self):
        """ Prepares a model directory for training. """
        self.setup()
        (self.model_dir_path / 'train').mkdir(parents=True, exist_ok=True)
        (self.model_log_dir_path / 'train').mkdir(parents=True, exist_ok=True)

    def setup_tuning(self):
        """ Prepares a model directory for tuning. """
        self.get_tuning_dir().mkdir(parents=True, exist_ok=True)
        (self.model_log_dir_path / 'train' / 'tuning').mkdir(parents=True, exist_ok=True)

    def get_model_name(self):
        """ :return: The name of the model. """
        return self.model_name

    def get_model_dir(self):
        """ :return: The path to the model directory within the workspace root. """
        return self.model_dir_path

    def get_plots_dir(self):
        """ :return: The path to the plots directory within the model directory. """
        return self.model_dir_path / 'plots'

    def get_model_path(self, platform_def: PlatformDefinitions):
        return self.model_dir_path / platform_def.get_model_filename()

    def get_model_summary_path(self):
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

    def get_train_log_dir(self):
        """ :return: The path to the training log directory within the model directory. """
        return self.model_log_dir_path / "train"

    def get_dryrun_imgs_dir(self):
        """ :return: The path to the dryrun_imgs directory. """
        return self.get_train_root_dir() / 'dryrun_imgs'

    def get_training_out_file(self):
        """ :return: The path to the model's training output file. """
        return self.get_train_root_dir() / 'output.json'

    def get_training_summary_plot(self):
        """ :return: The path to the model's training plot. """
        return self.get_train_root_dir() / 'output.png'

    def get_training_log(self, dryrun=False):
        """ :return: The path to the model's training log. """
        if dryrun:
            return self.get_training_dryrun_log_path()
        return self.get_train_log_dir() / 'log.txt'

    def get_training_dryrun_log_path(self):
        """ :return: The path to the model's training dryrun log. """
        return self.get_train_log_dir() / 'log_dryrun.txt'

    def get_platform_training_config(self, platform_defs: PlatformDefinitions) -> Path:
        """
        :param platform_defs: The definitions for the platform
        :return: Path to the platform config (after modification) that was actually used to train.
        """
        return self.get_train_root_dir() / f"platform_config{platform_defs.get_config_suffix()}"

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

    def get_tuning_log_dir(self) -> Path:
        """ :return: Path to a directory containing a copy of the tuning log files. """
        return self.get_train_log_dir() / "tuning"

    def get_tuning_log(self) -> Path:
        """ :return: The path to the model's tuning log. """
        return self.get_tuning_dir() / "log.txt"

    @staticmethod
    def get_relocated_tuning_log(target_dir: str, prefix: str = None) -> Path:
        """ :return: The path for a tuning log file that's been relocated to a target directory. """
        if prefix:
            return Path(target_dir) / f"{prefix}_log.txt"
        else:
            return Path(target_dir) / "log.txt"

    @staticmethod
    def get_relocated_tuning_output(target_dir: str) -> Path:
        """ :return: The path for a tuning output file that's been relocated to a target directory. """
        return Path(target_dir) / "output.json"

    @staticmethod
    def get_tuning_result_file(target_dir: str) -> Path:
        """ :return: The path for a tuning result file inside a target directory. """
        return Path(target_dir) / "result.json"

    # ============ Evaluation ============

    def get_eval_dir_mgr(self, dataset_path: str = None) -> EvalDirMgr:
        """
        Construct an EvalDirMgr object based on the current model and the provided dataset path.
        :param dataset_path: The path to the dataset file.
        :return: An EvalDirMgr object.
        """
        # TODO: Why do we support dataset_path of None?
        #  This seems like an error somewhere else
        # dataset_arg = Path(dataset_path).stem if dataset_path else None
        dataset_arg = None
        if dataset_path is not None:
            p = Path(dataset_path)
            if p.parts[0] == "data_sets":
                dataset_arg = Path(*p.parts[1:-1]) / p.stem
            else:
                dataset_arg = Path(*p.parts[:-1]) / p.stem
        return EvalDirMgr(self.model_dir_path, dataset_arg)

    def iter_eval_dirs(self):
        """
        :return: A generator over the eval directory which returns EvalDirMgrs.
        """
        eval_dir = EvalDirMgr.get_base_path(self.model_dir_path)
        for item in eval_dir.iterdir():
            if item.is_dir():
                yield EvalDirMgr(self.model_dir_path, item.name)

    # ============ MMDetection ============

    def get_mmd_latest_model_path(self):
        """ :return: The path to the latest mmdetection model."""
        return self.get_train_scratch_path() / 'latest.pth'

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
        if not os.path.exists(self.model_dir_path):
            logger.error(f"Model directory '{self.model_dir_path}' does not exist!! EXITING!!")
            exit(-1)

    def create_tensorboard_directory_name(self, tensorboard_root):
        time_str = datetime.datetime.now().strftime('%m%d_%H%M') + '_'
        hostname_str = socket.gethostname() + '_'
        return os.path.join(tensorboard_root, time_str + hostname_str + self.model_name)


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
