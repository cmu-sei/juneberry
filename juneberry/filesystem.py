#! /usr/bin/env python3

"""
Script that supports the model layout.

NOTE: These paths are all relative to the juneberry workspace root.
"""

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

import logging
import os
from pathlib import Path
import sys
import datetime
import socket
import hashlib


def version_check(data_type: str, config_str, latest_str, use_revision=False):
    """
    Logs an error if a) config version not found, b) config version number is greater than latest version, or c) minor
    or major are different between versions. Logs a warning if versions are the same except for the revision field.
    Versions should be in the form major.minor.revision.
    param data_type: the kind of config file being checked, e.g. experiment, experiment outline, etc.
    param config_str: the formatVersion found in the config file
    param latest_str: the latest formatVersion as specified in the documentation
    param use_revision: set to True if revision field is in use
    """

    if (config_str is None) or (config_str == ''):
        logging.error(f"Failed to find formatVersion in {data_type} config")
        sys.exit(-1)

    else:
        config_num = VersionNumber(config_str, use_revision)
        latest_num = VersionNumber(latest_str, use_revision)

        if config_num > latest_num:
            logging.error(f"{data_type} config formatVersion {config_str} greater than latest version {latest_str}")
            sys.exit(-1)

        elif config_num.major != latest_num.major or config_num.minor != latest_num.minor:
            logging.error(f"{data_type} config formatVersion {config_str} does not match latest version {latest_str}")
            sys.exit(-1)

        elif use_revision:
            if config_num.revision != latest_num.revision:
                logging.warning(f"{data_type} config formatVersion {config_str} "
                                f"revision field does not match latest version {latest_str}")


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
            logging.info(f"{msg}: {file}")
            if not dry_run:
                if file.is_dir():
                    is_empty = not any(file.iterdir())
                    if is_empty:
                        file.rmdir()
                    else:
                        logging.error(f"\tDirectory {file} is not empty so removal failed!")
                else:
                    file.unlink()


class ExperimentCreator:
    def __init__(self, experiment_name):
        """
        :param experiment_name: Name of the experiment directory.
        """
        self.experiment_name = experiment_name
        self.experiment_dir_path = Path('experiments') / self.experiment_name
        self.experiment_config_dir_path = Path('models') / self.experiment_name

    def get_experiment_creation_log(self):
        """ :return: The path to the experiment creation log """
        return self.experiment_dir_path / 'log_experiment_creation.txt'

    def get_experiment_creation_dryrun_log_path(self):
        """ :return: The path to the dry run of the experiment creation log """
        return self.experiment_dir_path / 'log_experiment_creation_dryrun.txt'

    def ensure_experiment_directory(self) -> None:
        """ Checks for the existence of the experiment directory and exits if the directory doesn't exist. """
        if not os.path.exists(self.experiment_dir_path):
            logging.error(f"Experiment directory '{self.experiment_dir_path}' does not exist!! EXITING!!")
            exit(-1)

    def get_experiment_outline(self):
        """ :return: The path to the model's config file. """
        return self.experiment_dir_path / 'experiment_outline.json'

    def get_experiment_config_dir(self):
        """ :return: The path to the directory containing all of the experiment configs."""
        return self.experiment_config_dir_path

    def get_experiment_config_file(self):
        """ :return: The path to the experiment's config file. """
        return self.experiment_dir_path / 'config.json'


class ModelManager:
    def __init__(self, model_name, model_version=None):
        """
        :param model_name: Name of model directory
        :param model_version: The version of the model.
        """
        if model_version is None:
            model_version = ""
        self.model_name = model_name
        self.model_version = model_version
        self.model_dir_path = Path('models') / self.model_name / self.model_version

    def get_model_name(self):
        """ :return: The name of the model. """
        return self.model_name

    def get_model_version(self):
        """ :return: The version of the model. """
        return self.model_version

    def get_model_dir(self):
        """ :return: The path to the model directory within the workspace root. """
        return self.model_dir_path

    def get_plots_dir(self):
        """ :return: The path to the plots directory within the model directory. """
        return self.model_dir_path / 'plots'

    def get_dryrun_imgs_dir(self):
        """ :return: The path to the dryrun_imgs directory """
        return self.model_dir_path / 'dryrun_imgs'

    def get_pytorch_model_file(self):
        """ :return: The path to model file """
        return self.model_dir_path / 'model.pt'

    def get_pytorch_model_summary_file(self):
        """ :return: The path to model summary file """
        return self.model_dir_path / 'model_summary.txt'

    def get_model_config(self):
        """ :return: The path to the model's config file """
        return self.model_dir_path / 'config.json'

    def get_model_diagram(self):
        """ :return: The path to the model's diagram file """
        return self.model_dir_path / 'model_diagram.png'

    def get_training_out_file(self):
        """ :return: The path to the model's config file """
        return self.model_dir_path / 'train_out.json'

    def get_training_summary_plot(self):
        """ :return: The path to the model's training plot """
        return self.model_dir_path / 'train_out.png'

    def get_training_log(self):
        """ :return: The path to the model's training log """
        return self.model_dir_path / 'log_train.txt'

    def get_training_dryrun_log_path(self):
        """ :return: The path to the model's training log """
        return self.model_dir_path / 'log_train_dryrun.txt'

    def get_predictions(self, data_set):
        """
        :param data_set: Name of data set that the model tested
        :return: The path to the predictions based on that data set
        """
        return self.model_dir_path / f'predictions_{Path(data_set).stem}.json'

    def get_predictions_log(self, data_set):
        """
        :param data_set: Name of data set that the model tested
        :return: The path to the model's prediction log for that data set
        """
        return self.model_dir_path / f'log_predictions_{os.path.splitext(data_set)[0]}.txt'

    def get_classification_log(self, data_set):
        """
        :param data_set: Name of data set that the model classified
        :return: The path to the model's classification log for that data set
        """
        return self.model_dir_path / f'log_classifications_{os.path.splitext(data_set)[0]}.txt'

    def get_layer_filter_visualization_path(self, layer_name):
        """
        Get the name of the layer visualization in the plots dir.
        :param layer_name: The name of the layer to visualize
        :return: The path to the visualization for that layer's filters
        """
        return self.get_plots_dir() / f"{layer_name}_filters_visualization.png"

    def get_gradients_plot_path(self, epoch):
        """
        Returns the path to the gradient file for the specified epoch
        :param epoch: The epoch
        :return: Path to the gradient plot for this epoch
        """
        return self.get_plots_dir() / f"gradient_{epoch}.png"

    def get_activations_plot_path(self, epoch):
        """
        Returns the path to the activations file for the specified epoch
        :param epoch: The epoch
        :return: Path to the activations plot for this epoch
        """
        return self.get_plots_dir() / f"activation_{epoch}.png"

    def get_dryrun_imgs_path(self, image_name):
        """ :return: The path to image in the dry run directory """
        return self.get_dryrun_imgs_dir() / f'{image_name}.png'

    def ensure_model_directory(self) -> None:
        """ Checks for the existence of the model directory and exits if the model directory doesn't exist. """
        if not os.path.exists(self.model_dir_path):
            logging.error(f"Model directory '{self.model_dir_path}' does not exist!! EXITING!!")
            exit(-1)

    def clean(self, dry_run=False):
        """
        Cleans all files and folders in model directory that was generated by the framework except for 'config.json'
        :param dry_run: Flag to commit cleaning
        """
        files_to_clean = []
        files_to_clean.append(self.get_pytorch_model_file())
        files_to_clean.append(self.get_pytorch_model_summary_file())
        files_to_clean.append(self.get_model_diagram())
        files_to_clean.append(self.get_training_out_file())
        files_to_clean.append(self.get_training_summary_plot())
        files_to_clean.append(self.get_training_log())
        files_to_clean.append(self.get_training_dryrun_log_path())
        # Using glob to find any and all files that are specific to data sets, epochs, etc.
        cwd = Path('.')
        files_to_clean.extend(list(cwd.glob(str(self.get_predictions("*")))))
        files_to_clean.extend(list(cwd.glob(str(self.get_predictions_log("*")))))
        files_to_clean.extend(list(cwd.glob(str(self.get_layer_filter_visualization_path("*")))))
        files_to_clean.extend(list(cwd.glob(str(self.get_gradients_plot_path("*")))))
        files_to_clean.extend(list(cwd.glob(str(self.get_activations_plot_path("*")))))
        files_to_clean.extend(list(cwd.glob(str(self.get_dryrun_imgs_path("*")))))
        files_to_clean.append(self.get_dryrun_imgs_dir())

        clean_if_exists(files_to_clean, dry_run)

    def create_tensorboard_directory_name(self, tensorboard_root):
        time_str = datetime.datetime.now().strftime('%m%d_%H%M') + '_'
        hostname_str = socket.gethostname() + '_'
        return os.path.join(tensorboard_root, time_str + hostname_str + self.model_name)


class ExperimentManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_dir = Path('experiments') / experiment_name

    def get_experiment_name(self):
        """ :return: The name of the experiment """
        return self.experiment_name

    def _get_experiment_dir(self):
        """ :return: The experiment directory """
        return self.experiment_dir

    def get_experiment_config(self):
        """ :return: The relative path to the experiment's config file """
        return self.experiment_dir / 'config.json'

    def get_experiment_file(self, file_name):
        """
        :param file_name: Name of file
        :return: The relative path to the experiment's file
        """
        return self.experiment_dir / file_name

    def get_experiment_log_path(self):
        """ :return: The path to the experiment log file """
        return self.experiment_dir / "log_experiment.txt"

    def get_experiment_dryrun_log_path(self):
        """ :return: The path to the experiment log file """
        return self.experiment_dir / "log_experiment_dryrun.txt"

    def clean(self, dry_run=False):
        """
        Cleans all files and folders in model directory that were generated by the framework except for
        'log_experiment.txt' and 'config.json'.
        :param dry_run: Flag to commit cleaning
        """
        files_to_clean = [self.get_experiment_dryrun_log_path()]
        clean_if_exists(files_to_clean, dry_run)


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

    def check_cache(self, data_root, directory, relative_path):
        # TODO: Reimplement this
        return relative_path, False


class VersionNumber:
    """
    Class for comparing linux-like version numbers on config files.
    NOTE: By default revision is NOT required for comparison.
    """

    def __init__(self, version: str, use_revision=False):
        """
        Constructs version number object from a string with 2 or three fields.  The fields
        are major.minor.revision.
        :param version:
        :param use_revision:
        """
        self.version = version
        self.use_revision = use_revision
        fields = version.split('.')

        if len(fields) == 3:
            self.major = int(fields[0])
            self.minor = int(fields[1])
            if use_revision:
                self.int_version = int(fields[0]) << 16 | int(fields[1]) << 8 | int(fields[2])
                self.revision = int(fields[2])
            else:
                self.int_version = int(fields[0]) << 16 | int(fields[1]) << 8
        elif len(fields) == 2:
            self.major = int(fields[0])
            self.minor = int(fields[1])
            self.int_version = int(fields[0]) << 16 | int(fields[1]) << 8
            logging.warning(f"Given only 2-part version number. Please update! Version: {version}")
        else:
            logging.error(f"Given version string with {len(fields)} fields. We require 2 or 3! Version: {version}")
            sys.exit(-1)

    def __eq__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version == other.int_version

    def __ne__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version != other.int_version

    def __lt__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version < other.int_version

    def __le__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version <= other.int_version

    def __gt__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version > other.int_version

    def __ge__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version >= other.int_version


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
