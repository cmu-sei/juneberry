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

import logging
import sys
import re
import os
from prodict import Prodict
from pathlib import Path

import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class LabProfile(Prodict):
    """
    A class to validate and manage the lab profile pulled from some config file.
    """
    num_workers: int
    num_gpus: int
    max_gpus: int
    no_paging: bool

    def _finish_init(self) -> None:
        """
        Validate the lab profile.
        :return: None
        """
        error_count = 0

        # Check that the number of gpus doesn't exceed the number of cuda visible devices
        visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_gpus is not None:
            visible_gpus = visible_gpus.count(",") + 1
            if self.num_gpus is not None:
                if self.num_gpus > visible_gpus:
                    error_count += 1

        # If errors found, report and exit
        if error_count > 0:
            logger.error(f"Found {error_count} errors in machine specs. EXITING.")
            sys.exit(-1)

    # This is the Prodict init method to set default values
    def init(self):
        # For most platform 4 is a reasonable default
        self.num_workers = 4
        self.no_paging = False

    @staticmethod
    def construct(data: dict):
        """
        Load, validate, and construct a LabProfile object
        :param data: The data to use to construct the object.
        :return: The constructed object.
        """

        # Construct the LabProfile object
        specs = LabProfile.from_dict(data)
        specs._finish_init()
        return specs

    @staticmethod
    def update_properties(profile_name: str, model_name: str, config_data: dict, profile_data: dict):
        """
        Update the LabProfile dictionary to include property overrides.
        :param profile_name: The profile name.
        :param model_name: The model name.
        :param config_data: The config data.
        :param profile_data: The dictionary of specs for execution.
        :return: The LabProfile object.
        """

        if profile_name in config_data:
            for key in config_data[profile_name].keys():
                if re.match(key, model_name):
                    if "include" in config_data[profile_name][key]:
                        temp_machine, temp_model = config_data[profile_name][key]["include"].split(':')
                        profile_data = LabProfile.update_properties(temp_machine, temp_model, config_data, profile_data)
                    else:
                        profile_data.update(config_data[profile_name][key])

        return profile_data

    @staticmethod
    def validate_workspace_config(config_data: dict, test: bool = False):
        error_count = 0
        for profile_key in config_data:
            for model_key in config_data[profile_key]:
                # Validate include format
                if "include" in config_data[profile_key][model_key]:
                    include_lst = config_data[profile_key][model_key]["include"].split(':')
                    if len(include_lst) == 2:
                        profile_name = include_lst[0]
                        model_key = include_lst[1]
                        # Check for invalid machine:model pair
                        if profile_name not in config_data:
                            error_count += 1
                        else:
                            if model_key not in config_data[profile_name]:
                                error_count += 1
                            else:
                                # Check for chained includes
                                if "include" in config_data[profile_name][model_key]:
                                    error_count += 1
                    else:
                        # Invalid include format
                        error_count += 1

        if error_count > 0:
            logger.error(f"Found {error_count} errors in workspace config. EXITING.")
            if test:
                return False
            else:
                sys.exit(-1)

        elif test:
            return True

    @staticmethod
    def load_file(path):
        # Load the workspace config file, if  it exists
        if Path(path).exists():
            return jbfs.load_file(path)
        else:
            return {}

    @staticmethod
    def load(data_path: str, profile_name: str = None, model_name: str = None, test: bool = False):
        """
        Loads the config file from the provided path, validates, and constructs the LabProfile object.
        :param data_path: Path to the config file.
        :param profile_name: The name of the profile.
        :param model_name: The name of the model.
        :param test: Flag for unit testing.
        :return: Loaded, validated, and constructed LabProfile object.
        """
        logger.info(f"Loading lab profile for name='{profile_name}', model='{model_name}'.")

        # Load the raw file.
        config_data = LabProfile.load_file(data_path)
        profile_data = {}

        # Validate config
        result = LabProfile.validate_workspace_config(config_data, test=test)
        if test:
            return result

        # Load from default:default
        profile_data = LabProfile.update_properties(profile_name="default", model_name="default",
                                                    config_data=config_data, profile_data=profile_data)

        # Load from machine:default
        if profile_name:
            profile_data = LabProfile.update_properties(profile_name=profile_name, model_name="default",
                                                        config_data=config_data, profile_data=profile_data)

        # Load from default:model
        if model_name:
            profile_data = LabProfile.update_properties(profile_name="default", model_name=model_name,
                                                        config_data=config_data, profile_data=profile_data)

        # Load from machine:model
        if profile_name and model_name:
            profile_data = LabProfile.update_properties(profile_name=profile_name, model_name=model_name,
                                                        config_data=config_data, profile_data=profile_data)

        # Construct the specs object
        return LabProfile.construct(profile_data)
