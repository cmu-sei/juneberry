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

from collections import defaultdict
import logging
from pathlib import Path
import re
import sys

from prodict import Prodict, List

import juneberry.filesystem as jbfs
import juneberry.config.util as conf_utils

logger = logging.getLogger(__name__)


class LabProfile(Prodict):
    """
    A class to store host specific values used for the training and evaluation of models.
    This can also be embedded in a model config.
    """
    max_gpus: int
    no_paging: bool
    num_gpus: int
    num_workers: int


class ProfileStanza(Prodict):
    """
    A data structure to that add a profile-name and model-name and option include to a profile entry..
    """
    name: str
    model: str
    include: str
    profile: LabProfile


class WorkspaceConfig(Prodict):
    FORMAT_VERSION = '0.1.0'
    SCHEMA_NAME = 'workspace_schema.json'

    profiles: List[ProfileStanza]

    def _finish_init(self, file_path: str = None, test: bool = False) -> bool:
        """
        Validates the workspace files, where valid means:
        - No duplicate profile/model pairs
        - Every include references a valid entry
        NOTE: Structure validation is handled by the schema
        :param test: Set to true to not exit.
        :return: Boolean indicating succes
        """
        error_list = []

        # If we don't have a profiles section, add and empty one to make the everything simpler.
        if self.profiles is None:
            self.profiles = []

        # Build profile-name map checking for duplicate entries
        profile_model_map = defaultdict(dict)
        all_includes = []
        for entry in self.profiles:
            if entry.model in profile_model_map[entry.name]:
                error_list.append(f"Duplicate profile-model entry found for {entry.name}:{entry.model}")
            else:
                profile_model_map[entry.name][entry.model] = 1

            if entry.include:
                # Make empty entries None to make it easier to process
                if len(entry.include.strip()) == 0:
                    entry.include = None
                else:
                    all_includes.append(entry.include)

        # Check that all the include exist
        for include in all_includes:
            include_pair = include.split(':')
            if len(include_pair) == 2:
                if not include_pair[1] in profile_model_map[include_pair[0]]:
                    error_list.append(f"Entry for include '{include}' not found. {profile_model_map}")
            else:
                error_list.append(f"Improperly formatted include '{include}'. Expecting profile_name:model_name.")

        # Summarize errors and exit if not in test mode
        if len(error_list) > 0:
            err_msg = f"Found {len(error_list)} errors in workspace config from {file_path}. EXITING."
            logger.error(err_msg)
            for error in error_list:
                logger.error(f">>> {error}")

            if test:
                return False
            else:
                raise RuntimeError(err_msg)

        return True

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a WorkspaceConfig object
        :param data: The data to use to construct the object.
        :param file_path: Optional path to the file for logging.
        :return: The constructed object.
        """
        # Validate
        if not conf_utils.validate_schema(data, WorkspaceConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in ModelConfig from {file_path}. See log. EXITING.")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup
        ws_config = WorkspaceConfig.from_dict(data)
        ws_config._finish_init(file_path)
        return ws_config

    def update_properties(self, profile_name: str, model_name: str, profile_data: dict) -> None:
        """
        Update the provided dictionary to have all specific entries.
        :param profile_name: The profile name.
        :param model_name: The model name.
        :param profile_data: The profile data to be updated.
        :return: None.
        """

        # Walk all the entries finding the one to include. If that entry has an include, grab its
        # values first. Note, we do this for the entire list because we may match against multiple
        # model names, because they may have wildcards.
        for entry in self.profiles:
            if entry.name == profile_name:
                if re.match(entry.model, model_name):
                    # If it has an include, then include its values first
                    if entry.include:
                        temp_machine, temp_model = entry.include.split(':')
                        self.update_properties(temp_machine, temp_model, profile_data)

                    # Now add non None values
                    print(f"update_properties adding: {profile_name}:{model_name} = {entry.profile}")
                    for k, v in entry.profile.items():
                        if v is not None:
                            profile_data[k] = v

                    # Since we found a match we are done
                    return

    @staticmethod
    def load(data_path: str = "./config.json"):
        """
        Loads the config from the provided path, validate and construct the config.
        :param data_path: Path to config.
        :return: Loaded, validated and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading WORKSPACE CONFIG from {data_path}")
        if Path(data_path).exists():
            data = jbfs.load_file(data_path)
        else:
            data = {}

        # Validate and construct the model.
        return WorkspaceConfig.construct(data, data_path)

    def get_profile(self, profile_name: str = None, model_name: str = None):
        """
        Loads the config file from the provided path, validates, and constructs the LabProfile object.
        :param profile_name: The name of the profile.
        :param model_name: The name of the model.
        :return: Loaded, validated, and constructed LabProfile object.
        """
        # Build up a dict to make the aggregated lab profile object
        print(f"get_profile: {profile_name}:{model_name}")

        profile_data = {}
        # Load from default:default
        self.update_properties(profile_name="default", model_name="default", profile_data=profile_data)

        # Load from machine:default
        if profile_name:
            self.update_properties(profile_name=profile_name, model_name="default", profile_data=profile_data)

        # Load from default:model
        if model_name:
            self.update_properties(profile_name="default", model_name=model_name, profile_data=profile_data)

        # Load from machine:model
        if profile_name and model_name:
            self.update_properties(profile_name=profile_name, model_name=model_name, profile_data=profile_data)

        # Construct the specs object
        lab_profile = LabProfile.from_dict(profile_data)

        # Now, set defaults
        if lab_profile.num_workers is None:
            lab_profile.num_workers = 4
        if lab_profile.no_paging is None:
            lab_profile.no_paging = False

        return lab_profile
