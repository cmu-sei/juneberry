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
from prodict import Prodict
import sys
import re
import os

import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class MachineSpecs(Prodict):
    """
    A class to validate and manage the machine specifications pulled from the machine config file.
    """
    properties = ["num_workers", "num_gpus"]
    num_workers: int
    num_gpus: int

    def _finish_init(self) -> None:
        """
        Validate machine specs fields
        :return: None
        """
        error_count = 0

        # Check that the number of gpus doesn't exceed the number of cuda visible devices
        visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_gpus:
            visible_gpus = visible_gpus.count(",") + 1
            if self.num_gpus:
                if self.num_gpus > visible_gpus:
                    error_count += 1

        # If errors found, report and exit
        if error_count > 0:
            logger.error(f"Found {error_count} errors in machine specs. EXITING.")
            sys.exit(-1)

    @staticmethod
    def construct(data: dict):
        """
        Load, validate, and construct a machine specs object.
        :param data: The data to use to construct the object.
        :return: The constructed object.
        """

        # Construct the MachineSpecs object
        specs = MachineSpecs.from_dict(data)
        specs._finish_init()
        return specs

    @staticmethod
    def update_properties(machine: str, model: str, config_data: dict, specs_data: dict):
        """
        Update the specs dictionary to include property overrides.
        :param machine: The machine class.
        :param model: The model class.
        :param config_data: The machine config.
        :param specs_data: The dictionary of specs for execution.
        :return: The machine specification object.
        """

        if machine in config_data:
            for key in config_data[machine].keys():
                if re.match(key, model):
                    if "include" in config_data[machine][key]:
                        include_lst = config_data[machine][key]["include"].split(':')
                        machine = include_lst[0]
                        model = include_lst[1]
                        specs_data = MachineSpecs.update_properties(machine, model, config_data, specs_data)

                    else:
                        for prop in config_data[machine][key].keys():
                            if config_data[machine][key][prop]:
                                specs_data[prop] = config_data[machine][key][prop]

        return specs_data

    @staticmethod
    def validate_workspace_config(config_data: dict):
        # Check for default:default option
        default_error_count = 0

        if "default" not in config_data:
            default_error_count += 1
        else:
            if "default" not in config_data["default"]:
                default_error_count += 1

        # Report missing default:default section
        if default_error_count > 0:
            logger.error(f"Config missing default:default section.")

        # Check for valid include
        include_error_count = 0

        for machine_key in config_data.keys():
            for model_key in config_data[machine_key]:
                if "include" in config_data[machine_key][model_key]:
                    include_lst = config_data[machine_key][model_key]["include"].split(":")
                    # Check for invalid machine:model pair
                    if include_lst[0] not in config_data:
                        include_error_count += 1
                    else:
                        if include_lst[1] not in config_data[include_lst[0]]:
                            include_error_count += 1
                        else:
                            # Check for chained includes
                            if "include" in config_data[include_lst[0]][include_lst[1]]:
                                include_error_count += 1

        # Report invalid include(s)
        if include_error_count > 0:
            logger.error(f"Config contains invalid include(s).")

        error_count = default_error_count + include_error_count
        if error_count > 0:
            logger.error(f"Found {error_count} errors in machine specs. EXITING.")
            sys.exit(-1)

    @staticmethod
    def load(data_path: str, machine_class: str = None, model_name: str = None):
        """
        Loads the machine config file from the provided path, validates, and constructs the MachineSpecs object.
        :param data_path: Path to the machine config annotations file.
        :param machine_class: The name of the class of machine.
        :param model_name: The name of the model.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading MACHINE CONFIG from {data_path}")
        config_data = jbfs.load_file(data_path)
        specs_data = {}

        # Validate config
        MachineSpecs.validate_workspace_config(config_data)

        # Load from default:default
        specs_data = MachineSpecs.update_properties(machine="default", model="default",
                                                    config_data=config_data, specs_data=specs_data)

        # Load from machine:default
        if machine_class:
            specs_data = MachineSpecs.update_properties(machine=machine_class, model="default",
                                                        config_data=config_data, specs_data=specs_data)

        # Load from default:model
        if model_name:
            specs_data = MachineSpecs.update_properties(machine="default", model=model_name,
                                                        config_data=config_data, specs_data=specs_data)

        # Load from machine:model
        if machine_class and model_name:
            specs_data = MachineSpecs.update_properties(machine=machine_class, model=model_name,
                                                        config_data=config_data, specs_data=specs_data)

        # Construct the specs object
        return MachineSpecs.construct(specs_data)
