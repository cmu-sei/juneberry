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
from prodict import List, Prodict
import sys
import re

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

        # Check that "include" contains a valid machine:model pair?

        # If errors found, report and exit
        if error_count > 0:
            logger.error(f"Found {error_count} errors in machine specs. EXITING.")
            sys.exit(-1)

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a machine specs object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
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
                        lst = config_data[machine][key]["include"].split(':')
                        machine = lst[0]
                        model = lst[1]
                        specs_data = MachineSpecs.update_properties(machine, model, config_data, specs_data)

                    else:
                        for prop in MachineSpecs.properties:
                            if prop in config_data[machine][key]:
                                specs_data[prop] = config_data[machine][key][prop]

        return specs_data

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

        # Check default:default
        specs_data = MachineSpecs.update_properties(machine="default", model="default",
                                                    config_data=config_data, specs_data=specs_data)

        # Check machine:default
        if machine_class:
            specs_data = MachineSpecs.update_properties(machine=machine_class, model="default",
                                                        config_data=config_data, specs_data=specs_data)

        # Check default:model
        if model_name:
            specs_data = MachineSpecs.update_properties(machine="default", model=model_name,
                                                        config_data=config_data, specs_data=specs_data)

        # Check machine:model
        if machine_class and model_name:
            specs_data = MachineSpecs.update_properties(machine=machine_class, model=model_name,
                                                        config_data=config_data, specs_data=specs_data)

        # Construct the specs object
        return MachineSpecs.construct(specs_data, data_path)
