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

import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class MachineConfig(Prodict):
    """
    A class to validate and manage the machine config file.
    """
    # TODO change the purpose of the prodict to store the machine:model specs, i.e. num_gpus, num_workers
    default: dict

    def _finish_init(self) -> None:
        """
        Validate machine config fields
        :return: None
        """
        error_count = 0

        # Verify that defaults are specified
        if self.default is None:
            error_count += 1
        if self.default.get('default') is None:
            error_count += 1

        # If errors found, report and exit
        if error_count > 0:
            logger.error(f"Found {error_count} errors in machine config. EXITING.")
            sys.exit(-1)

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a machine config object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: The constructed object.
        """

        # Construct the MachineConfig object
        config = MachineConfig.from_dict(data)
        config._finish_init()
        return config

    @staticmethod
    def load(data_path: str):
        """
        Loads the machine config file from the provided path, validates, and constructs the MachineConfig object.
        :param data_path: Path to the machine config annotations file.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading MACHINE CONFIG from {data_path}")
        data = jbfs.load_file(data_path)

        # Construct the config.
        return MachineConfig.construct(data, data_path)

