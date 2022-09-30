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

import logging
import sys

from prodict import Prodict

import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Hashes(Prodict):
    FORMAT_VERSION = '0.3.0'
    SCHEMA_NAME = 'hashes_schema.json'

    model_architecture: str

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Validate and construct a Hashes object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed object.
        """

        # Validate with our schema
        if not jb_conf_utils.validate_schema(data, Hashes.SCHEMA_NAME):
            logger.error(f"Validation errors in Hashes object from {file_path}. See log. Exiting!")
            sys.exit(-1)

        # Finally, construct the object
        return Hashes.from_dict(data)

    @staticmethod
    def load(data_path: str):
        """
        Load the config from the provided path, validate, and construct the config.
        :param data_path: Path to config.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading HASHES CONFIG from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return Hashes.construct(data, data_path)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to json."""
        return jb_conf_utils.prodict_to_dict(self)

    def save(self, data_path: str) -> None:
        """
        Save the HashesConfig to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        jb_conf_utils.validate_and_save_json(self.to_json(), data_path, Hashes.SCHEMA_NAME)
