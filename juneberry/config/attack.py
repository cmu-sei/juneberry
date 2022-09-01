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

from prodict import List, Prodict

from juneberry.config.plugin import Plugin
import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Models(Prodict):
    meta: str
    private: str
    shadow: str
    shadow_disjoint_quantity: int
    shadow_superset_quantity: int


class DataConfigs(Prodict):
    in_out_builder: Plugin
    query_data: str
    training_data: str


class Watermarks(Prodict):
    disjoint_args: List[Prodict]
    private_disjoint_args: Prodict
    private_superset_args: Prodict
    query_watermarks: Plugin
    superset_args: List[Prodict]
    training_watermarks: Plugin


class PropertyInferenceAttackConfig(Prodict):
    FORMAT_VERSION = '0.1.0'
    SCHEMA_NAME = 'property_inference_attack_schema.json'

    data_configs: DataConfigs
    models: Models
    watermarks: Watermarks

    def _finish_init(self):
        """
        TODO: Value cleanup steps go here.
        """
        pass

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Validate and construct an AttackConfig object.
        :param data: The data to use to construct the object.
        :param file_path: (Optional) path to a file that may have been loaded. Used for logging.
        :return: A constructed and validated object.
        """

        jb_conf_utils.require_version(data, PropertyInferenceAttackConfig.FORMAT_VERSION, file_path, 'AttackConfig')
        if not jb_conf_utils.validate_schema(data, PropertyInferenceAttackConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in AttackConfig from {file_path}. See log. EXITING.")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup.
        attack_config = PropertyInferenceAttackConfig.from_dict(data)
        attack_config._finish_init()
        return attack_config

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validates, and constructs the config.
        :param data_path: Path to config.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading ATTACK CONFIG from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the attack config.
        return PropertyInferenceAttackConfig.construct(data, data_path)
