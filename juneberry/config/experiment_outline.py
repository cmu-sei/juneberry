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
import typing

from prodict import Prodict, List

import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Model(Prodict):
    filters: List
    maximum_evaluations: int


class Report(Prodict):
    classes: str
    description: str
    fqcn: str
    kwargs: Prodict
    test_tag: str


class Test(Prodict):
    tag: str
    dataset_path: str
    classify: int


class Variable(Prodict):
    nickname: str
    config_field: str
    vals: typing.Union[typing.List, str]


class ExperimentOutline(Prodict):
    FORMAT_VERSION = '0.2.0'
    SCHEMA_NAME = 'experiment_outline_schema.json'

    baseline_config: str
    description: str
    filters: List
    format_version: str
    model: Model
    reports: List[Report]
    tests: List[Test]
    timestamp: str
    variables: List[Variable]

    def _finish_init(self, experiment_name):
        self.experiment_name = experiment_name

        # Check formatVersion
        # jbvs.version_check("EXPERIMENT OUTLINE", self.format_version, FORMAT_VERSION, True)

        # Verify variables aren't constants
        for variable in self.variables:
            if isinstance(variable.values, list):
                if len(variable.values) < 2:
                    self.valid = False
                    logger.error(f"Insufficient possibilities for '{variable}' in '{self.experiment_name}'. "
                                 f"this variable from the experiment outline or add more possibilities.")
                    sys.exit(-1)

    def analyze_experiment_variables(self) -> None:
        """
        This method identifies the experiment variables and calculates the number of possible combinations.
        :return: Nothing.
        """
        logger.info(f"Identified {len(self.variables)} variables:")

        combinations = 1

        for variable in self.variables:
            if type(variable['vals']) is str:
                count = 1
                logger.info(f"  {count:3d} random value  for {variable['config_field']}")
            else:
                count = len(variable['vals'])
                logger.info(f"  {count:3d} possibilities for {variable['config_field']}")

            combinations *= count

        logger.info(f"{combinations:5d} unique configurations in the outline file for '{self.experiment_name}'.")

    def check_experiment_variables(self) -> None:
        """
        This method verifies that each variable in the experiment has more than one possibility. If a variable
        has only one possibility, then it should not be a variable.
        :return: Nothing.
        """

        for variable in self.variables:

            if type(variable.values) is list:
                count = len(variable.vals)

                if count < 2:
                    logger.error(f"Insufficient possibilities for '{variable}' in '{self.experiment_name}'. "
                                 f"this variable from the experiment outline or add more possibilities.")
                    sys.exit(-1)

    @staticmethod
    def construct(data: dict, experiment_name: str, file_path: str = None):
        """
        Validate and construct a ModelConfig object.
        :param data: The data to use to construct the object.
        :param experiment_name: The name of the experiment to use for logging.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed object.
        """
        # Validate with our schema
        jb_conf_utils.require_version(data, ExperimentOutline.FORMAT_VERSION, file_path, 'ExperimentOutline')
        if not jb_conf_utils.validate_schema(data, ExperimentOutline.SCHEMA_NAME):
            logger.error(f"Validation errors in {file_path}. See log. Exiting.")
            sys.exit(-1)

        # Finally, construct the object
        config = ExperimentOutline.from_dict(data)
        config._finish_init(experiment_name)
        return config

    @staticmethod
    def load(data_path: str, experiment_name: str):
        """
        Loads the config from the provided path, validates, and constructs the config.
        :param data_path: Path to config.
        :param experiment_name: The name of the experiment to use for logging.
        :return: Loaded, validated and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading EXPERIMENT OUTLINE from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return ExperimentOutline.construct(data, experiment_name, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the ExperimentOutline to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        jb_conf_utils.validate_and_save_json(self.to_json(), data_path, ExperimentOutline.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to json"""
        return jb_conf_utils.prodict_to_dict(self)

