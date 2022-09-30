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
from pathlib import Path
import sys

from prodict import List, Prodict

from juneberry.config.plugin import Plugin
import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Hyperparameter(Prodict):
    hyperparameter_name: str
    fqcn: str
    kwargs: Prodict


class TrialResources(Prodict):
    cpu: int
    gpu: int


class TuningParameters(Prodict):
    checkpoint_interval: int
    metric: str
    mode: str
    scope: str

    def init(self):
        self.checkpoint_interval = 0
        self.metric = 'loss'
        self.mode = 'min'
        self.scope = 'last'


class TuningConfig(Prodict):
    FORMAT_VERSION = '0.1.0'
    SCHEMA_NAME = 'tuning_schema.json'

    description: str
    format_version: str
    num_samples: int
    scheduler: Plugin
    search_algorithm: Plugin
    search_space: List[Hyperparameter]
    timestamp: str
    trial_resources: TrialResources
    tuning_parameters: TuningParameters

    def init(self) -> None:
        pass

    def _finish_init(self, file_path: str = None) -> None:
        """
        Initializes a tuning configuration object from a config data structure.
        :param file_path: Optional - string indicating the tuning config file.
        :return: Nothing.
        """
        # Set the file_path
        self.file_path = Path(file_path) if file_path is not None else None

        # Initialize trial resources if none were defined.
        if self.trial_resources is None:
            self.trial_resources = TrialResources()

        # Initialize GPU trial resources if none were defined.
        if self.trial_resources.gpu is None:
            self.trial_resources.gpu = 0

        # Initialize CPU trial resources if none were defined.
        if self.trial_resources.cpu is None:
            self.trial_resources.cpu = 1 if self.trial_resources.gpu == 0 else 0

        # Initial tuning parameters if none were defined.
        if self.tuning_parameters is None:
            self.tuning_parameters = TuningParameters()

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a Tuning config object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed and validated object.
        """
        # Validate
        if not jb_conf_utils.validate_schema(data, TuningConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in TuningConfig from {file_path}. See log. Exiting.")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup
        tuning_config = TuningConfig.from_dict(data)
        tuning_config._finish_init(file_path)
        return tuning_config

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validates, and constructs the config.
        :param data_path: Path to config.
        :return: Loaded, validated and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading TUNING CONFIG from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return TuningConfig.construct(data, data_path)
