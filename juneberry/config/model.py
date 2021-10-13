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

from collections import namedtuple
from enum import Enum
import json
import logging
from pathlib import Path
from prodict import List, Prodict
import random
import sys
import typing

import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs
import juneberry.version_system as jbvs

logger = logging.getLogger(__name__)

SplittingConfig = namedtuple('SplittingType', 'algo args randomizer')

ShapeHWC = namedtuple('ShapeHWC', 'height width channels')


# =======

class LRStepFrequency(str, Enum):
    BATCH = "batch"
    EPOCH = "epoch"


class SplittingAlgo(str, Enum):
    FROM_FILE = 'from_file'
    NONE = "none"
    RANDOM_FRACTION = 'random_fraction'
    TORCHVISION = 'torchvision'


# TODO Switch to plugin
class TransformEntry(Prodict):
    fqcn: str
    kwargs: Prodict


class Plugin(Prodict):
    fqcn: str
    kwargs: Prodict


class ModelArchitecture(Prodict):
    module: str
    args: Prodict
    previous_model: typing.Union[str, None]
    previous_model_version: typing.Union[str, None]

    def get_shape_hwc(self):
        """ :return The height, width and channels in a named tuple. """
        return ShapeHWC(self.args['img_height'], self.args['img_width'], self.args['channels'])


class StoppingCriteria(Prodict):
    direction: str
    history_key: str
    plateau_count: int
    abs_tol: float
    threshold: float

    def init(self):
        self.direction = 'le'
        self.history_key = 'val_loss'
        self.abs_tol = 0.0001


class ValidationArgs(Prodict):
    seed: int
    fraction: float
    file_path: str


class Validation(Prodict):
    algorithm: str
    arguments: ValidationArgs


class PytorchOptions(Prodict):
    accuracy_args: Prodict
    accuracy_fn: str
    loss_args: Prodict
    loss_fn: str
    lr_schedule_args: Prodict
    lr_schedule_fn: str
    lr_step_frequency: str
    optimizer_args: Prodict
    optimizer_fn: str


class Detectron2(Prodict):
    metric_interval: int
    overrides: Prodict

    def init(self):
        self.metric_interval = 1


class TensorFlow(Prodict):
    callbacks: List[Plugin]
    loss_args: Prodict
    loss_fn: str
    lr_schedule_args: Prodict
    lr_schedule_fn: str
    metrics: list
    optimizer_args: Prodict
    optimizer_fn: str


class ModelConfig(Prodict):
    FORMAT_VERSION = '0.2.0'
    SCHEMA_NAME = 'model_schema.json'

    batch_size: int
    description: str
    detectron2: Detectron2
    epochs: int
    evaluation_output: str
    evaluation_procedure: str
    evaluation_transforms: List[TransformEntry]
    evaluation_target_transforms: List[TransformEntry]
    format_version: str
    # TODO: Expand hints
    hints: Prodict
    label_mapping: typing.Union[Prodict, str]
    # TODO: Define mmdetection
    mmdetection: Prodict
    model_architecture: ModelArchitecture
    model_transforms: List[TransformEntry]
    platform: str
    preprocessors: List[TransformEntry]
    pytorch: PytorchOptions
    seed: typing.Union[int, None]
    stopping_criteria: StoppingCriteria
    summary_info: Prodict
    task: str
    tensorflow: TensorFlow
    timestamp: str
    training_dataset_config_path: str
    training_transforms: List[TransformEntry]
    training_target_transforms: List[TransformEntry]
    validation: Validation

    def init(self) -> None:
        """
        This is NOT init. This is a similar method called by Prodict to set defaults
        on values BEFORE to_dict is called.
        """
        self.task = "classification"

    def _finish_init(self):
        """
        Initializes a ModelConfig object.
        """

        # Check the format_version attribute.
        if self.format_version is not None:
            jbvs.version_check("MODEL", self.format_version, ModelConfig.FORMAT_VERSION, True)

        # There are a handful of keys that should be set to {} if they are still None.
        empty_keys = ["pytorch"]
        for key in empty_keys:
            if self.get(key, None) is None:
                self[key] = {}

        # There are a handful of keys that should be set to [] if they are still None.
        empty_keys = ["training_transforms", "evaluation_transforms", "preprocessors"]
        for key in empty_keys:
            if self.get(key, None) is None:
                self[key] = []

        # The mapping is used to describe all the integer class labels the model
        # is aware of in terms of human-readable strings. The label_mapping can be a
        # dictionary or string, but the label_dict will only be the dictionary version
        # of the mapping.
        # A valid label_mapping attribute will either be a string or a dictionary. The dictionary
        # style is what we want to use, so set the label_dict attribute to that dictionary.
        self.label_dict = None
        if self.label_mapping is not None:

            # If label_mapping is a string, we want to read the file at that path and get the
            # dictionary from inside the indicated file.
            if type(self.label_mapping) is str:
                self.label_mapping = Path(self.label_mapping)
                with open(self.label_mapping) as mapping_file:
                    file_content = json.load(mapping_file)
                # TODO: Convert these files too and look for both/either
                self.label_dict = file_content['labelNames']

            # label_mapping is already a dictionary, so just set label_dict to that
            else:
                self.label_dict = self.label_mapping

        # If present, the training_dataset_config_path should be converted from string to Path.
        if self.training_dataset_config_path is not None:
            self.training_dataset_config_path = Path(self.training_dataset_config_path)

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a config object from a supposedly VALID and LATEST FORMAT model.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed and validated object.
        """
        conf_utils.require_version(data, ModelConfig.FORMAT_VERSION, file_path, 'ModelConfig')
        if not conf_utils.validate_schema(data, ModelConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in ModelConfig from {file_path}. See log. EXITING!")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup
        model_config = ModelConfig.from_dict(data)
        model_config._finish_init()
        return model_config

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validate and construct the config.
        :param data_path: Path to config.
        :return: Loaded, validated and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading MODEL CONFIG from {data_path}")
        data = jbfs.load_file(data_path)

        # Validate and construct the model.
        return ModelConfig.construct(data, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the ExperimentConfig to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, ModelConfig.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to CURRENT json"""
        as_dict = conf_utils.prodict_to_dict(self)

        if 'label_dict' in as_dict:
            del as_dict['label_dict']

        return as_dict

    def get_previous_model(self):
        # TODO: Remove
        """ :return" Previous model architecture and version from which to load weights. """
        return self.model_architecture.previous_model, self.model_architecture.previous_model_version

    def get_validation_split_config(self):
        """
        :return: Algorithm name, algorithm arguments, and a randomizer for validation splitting.
        """
        # TODO: Move this into a constructor for the SplittingConfig?
        if self.validation is None:
            return SplittingConfig(None, None, None)

        splitting_algo = self.validation.algorithm
        splitting_args = self.validation.arguments

        # Set seed if there is one
        randomizer = None
        if splitting_algo == "random_fraction":
            if 'seed' in splitting_args:
                logger.info("Setting VALIDATION seed to: " + str(splitting_args['seed']))
                randomizer = random.Random()
                randomizer.seed(splitting_args['seed'])

        return SplittingConfig(splitting_algo, splitting_args, randomizer)

    def validate_for_training(self):
        """
        If some minimum group of attributes is present in the ModelConfig, then the model can be
        used for training tasks.
        """
        return not (self.training_dataset_config_path is None or self.epochs is None)

    def validate_for_evaluation(self):
        """
        If some minimum group of attributes is present in the ModelConfig, then the model can be
        used for evaluation tasks.
        """
        return not (self.evaluation_procedure is None or self.evaluation_output is None)