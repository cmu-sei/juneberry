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
import logging
from pathlib import Path
import random
import sys
import typing

from jsonpath_ng.ext import parse
from prodict import List, Prodict

from juneberry.config.plugin import Plugin
import juneberry.config.util as jb_conf_utils
from juneberry.config.workspace import LabProfile
import juneberry.filesystem as jb_fs

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
    TENSORFLOW = "tensorflow"
    TORCHVISION = 'torchvision'


class ModelArchitecture(Prodict):
    fqcn: str
    kwargs: Prodict
    module: str
    args: Prodict
    previous_model: typing.Union[str, None]

    def get_shape_hwc(self):
        """ :return The height, width and channels in a named tuple. """
        return ShapeHWC(self.kwargs['img_height'], self.kwargs['img_width'], self.kwargs['channels'])


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
    loss_args: Prodict
    loss_fn: str
    lr_schedule_args: Prodict
    lr_schedule_fn: str
    lr_step_frequency: str
    optimizer_args: Prodict
    optimizer_fn: str
    strict: bool

    def init(self):
        self.strict = True


class Detectron2(Prodict):
    enable_val_loss: bool
    metric_interval: int
    overrides: Prodict
    supplements: List[str]

    def init(self):
        self.enable_val_loss = False
        self.metric_interval = 1


class MMDetection(Prodict):
    load_from: str
    overrides: Prodict
    test_pipeline_stages: List[Prodict]
    train_dataset_wrapper: Plugin
    train_pipeline_stages: List[Prodict]
    val_pipeline_stages: List[Prodict]


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
    FORMAT_VERSION = '0.3.0'
    SCHEMA_NAME = 'model_schema.json'
    batch_size: int
    description: str
    detectron2: Detectron2
    epochs: int
    evaluation_metrics: List[Plugin]
    evaluation_metrics_formatter: Plugin
    evaluation_transforms: List[Plugin]
    evaluation_target_transforms: List[Plugin]
    evaluator: Plugin
    file_path: Path
    format_version: str
    lab_profile: LabProfile
    label_mapping: typing.Union[Prodict, str]
    mmdetection: MMDetection
    model_architecture: ModelArchitecture
    model_transforms: List[Plugin]
    preprocessors: List[Plugin]
    pytorch: PytorchOptions
    reports: List[Plugin]
    seed: typing.Union[int, None]
    stopping_criteria: StoppingCriteria
    summary_info: Prodict
    tensorflow: TensorFlow
    timestamp: str
    trainer: Plugin
    training_dataset_config_path: str
    training_metrics: List[Plugin]
    training_transforms: List[Plugin]
    training_target_transforms: List[Plugin]
    validation: Validation

    def init(self) -> None:
        """
        This is NOT init. This is a similar method called by Prodict to set defaults
        on values BEFORE to_dict is called.
        """

    def _finish_init(self, file_path: str = None):
        """
        Initializes a ModelConfig object.
        """

        # Check the format_version attribute.
        # We currently don't do a version check because we don't have any breaking version changes.
        # if self.format_version is not None:
        #     jbvs.version_check("MODEL", self.format_version, ModelConfig.FORMAT_VERSION, True)

        self.file_path = Path(file_path) if file_path is not None else None
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
                file_content = jb_fs.load_json(self.label_mapping)
                if 'labelNames' in file_content:
                    self.label_dict = file_content['labelNames']
                else:
                    logger.error(f"Could not retrieve a label_mapping from {self.label_mapping}. Is it a JSON file "
                                 f"containing the key 'labelNames'? EXITING.")
                    sys.exit(-1)

            # label_mapping is already a dictionary, so just set label_dict to that
            else:
                self.label_dict = self.label_mapping

        # If present, the training_dataset_config_path should be converted from string to Path.
        if self.training_dataset_config_path is not None:
            self.training_dataset_config_path = Path(self.training_dataset_config_path)

        # For model_architecture if they specified model/args, convert to fqcn/kwargs and provide a warning.
        model_arch = self.model_architecture
        if model_arch.module is not None:
            if model_arch.fqcn is not None:
                logger.warning("The model_architecture contains both 'module' (deprecated) AND 'fqcn'. Using 'fqcn.'")
            else:
                logger.warning("Found 'module' (deprecated) in the model_architecture. Converting it to 'fqcn'.")
                model_arch.fqcn = model_arch.module

        if model_arch.args is not None:
            if model_arch.kwargs is not None:
                logger.warning("The model_architecture contains 'args' (deprecated) AND 'kwargs'. Using 'kwargs.'")
            else:
                logger.warning("Found 'args' (deprecated) in the model_architecture. Converting it to 'kwargs'.")
                model_arch.kwargs = model_arch.args


    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a config object from a supposedly VALID and LATEST FORMAT model.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed and validated object.
        """
        # We currently don't do a version check because we don't have any breaking version changes.
        # jb_conf_utils.require_version(data, ModelConfig.FORMAT_VERSION, file_path, 'ModelConfig')

        # Validate
        if not jb_conf_utils.validate_schema(data, ModelConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in ModelConfig from {file_path}. See log. EXITING.")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup
        model_config = ModelConfig.from_dict(data)
        model_config._finish_init(file_path)
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
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return ModelConfig.construct(data, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the ExperimentConfig to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        jb_conf_utils.validate_and_save_json(self.to_json(), data_path, ModelConfig.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to CURRENT json"""
        as_dict = jb_conf_utils.prodict_to_dict(self)

        ignore_attrs = ["file_path", "label_dict"]
        for attr_name in ignore_attrs:
            if attr_name in as_dict:
                del as_dict[attr_name]

        return as_dict

    def adjust_attributes(self, adjustment_dict: dict) -> Prodict:
        """
        This method is responsible for replacing attributes in a ModelConfig.
        :param adjustment_dict: A dictionary where the keys indicate which ModelConfig attributes
        to change, and the corresponding value indicates what the attribute value should be changed to.
        :return: A ModelConfig whose attributes have been adjusted to the desired values.
        """
        # Loop through all keys in the adjustment dictionary and make the substitutions in the
        # current model config.
        for k in adjustment_dict.keys():
            jsonpath_expr = parse(k)
            jsonpath_expr.update(self, adjustment_dict[k])

        # Return the adjusted ModelConfig.
        return self

    def get_previous_model(self):
        """ :return" Previous model architecture from which to load weights. """
        return self.model_architecture.previous_model

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
