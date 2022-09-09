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

from datetime import datetime, timedelta
import logging
import sys
from typing import Union

from prodict import Prodict

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Times(Prodict):
    duration: float
    end_time: str
    epoch_duration_sec: list
    start_time: str


class Results(Prodict):
    # Accuracies, losses and errors are all lists of float values, one per epoch
    accuracy: list
    batch_loss: list
    false_negative: list
    fg_cls_accuracy: list
    learning_rate: list
    loss: list
    loss_bbox: list
    loss_box_reg: list
    loss_cls: list
    loss_rpn_bbox: list
    loss_rpn_cls: list
    loss_rpn_loc: list
    model_hash: str
    num_bg_samples: list
    num_fg_samples: list
    num_neg_anchors: list
    num_pos_anchors: list
    onnx_model_hash: str
    test_error: list
    timetest: list
    train_error: list
    val_accuracy: list
    val_error: list
    val_loss: list
    val_loss_bbox: list
    val_loss_cls: list
    val_loss_rpn_bbox: list
    val_loss_rpn_cls: list


class Options(Prodict):
    batch_size: int
    data_type: str
    epochs: int
    label_mapping: Union[dict, str]
    model_architecture: Prodict
    model_name: str
    num_training_images: int
    num_validation_images: int
    seed: int
    training_dataset_config_path: str
    validation_dataset_config_path: str


class TrainingOutput(Prodict):
    FORMAT_VERSION = '0.2.0'
    SCHEMA_NAME = 'training_output_schema.json'

    format_version: str
    options: Options
    results: Results
    times: Times

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Validate and construct an object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed object.
        """
        # Validate with our schema
        jb_conf_utils.require_version(data, TrainingOutput.FORMAT_VERSION, file_path, 'TrainingOutput')
        if not jb_conf_utils.validate_schema(data, TrainingOutput.SCHEMA_NAME):
            logger.error(f"Validation errors in TrainingOutput from {file_path}. See log. EXITING!")
            sys.exit(-1)

        return TrainingOutput.from_dict(data)

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validates and constructs the object.
        :param data_path: Path to config.
        :return: Loaded, validated and constructed object.
        """
        logger.info(f"Loading TRAINING OUTPUT from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return TrainingOutput.construct(data)

    def save(self, data_path: str) -> None:
        """
        Save the TrainingOutput to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        jb_conf_utils.validate_and_save_json(self.to_json(), data_path, TrainingOutput.SCHEMA_NAME)

    def to_json(self):
        return jb_conf_utils.prodict_to_dict(self)


class TrainingOutputBuilder:
    """ A helper class used to aid in the construction of the training output. """

    def __init__(self):
        self.output = TrainingOutput()

        self.output.format_version = TrainingOutput.FORMAT_VERSION
        self.output.options = Options()
        self.output.times = Times()
        self.output.results = Results()

    def set_from_model_config(self, model_name: str, model_config: ModelConfig) -> None:
        """
        Extracts salient values from the model config and places them into the object.
        :param model_name: The name of the model in the models directory.
        :param model_config: The model config object.
        :return: None
        """

        self.output.options.batch_size = model_config.batch_size
        self.output.options.epochs = model_config.epochs

        # Recording the model_architecture is a little more involved. If the model_architecture contains
        # kwargs, there's a chance a "labels" kwarg could be introduced (particularly in TensorFlow trainers)
        # which may have integer keys. Integer keys would cause problems for Prodict and JSON, so it's best
        # to eliminate the "labels" kwarg if one is present. Historically, when the labels were included in
        # the TrainingOutput, they had a dedicated key and were not associated with the model architecture.
        kwargs = model_config.model_architecture.kwargs
        labels = kwargs.pop('labels') if kwargs is not None and "labels" in kwargs.keys() else None
        self.output.options.model_architecture = model_config.model_architecture

        # Restore the labels if they were removed. self.output.options.model_architecture and
        # model_config.model_architecture have different IDs, so restoring the labels in
        # model_config.model_architecture will not affect self.output.options.model_architecture.
        if labels:
            kwargs['labels'] = labels

        self.output.options.model_name = model_name
        self.output.options.seed = model_config.seed
        self.output.options.training_dataset_config_path = model_config.training_dataset_config_path
        # self.output.options.label_mapping = model_config.label_mapping

        if model_config.validation.algorithm == "from_file":
            self.output.options.validation_dataset_config_path = model_config.validation.arguments['file_path']

        # TODO: Add learning rate?

    def set_from_dataset_config(self, dataset_config: DatasetConfig) -> None:
        """
        Extracts salient values from the dataset config object and places them into the
        training output.
        :param dataset_config: A dataset config.
        :return: None
        """
        self.output.options.training_dataset_config_path = dataset_config.file_path
        self.output.options.data_type = dataset_config.data_type

    def set_times(self, start_time: datetime, end_time: datetime) -> None:
        """
        Sets the appropriate start and stop times in the training output.
        :param start_time: The time training started.
        :param end_time: The time training ended.
        :return: None
        """
        self.output.times.start_time = start_time.isoformat()
        self.output.times.end_time = end_time.isoformat()
        self.output.times.duration = (end_time - start_time).total_seconds()

    def record_epoch_duration(self, duration: timedelta) -> None:
        """
        Appends a duration to the list in the Training Output that tracks epoch durations.
        :param duration: The duration (as a datetime timedelta) in seconds to store in the
        times.epoch_duration_sec list in the Training Output.
        :return: Nothing.
        """
        self.output.times.epoch_duration_sec.append(duration)

    def to_dict(self) -> dict:
        """ :return: Returns a pure version of the data structure as a dict. """
        return jb_conf_utils.prodict_to_dict(self.output)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to CURRENT json"""
        as_dict = jb_conf_utils.prodict_to_dict(self.output)

        return as_dict

    def save(self, data_path: str) -> None:
        """
        Saves the training output to the specified path.
        :param data_path: The path to save to.
        :return: None
        """
        self.output.save(data_path)
