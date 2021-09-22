#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn 
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow 
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

from datetime import datetime
import logging
from prodict import Prodict
import sys

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs
import juneberry.utils as jb_utils

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
    model_name: str
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
    colorspace: str
    dimensions: str
    epochs: int
    model_architecture: Prodict
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
        conf_utils.require_version(data, TrainingOutput.FORMAT_VERSION, file_path, 'TrainingOutput')
        if not conf_utils.validate_schema(data, TrainingOutput.SCHEMA_NAME):
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
        data = jbfs.load_file(data_path)

        # Validate and construct the model.
        return TrainingOutput.construct(data)

    def save(self, data_path: str) -> None:
        """
        Save the TrainingOutput to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, TrainingOutput.SCHEMA_NAME)

    def to_json(self):
        return conf_utils.prodict_to_dict(self)


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
        self.output.results.model_name = model_name

        self.output.options.batch_size = model_config.batch_size
        self.output.options.epochs = model_config.epochs
        self.output.options.model_architecture = model_config.model_architecture
        self.output.options.seed = model_config.seed
        self.output.options.training_dataset_config_path = model_config.training_dataset_config_path

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

    def to_dict(self) -> dict:
        """ :return: Returns a pure version of the data structure as a dict. """
        return conf_utils.prodict_to_dict(self.output)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to CURRENT json"""
        as_dict = conf_utils.prodict_to_dict(self.output)

        return as_dict

    def save(self, data_path: str) -> None:
        """
        Saves the training output to the specified path.
        :param data_path: The path to save to.
        :return: None
        """
        self.output.save(data_path)

