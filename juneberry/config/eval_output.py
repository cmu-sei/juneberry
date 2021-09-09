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

import copy
from datetime import datetime
import logging
from prodict import Prodict
import sys
from typing import Any

import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs
import juneberry.utils as jb_utils

logger = logging.getLogger(__name__)


class Dataset(Prodict):
    classes: Any
    config: str
    histogram: Any


class Metrics(Prodict):
    accuracy: float
    balanced_accuracy: float
    bbox: Any
    bbox_per_class: Any


class Model(Prodict):
    hash: str
    name: str
    num_classes: int


class Options(Prodict):
    dataset: Dataset
    model: Model


class Results(Prodict):
    classifications: list
    labels: list
    metrics: Metrics
    predictions: list


class Times(Prodict):
    duration: float
    end_time: Any
    start_time: Any


class EvaluationOutput(Prodict):
    FORMAT_VERSION = '0.1.0'
    SCHEMA_NAME = 'evaluation_output_schema.json'

    format_version: str
    options: Options
    results: Results
    times: Times

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a config object from a supposedly VALID and LATEST FORMAT model.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed and validated object.
        """
        conf_utils.require_version(data, EvaluationOutput.FORMAT_VERSION, file_path, 'EvaluationOutput')
        if not conf_utils.validate_schema(data, EvaluationOutput.SCHEMA_NAME):
            logger.error(f"Validation errors in EvaluationOutput from {file_path}. See log. EXITING!")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup
        model_config = EvaluationOutput.from_dict(data)
        return model_config

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validate and construct the config.
        :param data_path: Path to config.
        :return: Loaded, validated and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading EVALUATION OUTPUT from {data_path}")
        data = jbfs.load_file(data_path)

        # Validate and construct the model.
        return EvaluationOutput.construct(data, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the ExperimentConfig to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, EvaluationOutput.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to CURRENT json"""
        return conf_utils.prodict_to_dict(self)


class EvaluationOutputBuilder:
    """A helper class used to aid in the construction of the evaluation output."""

    def __init__(self):
        self.output = EvaluationOutput()

        self.output.format_version = EvaluationOutput.FORMAT_VERSION

        self.output.options = Options()
        self.output.options.dataset = Dataset()
        self.output.options.model = Model()

        self.output.results = Results()
        self.output.results.metrics = Metrics()

        self.output.times = Times()

    def set_times(self, start_time: datetime, end_time: datetime) -> None:
        """
        Sets the appropriate start and stop times in the evaluation output.
        :param start_time: The time the evaluation started.
        :param end_time: The time the evaluation ended.
        :return: None
        """
        self.output.times.start_time = start_time.isoformat()
        self.output.times.end_time = end_time.isoformat()
        self.output.times.duration = (end_time - start_time).total_seconds()

    def save_predictions(self, data_path: str) -> None:
        """
        Save the predictions portion of the evaluation output to the specified path.
        :param data_path: The path to save to.
        :return: Nothing.
        """
        # Save the predictions content to file.
        self.output.save(data_path)

    def save_metrics(self, data_path: str) -> None:
        """
        Save the metrics portion of the evaluation output to the specified path.
        :param data_path:
        :return:
        """
        # Create a copy of the output so we don't lose the original data.
        metrics = copy.deepcopy(self.output)

        # Remove the attributes that don't belong in the metrics file.
        metrics.options.dataset.classes = None
        metrics.results.labels = None
        metrics.results.predictions = None
        metrics.results.classifications = None

        # Save the metrics content to file.
        metrics.save(data_path)


