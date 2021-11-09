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

from enum import Enum
import logging
from pathlib import Path
import sys

from prodict import Prodict, List

import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    PLOT_PR = 'plot_pr'
    PLOT_ROC = 'plot_roc'
    SUMMARY = 'summary'
    ALL_ROC = 'all_roc'
    ALL_PR = 'all_pr'


class ModelTest(Prodict):
    classify: int
    dataset_path: str
    filters: List[str]
    tag: str
    use_train_split: bool
    use_val_split: bool


class Model(Prodict):
    name: str
    filters: List[str]
    tests: List[ModelTest]
    version: str


class ReportTest(Prodict):
    tag: str
    classes: str


class Report(Prodict):
    # type is [all_pr | all_roc | plot_pr | plot_roc | summary]
    classes: str
    csv_name: str
    description: str
    iou: float
    output_dir: str
    output_name: str
    plot_title: str
    tests: List[ReportTest]
    type: str


class Filter(Prodict):
    tag: str
    cmd: List[str]
    inputs: List[str]
    outputs: List[str]


class ExperimentConfig(Prodict):
    """
    A class to validate and manage the experiment config.
    """
    FORMAT_VERSION = '0.2.0'
    SCHEMA_NAME = 'experiment_schema.json'

    description: str
    filters: List[Filter]
    format_version: str
    models: List[Model]
    reports: List[Report]
    timestamp: str

    def _finish_init(self) -> None:
        """
        Finish setting up fields and validate some fields
        :return: None
        """
        error_count = 0

        # Check format_version
        # jbvs.version_check("EXPERIMENT", config.get('formatVersion', None), FORMAT_VERSION, True)

        # Keep track of all the tags we find so we can validate the reports against the tags.
        tag_set = set()

        # Process the models
        for i, model in enumerate(self.models):
            # At this point it must exist.
            model_manager = jbfs.ModelManager(model.name)
            if not model_manager.get_model_dir().exists():
                logger.error(f"Model not found: {model_manager.get_model_dir()}")
                error_count += 1

            for test in model.tests:
                if test.tag in tag_set:
                    logger.error(f"Found duplicate tag: tag= '{test.tag}', models[{i}].")
                    error_count += 1
                else:
                    tag_set.add(test.tag)

                if not Path(test.dataset_path).exists():
                    logger.error(f"Dataset not found: {test.dataset_path}")
                    error_count += 1

        # If we don't have reports, then add an empty section so folks can just iterate
        if self.reports is None:
            self.reports = []

        # The reports should have a tests section to make sense and tags in those tests
        for i, report in enumerate(self.reports):
            if report.type == ReportType.PLOT_ROC or report.type == ReportType.PLOT_PR:

                for test in report.tests:
                    if test.tag not in tag_set:
                        logger.error(f"Unknown report tag. tag='{test['tag']}', report index = {i}.")
                        error_count += 1

        if error_count > 0:
            logger.error(f"Found {error_count} errors in experiment config. EXITING.")
            sys.exit(-1)

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a config object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: The constructed object.
        """
        # Validate with our schema
        conf_utils.require_version(data, ExperimentConfig.FORMAT_VERSION, file_path, 'ExperimentConfig')
        if not conf_utils.validate_schema(data, ExperimentConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in {file_path}. See log. EXITING.")
            sys.exit(-1)

        # Finally, construct the object
        config = ExperimentConfig.from_dict(data)
        config._finish_init()
        return config

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validates, and constructs the config.
        :param data_path: Path to config.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading EXPERIMENT CONFIG from {data_path}")
        data = jbfs.load_file(data_path)

        # Construct the config.
        return ExperimentConfig.construct(data, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the ExperimentConfig to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, ExperimentConfig.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to json"""
        return conf_utils.prodict_to_dict(self)


