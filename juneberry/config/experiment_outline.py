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

import logging
from prodict import Prodict, List
import sys
import typing

import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class Model(Prodict):
    filters: List


class Report(Prodict):
    # type is [all_pr | all_roc | plot_pr | plot_roc | summary]
    type: str
    description: str
    test_tag: str
    classes: str
    output_name: str


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
        conf_utils.require_version(data, ExperimentOutline.FORMAT_VERSION, file_path, 'ExperimentOutline')
        if not conf_utils.validate_schema(data, ExperimentOutline.SCHEMA_NAME):
            logger.error(f"Validation errors in {file_path}. See log. EXITING")
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
        data = jbfs.load_file(data_path)

        # Validate and construct the model.
        return ExperimentOutline.construct(data, experiment_name, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the ExperimentOutline to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, ExperimentOutline.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to json"""
        return conf_utils.prodict_to_dict(self)

