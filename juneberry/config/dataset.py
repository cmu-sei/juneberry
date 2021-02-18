#! /usr/bin/env python3

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import sys
import random
import logging

from enum import Enum
from pathlib import Path

import juneberry
import juneberry.filesystem as jbfs

FORMAT_VERSION = '3.2.0'


class DataType(Enum):
    UNDEFINED = 0
    IMAGE = 1
    TABULAR = 2


class TaskType(Enum):
    UNDEFINED = 0
    CLASSIFICATION = 1
    OBJECTDETECTION = 2


class DatasetConfig:
    def __init__(self, config, relative_path: Path = None):
        """
        Initializes a data set configuration object from a config data structure.
        :param config: The config data structure.
        :param relative_path: Path for "relative" tabular operations.
        """

        self.config = config.copy()
        self.valid = True
        self.found_errors = 0
        self.task_type = None
        self.relative_path = relative_path

        # Extract required properties from the config
        self.num_model_classes = config.get('numModelClasses', 0)
        self.data_type = DataType[config.get('dataType', "UNDEFINED").upper()]
        self.format_version = config.get('formatVersion', '')

        # Attempt to extract some optional fields
        self.description = config.get('description', None)
        self.timestamp = config.get('timestamp', None)
        self.sampling = config.get('sampling', None)
        self.is_binary = self.num_model_classes == 2

        # Retrieve the label names
        self.label_names = self.retrieve_label_names()

        # Check the config for errors
        self.validate_config()

        if self.found_errors > 0:
            self.valid = False
            logging.error(f"Found {self.found_errors} error(s) in config file. See log for details.")
            sys.exit(-1)

    def validate_config(self):
        # Record an error if a REQUIRED item is missing
        for param in ['numModelClasses', 'dataType']:
            if param not in self.config:
                logging.error(f"Failed to find {param} in DATA SET config")
                self.found_errors += 1

        # Check format version
        jbfs.version_check("DATA SET", self.format_version, FORMAT_VERSION, True)

        # Some checks that are specific to the Image data type
        if self.is_image_type():
            try:
                self.task_type = TaskType[self.config['imageData']['taskType'].upper()]
            except KeyError:
                logging.error(f"Failed to find taskType in imageData section of DATA SET config")
                self.found_errors += 1

            required_keys = []
            if self.task_type == TaskType["CLASSIFICATION"]:
                required_keys = ['directory', 'label']
            elif self.task_type == TaskType["OBJECTDETECTION"]:
                required_keys = ['directory']

            for idx, entry in enumerate(self.config['imageData']['sources']):
                for key in required_keys:
                    if key not in entry:
                        logging.error(f"imageData source entry {idx} missing required keyword '{key}'.")
                        self.found_errors += 1

        # Some checks that are specific to the Tabular data type
        elif self.is_tabular_type():
            for idx, entry in enumerate(self.config['tabularData']['sources']):
                for key in ['path']:
                    if key not in entry:
                        logging.error(f"tabularData source entry {idx} missing required keyword '{key}'.")
                        self.found_errors += 1
                if 'root' in entry:
                    root = entry['root']
                    if root != 'dataroot' and root != 'workspace' and root != 'relative':
                        logging.error(f"tabularData sources entry {idx} property 'root' has invalid value '{root}'. "
                                      f"Must be either 'dataroot', 'workspace', or 'relative'")
                        self.found_errors += 1
            if 'labelIndex' not in self.config['tabularData']:
                logging.error(f"tabularData missing labelIndex!!")
                self.found_errors += 1

    def retrieve_label_names(self):
        label_names = {}
        for k, v in self.config['labelNames'].items():
            label_names[int(k)] = v
            if int(k) >= self.num_model_classes:
                logging.error(f"Label name entry {k} exceeds num_model_classes {self.num_model_classes}.")
                self.found_errors += 1
        return label_names

    def is_image_type(self):
        """ :return: True if this configuration is for an image data set."""
        return self.data_type == DataType.IMAGE

    def is_tabular_type(self):
        """ :return: True if this configuration is for a tabular data set."""
        return self.data_type == DataType.TABULAR

    def get_image_sources(self):
        """ :return: Image sources. """
        return self.config['imageData']['sources']

    def _resolve_tabular_source_path(self, source):
        """
        "root": <OPTIONAL: [ dataroot (default) | workspace | relative ] >
        "path": < glob path within the above defined root >

        :param source: The data source stanza
        :return: A list of resolved paths and the label index for all.
        """
        root = source.get('root', 'dataroot')
        path = str(source['path'])
        paths = []
        if root == 'dataroot':
            paths = list(Path(juneberry.DATA_ROOT).glob(path))
        elif root == 'workspace':
            paths = list(Path(juneberry.WORKSPACE_ROOT).glob(path))
        elif root == 'relative':
            if self.relative_path is None:
                logging.error("No relative path set for relative tabular file loading. EXITING")
                sys.exit(-1)
            paths = list(self.relative_path.glob(path))
        else:
            logging.error(f"Unknown source root '{root}'")

        return paths

    def get_resolved_tabular_source_paths_and_labels(self):
        """
        Generates a set of file, index pairs from the source section of the configuration.
        :return: A list of the resolved file paths and the label
        """
        if not self.is_tabular_type():
            logging.error(f"Trying to resolve tabular source on a non-tabular data set. Type is: "
                          f"{self.data_type}. EXITING.")

        file_list = []
        for source in self.config['tabularData']['sources']:
            file_list.extend(self._resolve_tabular_source_path(source))

        return file_list, self.config['tabularData']['labelIndex']

    def has_sampling(self):
        """ :return: True if this has a sampling stanza. """
        return self.sampling is not None

    def get_sampling_config(self):
        """
        :return: Sampling algorithm name, sampling algorithm arguments, and a randomizer for sampling
        """
        sampling_algo = self.sampling['algorithm']
        sampling_args = self.sampling['arguments']

        # Set seed if there is one
        randomizer = None
        if sampling_algo in ("randomFraction", "randomQuantity", "roundRobin"):
            seed = sampling_args.get('seed', None)
            if seed is None:
                logging.error("Sampler 'randomFraction' and 'randomQuantity' require a 'seed' in 'arguments'")
                sys.exit(-1)
            else:
                logging.info("Setting SAMPLING seed to: " + str(sampling_args['seed']))
                randomizer = random.Random()
                randomizer.seed(sampling_args['seed'])

        return sampling_algo, sampling_args, randomizer

    # TODO: Add get label mapping

    # TODO: Add conveniences for source stuff
