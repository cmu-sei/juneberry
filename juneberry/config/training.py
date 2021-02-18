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

import random
import logging
from pathlib import Path
import juneberry.filesystem as jbfs

FORMAT_VERSION = '3.3.0'


# TODO: Switch config to come from a factory, such as file factory.
class TrainingConfig:
    """
    An object used to parse and manage training configurations.
    """

    def __init__(self, model_name, config):
        self.valid = True
        self.model_name = model_name
        self.config = config.copy()

        # Check for REQUIRED items in config file.
        for param in ['dataSetConfigPath', 'platform', 'epochs', 'batchSize', 'seed', 'modelArchitecture']:
            if param not in config:
                logging.error(f"Failed to find {param} in TRAINING config")
                self.valid = False

        # Pull out the values
        self.data_set_path = Path(config.get('dataSetConfigPath', None))
        self.platform = config.get('platform', None)
        self.epochs = config.get('epochs', None)
        self.batch_size = config.get('batchSize', None)
        self.seed = config.get('seed', None)
        self.model_architecture = config.get('modelArchitecture', None)
        self.format_version = config.get('formatVersion', None)

        if 'args' not in self.model_architecture:
            self.model_architecture['args'] = {}

        # Check formatVersion
        jbfs.version_check("TRAINING", self.format_version, FORMAT_VERSION, True)

        # Compiler args
        self.pytorch = config.get('pytorch', {})

        self.training_transforms = config.get('trainingTransforms', None)
        self.prediction_transforms = config.get('predictionTransforms', None)

        self.validation = config.get('validation', None)

        # Set some default values
        if 'stopping' not in config:
            self.config['stopping'] = {}

        stopping_options = self.config['stopping']
        if 'plateau_abs_tol' not in stopping_options:
            stopping_options['plateau_abs_tol'] = 0.0001

    def __setitem__(self, key, value):
        self.config[key] = value

    def __getitem__(self, item):
        return self.config[item]

    def get(self, key, default):
        return self.config.get(key, default)

    def get_previous_model(self):
        """ :return" Previous model architecture and version from which to load weights. """
        return self.model_architecture.get('previousModel', None), self.model_architecture.get('previousModelVersion',
                                                                                               None)

    def has_validation_split(self):
        """ :return: True if validation stanza exists. """
        return self.validation is not None

    def get_validation_split_config(self):
        """
        :return: Algorithm name, algorithm arguments, and a randomizer for validation splitting.
        """
        splitting_algo = self.validation['algorithm']
        splitting_args = self.validation['arguments']

        # Set seed if there is one
        randomizer = None
        if splitting_algo == "randomFraction":
            if 'seed' in splitting_args:
                logging.info("Setting VALIDATION seed to: " + str(splitting_args['seed']))
                randomizer = random.Random()
                randomizer.seed(splitting_args['seed'])

        return splitting_algo, splitting_args, randomizer
