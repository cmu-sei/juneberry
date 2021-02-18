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
import logging
import juneberry.filesystem as jbfs

FORMAT_VERSION = '1.0.0'


class ExperimentOutline:
    def __init__(self, experiment_name, config):
        self.valid = True
        self.experiment_name = experiment_name
        self.config = config.copy()

        # Check for REQUIRED items
        for param in ['baselineConfig', 'variables']:
            if param not in config:
                logging.error(f"Failed to find {param} in experiment outline!")
                self.valid = False

        # Pull out the values
        self.baseline_config = config.get('baselineConfig', None)
        self.variables = config.get('variables', None)
        self.tests = config.get('tests', None)
        self.reports = config.get('reports', None)
        self.format_version = config.get('formatVersion', None)

        # Check formatVersion
        jbfs.version_check("EXPERIMENT OUTLINE", self.format_version, FORMAT_VERSION, True)

        # Verify variables aren't constants
        self.check_experiment_variables()

    def analyze_experiment_variables(self):
        """
        This method identifies the experiment variables and calculates the number of possible combinations.
        :return: Nothing.
        """
        logging.info(f"Identified {len(self.variables)} variables:")

        combinations = 1

        for variable in self.variables:
            if type(variable['values']) is str:
                count = 1
                logging.info(f"  {count:3d} random value  for {variable['configField']}")
            else:
                count = len(variable['values'])
                logging.info(f"  {count:3d} possibilities for {variable['configField']}")

            combinations *= count

        logging.info(f"{combinations:5d} unique configurations in the outline file for '{self.experiment_name}'.")

    def check_experiment_variables(self):
        """
        This method verifies that each variable in the experiment has more than one possibility. If a variable
        has only one possibility, then it should not be a variable.
        :return: Nothing.
        """

        for variable in self.variables:

            if type(variable['values']) is list:
                count = len(variable['values'])

                if count < 2:
                    self.valid = False
                    logging.error(f"Insufficient possibilities for '{variable}' in '{self.experiment_name}'. "
                                  f"this variable from the experiment outline or add more possibilities.")
                    sys.exit(-1)
