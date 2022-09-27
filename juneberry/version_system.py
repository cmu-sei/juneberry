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
import sys

logger = logging.getLogger(__name__)


def version_check(data_type: str, config_str, latest_str, use_revision=False):
    """
    Logs an error if a) config version not found, b) config version number is greater than latest version, or c) minor
    or major are different between versions. Logs a warning if versions are the same except for the revision field.
    Versions should be in the form major.minor.revision.
    param data_type: the kind of config file being checked, e.g. experiment, experiment outline, etc.
    param config_str: the formatVersion found in the config file
    param latest_str: the latest formatVersion as specified in the documentation
    param use_revision: set to True if revision field is in use
    """

    if (config_str is None) or (config_str == ''):
        logger.error(f"Failed to find formatVersion in {data_type} config")
        sys.exit(-1)

    else:
        config_num = VersionNumber(config_str, use_revision)
        latest_num = VersionNumber(latest_str, use_revision)

        if config_num > latest_num:
            logger.error(f"{data_type} config formatVersion {config_str} greater than latest version {latest_str}")
            sys.exit(-1)

        elif config_num.major != latest_num.major or config_num.minor != latest_num.minor:
            logger.error(f"{data_type} config formatVersion {config_str} does not match latest version {latest_str}")
            sys.exit(-1)

        elif use_revision:
            if config_num.revision != latest_num.revision:
                logger.warning(f"{data_type} config formatVersion {config_str} "
                               f"revision field does not match latest version {latest_str}")


class VersionNumber:
    """
    Class for comparing linux-like version numbers on config files.
    NOTE: By default revision is NOT required for comparison.
    """

    def __init__(self, version: str, use_revision=False):
        """
        Constructs version number object from a string with 2 or three fields.  The fields
        are major.minor.revision.
        :param version: A string representing the version number.
        :param use_revision: Boolean indicating whether or not to include the revision number in the comparison.
        """
        self.version = version
        self.use_revision = use_revision
        fields = version.split('.')

        if len(fields) == 3:
            self.major = int(fields[0])
            self.minor = int(fields[1])
            if use_revision:
                self.int_version = int(fields[0]) << 16 | int(fields[1]) << 8 | int(fields[2])
                self.revision = int(fields[2])
            else:
                self.int_version = int(fields[0]) << 16 | int(fields[1]) << 8
        elif len(fields) == 2:
            self.major = int(fields[0])
            self.minor = int(fields[1])
            self.int_version = int(fields[0]) << 16 | int(fields[1]) << 8
            logger.warning(f"Given only 2-part version number. Please update! Version: {version}")
        else:
            logger.error(f"Given version string with {len(fields)} fields. We require 2 or 3! Version: {version}")
            sys.exit(-1)

    def __eq__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version == other.int_version

    def __ne__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version != other.int_version

    def __lt__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version < other.int_version

    def __le__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version <= other.int_version

    def __gt__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version > other.int_version

    def __ge__(self, other):
        if isinstance(other, str):
            other = VersionNumber(other)

        return self.int_version >= other.int_version
