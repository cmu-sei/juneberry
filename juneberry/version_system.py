#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
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
