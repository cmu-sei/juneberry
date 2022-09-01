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

import logging
import sys

from prodict import List, Prodict

from juneberry.config.plugin import Plugin
import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class ReportConfig(Prodict):
    FORMAT_VERSION = '0.1.0'
    SCHEMA_NAME = 'report_schema.json'
    reports: List[Plugin]

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a config object from a supposedly VALID and LATEST FORMAT report.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed and validated object.
        """

        # Validate
        if not jb_conf_utils.validate_schema(data, ReportConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in ReportConfig from {file_path}. See log. Exiting.")
            sys.exit(-1)

        # Finally, construct the object and do a final value cleanup
        report_config = ReportConfig.from_dict(data)
        return report_config

    @staticmethod
    def load(data_path: str):
        """
        Load the config from the provided path, validate, and construct the config.
        :param data_path: Path to config.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading REPORT CONFIG from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return ReportConfig.construct(data, data_path)
