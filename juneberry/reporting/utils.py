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
from pathlib import Path
from typing import List

from juneberry.config.experiment import ExperimentConfig
from juneberry.config.report import ReportConfig
import juneberry.filesystem as jbfs
import juneberry.loader as jb_loader

logger = logging.getLogger(__name__)


def construct_report(fqcn, args):
    report = jb_loader.construct_instance(fqcn, args)
    return report


def extract_experiment_reports(experiment_name: str):
    experiment_manager = jbfs.ExperimentManager(experiment_name)
    experiment_config = ExperimentConfig.load(experiment_manager.get_experiment_config())

    return experiment_config.reports


def extract_model_reports():
    pass


def extract_file_reports(file_list: List):
    report_list = []

    for file in file_list:
        report_config = ReportConfig.load(file)
        for report in report_config.reports:
            report_list.append(report)

    return report_list


def determine_report_path(output_dir: Path, input_str: str, default_filename: str):
    # If an output string was not provided, save the report to the
    # current working directory with the default filename.
    if input_str == "":
        return output_dir / default_filename

    # Otherwise, check if the provided output string is a file or directory.
    else:
        output_path = Path(input_str)

        if "." in output_path.parts[-1]:
            return output_path
        else:
            return output_path / default_filename
