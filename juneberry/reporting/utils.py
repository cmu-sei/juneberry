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
from pathlib import Path
from typing import List

from juneberry.config.experiment import ExperimentConfig
from juneberry.config.model import ModelConfig
from juneberry.config.report import ReportConfig
import juneberry.filesystem as jb_fs
import juneberry.loader as jb_loader
from juneberry.reporting.report import Report

logger = logging.getLogger(__name__)


def construct_report(fqcn: str, kwargs: dict) -> Report:
    """
    The purpose of this function is to construct an instance of a Report using an
    FQCN and some args.
    :param fqcn: A string indicating the fqcn of the Report to create.
    :param kwargs: A dictionary containing any args that should be passed to the
    function when building the report instance.
    :return: The resulting Report object.
    """
    return jb_loader.construct_instance(fqcn, kwargs)


def extract_experiment_reports(experiment_name: str) -> List:
    """
    This function is responsible for extracting all the "reports" stanzas from a
    Juneberry experiment's config file.
    :param experiment_name: A string indicating the Juneberry experiment whose config
    will be targeted for "reports" extraction.
    :return: A List of all the "reports" found inside the Experiment's config.json.
    """
    # Create an ExperimentManager for the target Experiment and use it to load the
    # experiment config.
    experiment_manager = jb_fs.ExperimentManager(experiment_name)
    experiment_config = ExperimentConfig.load(experiment_manager.get_experiment_config())

    # Return the "reports" stanza from the experiment config.
    return experiment_config.reports


def extract_model_reports(model_name: str) -> List:
    """
    This function is responsible for extracting all the "reports" stanzas from a
    Juneberry model's config file.
    :param model_name: A string indicating the Juneberry model whose config
    will be targeted for "reports" extraction.
    :return: A List of all the "reports" found inside the Model's config.json.
    """
    # Create a ModelManager for the target Model and use it to load the model config.
    model_manager = jb_fs.ModelManager(model_name)
    model_config = ModelConfig.load(model_manager.get_model_config())

    # Return the "reports" stanza from the model config.
    return model_config.reports


def extract_file_reports(file_list: List[str]) -> List:
    """
    This function is responsible for extracting all the "reports" stanza from a list
    containing strings describing the Path(s) to one (or more) JSON file(s).
    :param file_list: A List of strings, where each string describes the Path to a
    Report config JSON file.
    :return: A combined List of all the "reports" found inside all of the target
    Report config files.
    """
    # Start with an empty list.
    report_list = []

    # Loop through each file in the file_list.
    for file in file_list:

        # Load the current file as a ReportConfig.
        report_config = ReportConfig.load(file)

        # Append each report listed in the Report config to the overall
        # list of reports.
        for report in report_config.reports:
            report_list.append(report)

    # Return the overall list of reports.
    return report_list


def determine_report_path(output_dir: Path, kwarg_str: str, default_filename: str) -> Path:
    """
    This function is responsible for determining the "correct" output Path for a Report file. It makes
    this decision based on three pieces of information: the current output directory for the Report, a
    string (usually from the contents of a Report JSON config) indicating the desired output filename
    for the report, and finally a string indicating what the default filename should be for the report
    if a filename was not provided.
    :param output_dir: A Path to the output directory for the Report. Since the base Report class will
    set an output_dir during initialization of the Report, this path usually reflects that output_dir.
    :param kwarg_str: Most Report classes contain a kwarg that controls the name of the output file
    for the Report. This parameter reflects the string provided to that kwarg.
    :param default_filename: Even though most Reports have a kwarg to control the name of the Report's
    output file, it's possible that the user may forget to provide a value for that kwarg. Should that
    happen, this string represents a default filename to use when no other filename can be determined.
    :return: A Path describing the name and location of the Report's output file.
    """
    # If an output string was not provided, save the report to the
    # current working directory with the default filename.
    if kwarg_str == "":
        return output_dir / default_filename

    # Otherwise, check if the desired output string is a file or directory.
    else:
        output_path = Path(kwarg_str)

        # If the kwarg is already a file, then just return the Path to that file.
        if "." in output_path.parts[-1]:
            return output_path

        # If the kwarg is a directory, then use that as the output directory for
        # a file with the default filename.
        else:
            return output_path / default_filename
