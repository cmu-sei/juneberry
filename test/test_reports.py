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

"""
Unit tests for operations related to Juneberry Reports.
"""

import csv
import json
import os
from pathlib import Path
from unittest import TestCase

from prodict import List, Prodict
import pytest

from juneberry.config.report import ReportConfig
from juneberry.reporting.report import Report
from juneberry.reporting.summary import REPORT_FILES_DIRNAME
import juneberry.reporting.utils as jb_report_utils
import utils


def make_generic_report(description: str = "", fqcn: str = "custom.report", kwargs: dict = None) -> dict:
    """
    The purpose of this function is to set up the minimum amount of information that all Report
    stanzas share.
    :param description: A string describing the report.
    :param fqcn: A string describing which class to use when building this report.
    :param kwargs: A dictionary containing the kwargs to be passed to the indicated fqcn when
    building the report.
    :return: A dictionary representation of a single Report.
    """
    # If kwargs were not provided, set kwargs as an empty dictionary.
    if kwargs is None:
        kwargs = {}

    # Return the basic structure of a Report stanza.
    return {
        "description": description,
        "fqcn": fqcn,
        "kwargs": kwargs
    }


def make_roc_report(description: str = "", output_filename: str = "", plot_title: str = None, line_width: int = None,
                    legend_scaling: float = None, legend_font_size: int = None, curve_sources: dict = None) -> dict:
    """
    This function is responsible for producing a Report stanza for an ROC report.
    """
    # At a minimum, an ROC plot Report requires an "output_filename" kwarg.
    kwargs_dict = {
        "output_filename": output_filename
    }

    # Add the other optional kwargs to the kwargs dict if they were provided.
    if plot_title is not None:
        kwargs_dict["plot_title"] = plot_title

    if line_width is not None:
        kwargs_dict["line_width"] = line_width

    if legend_scaling is not None:
        kwargs_dict["legend_scaling"] = legend_scaling

    if legend_font_size is not None:
        kwargs_dict["legend_font_size"] = legend_font_size

    if curve_sources is not None:
        kwargs_dict["curve_sources"] = curve_sources

    # Build and return the ROC Report stanza.
    return make_generic_report(description=description, fqcn="juneberry.reporting.roc.ROCPlot", kwargs=kwargs_dict)


def make_pr_report(description: str = "", output_dir: str = "", iou: float = None, tp_threshold: float = None,
                   stats_fqcn: str = None, curve_sources: dict = None) -> dict:
    """
    This function is responsible for producing a Report stanza for a PR report.
    """
    # At a minimum, a PR curve Report requires an "output_dir" kwarg.
    kwargs_dict = {
        "output_dir": output_dir
    }

    # Add the other optional kwargs to the kwargs dict if they were provided.
    if iou is not None:
        kwargs_dict["iou"] = iou

    if tp_threshold is not None:
        kwargs_dict["tp_threshold"] = tp_threshold

    if stats_fqcn is not None:
        kwargs_dict["stats_fqcn"] = stats_fqcn

    if curve_sources is not None:
        kwargs_dict["curve_sources"] = curve_sources

    # Build and return the PR Report stanza.
    return make_generic_report(description=description, fqcn="juneberry.reporting.pr.PRCurve", kwargs=kwargs_dict)


def make_summary_report(description: str = "", md_filename: str = "", csv_filename: str = None,
                        metrics_files: List[str] = None, plot_files: List[str] = None) -> dict:
    """
    This function is responsible for producing a Report stanza for a Summary report.
    """
    # At a minimum, a Summary Report requires an "md_filename" kwarg.
    kwargs_dict = {
        "md_filename": md_filename
    }

    # Add the other optional kwargs to the kwargs dict if they were provided.
    if csv_filename is not None:
        kwargs_dict["csv_filename"] = csv_filename

    if metrics_files is not None:
        kwargs_dict["metrics_files"] = metrics_files

    if plot_files is not None:
        kwargs_dict["plot_files"] = plot_files

    # Build and return the Summary Report stanza.
    return make_generic_report(description=description, fqcn="juneberry.reporting.summary.Summary", kwargs=kwargs_dict)


def make_dummy_trained_model(index: int, tmp_path, accuracy_value: float = 0.5, duration: float = 10.0) -> None:
    """
    This function is responsible for creating the minimum amount of information required to fool the Summary
    report into thinking it's retrieving data from a full trained model.
    :param index: An integer that's used to construct unique model and dataset names.
    :param tmp_path: The temporary directory created by pytest during a test invocation. The assumption is that
    this temp directory has been set to the Juneberry workspace, so the model and data_set sub-directories should
    be set to exist inside that workspace.
    :param accuracy_value: A float indicating which accuracy value to use in the model's metrics file.
    :param duration: A float indicating how much time was spent training the model.
    :return: Nothing.
    """
    # Create a dictionary with the minimum amount of content required for a metrics.json file.
    metrics_data = {
        "format_version": "0.1.0",
        "options": {
            "dataset": {
                "config": f"config_{index}.json"
            },
            "model": {
                "name": f"model_{index}"
            }
        },
        "results": {
            "metrics": {
                "classification": {
                    "accuracy": accuracy_value,
                    "balanced_accuracy": accuracy_value
                }
            }
        },
        "times": {
            "duration": duration
        }
    }

    # Determine the location for the metrics file and create the JSON file using the metrics
    # dictionary. Technically, the metrics file should be located inside the model's eval
    # directory, but since the location of the metrics file can be provided directly to the
    # Summary via a kwarg, it's OK to use a simpler path.
    file_path = tmp_path / f"metrics{index}.json"
    with open(file_path, 'w') as out_file:
        json.dump(metrics_data, out_file, indent=4)

    # Create a dictionary with the minimum amount of content required for a training
    # output.json file.
    output_data = {
        "format_version": "0.2.0",
        "options": {
            "batch_size": 0,
            "epochs": 0,
            "model_architecture": {},
            "model_name": "",
            "seed": 0,
            "training_dataset_config_path": "",
        },
        "results": {
            "accuracy": [],
            "loss": [],
            "val_accuracy": [],
            "val_loss": [],
        },
        "times": {
            "duration": duration * 2
        }
    }

    # Determine the location for the training output.json file and create the JSON
    # file using the output_data dictionary. This file will end up being retrieved
    # by the model name, so it's important to place it inside the appropriate models
    # directory.
    output_file = tmp_path / f"models/model_{index}/train/output.json"
    output_file.parent.mkdir(parents=True)
    with open(output_file, 'w') as out_file:
        json.dump(output_data, out_file, indent=4)

    # The Summary report will include a link to the training output image, so that
    # file needs to exist in the model directory. The file does not need any content.
    training_output_img = tmp_path / f"models/model_{index}/train/output.png"
    training_output_img.touch()


def construct_report_from_dict(report_dict: dict) -> Report:
    """
    This function is responsible for constructing a Report object from a report stanza.
    :param report_dict: A dictionary containing all of the information needed to build a single Report.
    :return: A Report object.
    """
    # Convert the dictionary to a Prodict for convenience.
    report = Prodict.from_dict(report_dict)

    # Construct and return the Report object.
    return jb_report_utils.construct_report(report.fqcn, report.kwargs)


def report_config_checks(report_config: ReportConfig) -> None:
    """
    This function is responsible for checking the Reports in a known ReportConfig and verifying that
    all of the attributes for each report have been initialized correctly.
    :param report_config: The ReportConfig to check. There should be 4 reports in this report
    config, and they should be in the following order: the ROC Report, the PR Report, a custom Report,
    and the Summary Report.
    :return: Nothing.
    """
    # Confirm there are 4 Reports in the ReportConfig.
    assert len(report_config.reports) == 4

    # Check the default values for the ROC Report.
    assert report_config.reports[0].description == ""
    assert report_config.reports[0].fqcn == "juneberry.reporting.roc.ROCPlot"
    assert report_config.reports[0].kwargs.output_filename == ""
    # The following keys should not appear in the ROC Report (since they should not have been set).
    for key in ["plot_title", "legend_scaling", "line_width", "legend_font_size", "curve_sources"]:
        assert key not in report_config.reports[0].kwargs

    # Check the default values for the PR Report.
    assert report_config.reports[1].description == ""
    assert report_config.reports[1].fqcn == "juneberry.reporting.pr.PRCurve"
    assert report_config.reports[1].kwargs.output_dir == ""
    # The following keys should not appear in the PR Report (since they should not have been set).
    for key in ["iou", "tp_threshold", "stats_fqcn", "curve_sources"]:
        assert key not in report_config.reports[1].kwargs

    # Check the default values for the Custom Report.
    assert report_config.reports[2].description == ""
    assert report_config.reports[2].fqcn == "custom.report"
    assert report_config.reports[2].kwargs == {}

    # Check the default values for the Summary Report.
    assert report_config.reports[3].description == ""
    assert report_config.reports[3].fqcn == "juneberry.reporting.summary.Summary"
    assert report_config.reports[3].kwargs.md_filename == ""


def test_config_construction(tmp_path) -> None:
    """
    The purpose of this test is to exercise construction of a ReportConfig. A reports stanza is created
    using all four Report types. The default values for each Report type are verified after construction
    of the ReportConfig.
    """
    reports_stanza = {"reports": [make_roc_report(), make_pr_report(), make_generic_report(), make_summary_report()]}
    report_config = ReportConfig.construct(reports_stanza)
    report_config_checks(report_config)


def test_experiment_reports(tmp_path):
    """
    The purpose of this test is to exercise the ability to extract a "reports" stanza from an
    experiment config and use it to produce a list of individual report stanzas.
    """
    # A name for the "dummy" experiment in this test.
    experiment_name = "test_experiment"

    # Change to the pytest temporary directory for this test.
    os.chdir(tmp_path)

    # Create a sample experiment config using the function from the experiment config unit tests.
    exp_config = utils.make_basic_experiment_config()

    # Determine the location for the experiment config, create the directory, and save the
    # data to the config file.
    exp_config_path = tmp_path / f"experiments/{experiment_name}/config.json"
    exp_config_path.parent.mkdir(parents=True)
    with open(exp_config_path, 'w') as json_file:
        json.dump(exp_config, json_file)

    # The ExperimentConfig will throw an error when it is loaded if the model directory
    # does not exist, so create a fake model directory inside the workspace.
    model_dir_path = tmp_path / "models/tabular_binary_sample"
    model_dir_path.mkdir(parents=True)

    # The ExperimentConfig will throw an error when it is loaded if the training dataset
    # does not exist, so create a fake dataset config inside the workspace.
    dataset_config_path = tmp_path / "data_sets/train_data_config.json"
    dataset_config_path.parent.mkdir(parents=True)
    dataset_config_path.touch()

    # Retrieve the list of Reports from the experiment config and verify the properties of
    # the Report that was extracted.
    extracted_reports_list = jb_report_utils.extract_experiment_reports(experiment_name)
    assert len(extracted_reports_list) == 1
    assert extracted_reports_list[0].description == "basic description"
    assert extracted_reports_list[0].fqcn == "juneberry.reporting.roc.ROCPlot"
    assert extracted_reports_list[0].kwargs.output_filename == "sample_roc_1.png"
    assert extracted_reports_list[0].kwargs.plot_title == "Sample ROC Plot"
    assert extracted_reports_list[0].tests == [{"tag": "pyt50", "classes": "0"}]


def test_model_reports(tmp_path):
    """
    The purpose of this test is to exercise the ability to extract a "reports" stanza from a
    model config and use it to produce a list of individual report stanzas.
    """
    # A name for the "dummy" model in this test.
    model_name = "test_model"

    # Change to the pytest temporary directory for this test.
    os.chdir(tmp_path)

    # Create a sample model config using the function from the model config unit tests. Add
    # a reports stanza to the model config.
    model_config_dict = utils.make_basic_model_config()
    model_config_dict["reports"] = [make_roc_report(), make_pr_report(), make_generic_report(), make_summary_report()]

    # Determine the location for the model config, create the directory, and save the
    # data to the config file.
    model_config_path = tmp_path / f"models/{model_name}/config.json"
    model_config_path.parent.mkdir(parents=True)
    with open(model_config_path, "w") as json_file:
        json.dump(model_config_dict, json_file)

    # Retrieve the list of Reports from the model config and verify the properties of all
    # four reports in the list.
    extracted_reports_list = jb_report_utils.extract_model_reports(model_name)
    report_config = ReportConfig.construct({"reports": extracted_reports_list})
    report_config_checks(report_config)


class TestROCReport(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply only to the
    ROC Report type.
    """

    @pytest.fixture(autouse=True)
    def init_tmp_path(self, tmp_path):
        """
        The purpose of this function is to make the pytest tmp_path fixture available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path

    def test_report_init_defaults(self) -> None:
        """
        The purpose of this test is to exercise that an ROC Report will initialize
        with the correct default values.
        """
        # Create a Report stanza for an ROC report with the minimum amount of data.
        config_file_content = make_roc_report(output_filename="")

        # Construct a Report object using that stanza.
        roc_report = construct_report_from_dict(config_file_content)

        # Verify that the Report object reflects the default values for an ROC Report.
        assert roc_report.report_path == Path.cwd() / "ROC_curves.png"
        assert roc_report.plot_title == "ROC Curve(s)"
        assert roc_report.legend_scaling == 1.0
        assert roc_report.line_width == 2
        assert roc_report.legend_font_size == 10
        assert roc_report.curve_sources is None

    def test_report_init_values(self) -> None:
        """
        The purpose of this test is to exercise that an ROC Report will set the correct
        "report_path" when a file is provided as the value for the "output_filename".
        """
        # Establish the values to test.
        output_filename = f"{self.tmp_path}/test/test.png"
        plot_title = "Test Title"
        legend_scaling = 100.0
        line_width = 15
        legend_font_size = 3
        curve_sources = {"predictions.json": "0,1,2"}

        # Create a Report stanza for an ROC report using the desired values.
        config_file_content = make_roc_report(output_filename=output_filename, plot_title=plot_title,
                                              legend_scaling=legend_scaling, line_width=line_width,
                                              legend_font_size=legend_font_size, curve_sources=curve_sources)

        # Construct a Report object using that stanza.
        roc_report = construct_report_from_dict(config_file_content)

        # Verify that the Report object reflects the desired values.
        assert roc_report.report_path == Path(output_filename)
        assert roc_report.plot_title == plot_title
        assert roc_report.legend_scaling == legend_scaling
        assert roc_report.line_width == line_width
        assert roc_report.legend_font_size == legend_font_size
        assert roc_report.curve_sources == curve_sources

    def test_output_filename_as_dir(self):
        """
        The purpose of this test is to exercise that an ROC Report will set the correct
        "report_path" when a directory is provided as the value for the "output_filename".
        """
        # Establish a directory to test.
        test_directory = str(self.tmp_path / "test")

        # Create a Report stanza for an ROC report using the test_directory.
        config_file_content = make_roc_report(output_filename=test_directory)

        # Construct a Report object using that stanza.
        roc_report = construct_report_from_dict(config_file_content)

        # Verify that the report_path is equivalent to the default ROC output filename, but
        # located inside the directory that was provided as the output_filename.
        assert roc_report.report_path == Path(test_directory) / "ROC_curves.png"


class TestPRReport(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply only to the
    PR Report type.
    """

    def test_report_init_defaults(self):
        """
        The purpose of this test is to exercise that a PR Report will initialize
        with the correct default values.
        """
        # Create a Report stanza for a PR report with the minimum amount of data.
        config_file_content = make_pr_report(output_dir="")

        # Construct a Report object using that stanza.
        pr_report = construct_report_from_dict(config_file_content)

        # Verify that the Report object reflects the default values for a PR Report.
        assert pr_report.output_dir == Path.cwd()
        assert pr_report.iou == 0.5
        assert pr_report.tp_threshold == 0.8
        assert pr_report.stats_fqcn == "juneberry.metrics.objectdetection.brambox.metrics.Summary"
        assert pr_report.curve_sources is None

    def test_report_init_values(self):
        """
        The purpose of this test is to exercise that a PR Report will set the correct
        "report_path" when a file is provided as the value for the "output_filename".
        """
        # Establish the values to test.
        output_dir = "test/"
        iou = 0.5
        curve_sources = {"key": "value"}
        tp_threshold = 0.5
        stats_fqcn = "metrics.Summary"

        # Create a Report stanza for a PR report using the desired values.
        config_file_content = make_pr_report(output_dir=output_dir, iou=iou, curve_sources=curve_sources,
                                             tp_threshold=tp_threshold, stats_fqcn=stats_fqcn)

        # Construct a Report object using that stanza.
        pr_report = construct_report_from_dict(config_file_content)

        # Verify that the Report object reflects the desired values.
        assert pr_report.output_dir == Path(output_dir)
        assert pr_report.iou == iou
        assert pr_report.curve_sources == curve_sources
        assert pr_report.tp_threshold == tp_threshold
        assert pr_report.stats_fqcn == stats_fqcn

    def test_output_dir_as_file(self):
        """
        The purpose of this test is to exercise that a PR Report will set the correct
        "output_dir" when a filename is provided as the value for the "output_dir".
        """
        # Establish the filename to test.
        test_filename = "test/test.png"

        # Create a Report stanza for a PR report using the desired values.
        config_file_content = make_pr_report(output_dir=test_filename)

        # Construct a Report object using that stanza.
        pr_report = construct_report_from_dict(config_file_content)

        # Verify that the output_dir is equivalent to the parent directory of the provided filename.
        assert pr_report.output_dir == Path(test_filename).parent


class TestSummaryReport(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply only to the
    Summary Report type.
    """

    def test_report_init_defaults(self) -> None:
        """
        The purpose of this test is to exercise that a Summary Report will initialize
        with the correct default values.
        """
        # Create a Report stanza for a Summary report with the minimum amount of data.
        config_file_content = make_summary_report(md_filename="")

        # Construct a Report object using that stanza.
        summary_report = construct_report_from_dict(config_file_content)

        # Verify that the Report object reflects the default values for a Summary Report.
        assert summary_report.report_path == Path.cwd() / "summary.md"
        assert summary_report.csv_path is None
        assert summary_report.metrics_files is None
        assert summary_report.plot_files is None
        assert summary_report.metric is None

    def test_md_and_csv_output_paths(self) -> None:
        """
        The purpose of this test is to verify that the Summary Report will correctly
        set the "report_path" and "csv_path" attributes correctly under a variety of conditions.
        """
        # Create a Report stanza for a Summary report with the minimum amount of data.
        config_file_content = make_summary_report(md_filename="")

        # Construct a Report object using that stanza.
        summary_report = construct_report_from_dict(config_file_content)

        # Confirm the correct default values for the report_path and csv_path.
        assert summary_report.report_path == Path.cwd() / "summary.md"
        assert summary_report.csv_path is None

        # Adjust the report stanza so that the md_filename and csv_filename are
        # both directories. Construct a new Report object with the stanza changes.
        config_file_content["kwargs"]["md_filename"] = "test/"
        config_file_content["kwargs"]["csv_filename"] = "test/"
        summary_report = construct_report_from_dict(config_file_content)

        # The expected behavior is that the report_path and csv_path should be set
        # to the default filenames, but inside the directory indicated by the stanza.
        assert summary_report.report_path == Path("test") / "summary.md"
        assert summary_report.csv_path == Path("test") / "summary.csv"

        # Adjust the report stanza so that the md_filename and csv_filename are
        # both filenames. Construct a new Report object with the stanza changes.
        config_file_content["kwargs"]["md_filename"] = "test/file.md"
        config_file_content["kwargs"]["csv_filename"] = "test/file.csv"
        summary_report = construct_report_from_dict(config_file_content)

        # The expected behavior is the the report_path and csv_path should be set
        # to the filenames indicated in the report stanza.
        assert summary_report.report_path == Path("test") / "file.md"
        assert summary_report.csv_path == Path("test") / "file.csv"

    @pytest.fixture(autouse=True)
    def init_tmp_path(self, tmp_path):
        """
        The purpose of this function is to make the pytest tmp_path fixture available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path

    def test_summary_construction(self):
        """
        The purpose of this test is to exercise the creation of a Summary Report.
        """
        # Change to the pytest temporary directory for this test.
        os.chdir(self.tmp_path)

        # The Summary Report pulls data from a metrics file and other output files from model
        # training. So create two model directories with dummy training data.
        metrics_files = []
        for i in range(2):
            file_path = self.tmp_path / f"metrics{i}.json"
            metrics_files.append(str(file_path))
            make_dummy_trained_model(i, self.tmp_path, accuracy_value=(0.1 * (i + 1)), duration=(5 * (i + 1)))

        # Create the report stanza for the Summary Report, construct the Report object,
        # and create the Summary Report.
        report_dict = make_summary_report(md_filename="", csv_filename="summary.csv", metrics_files=metrics_files)
        report = Prodict.from_dict(report_dict)
        summary_report = jb_report_utils.construct_report(report.fqcn, report.kwargs)
        summary_report.create_report()

        # Load the markdown file that was created by the Summary Report.
        md_path = self.tmp_path / "summary.md"
        with open(md_path) as md_file:
            md_file_content = md_file.read().split("\n")

        # Verify that each line in the markdown file matches the expected values.
        assert md_file_content[0] == "# Experiment summary"
        assert md_file_content[1] == "Model | Duration (seconds) | Eval Dataset | Accuracy | Train Chart"
        assert md_file_content[2] == "--- | --- | --- | --- | ---"
        assert md_file_content[3] == f"model_0 | 10.0 | config_0.json | 10.00% | " \
                                     f"[Training Chart](./{REPORT_FILES_DIRNAME}/model_0_output.png)"
        assert md_file_content[4] == f"model_1 | 20.0 | config_1.json | 20.00% | " \
                                     f"[Training Chart](./{REPORT_FILES_DIRNAME}/model_1_output.png)"

        # Load the CSV file that was created by the Summary Report.
        csv_path = self.tmp_path / "summary.csv"
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            row_list = list(reader)[0:]

        # Verify that each row of the CSV file matches the expected values.
        assert row_list[0] == ['model', 'duration', 'eval_dataset', "{'Accuracy'}"]
        assert row_list[1] == ['model_0', '10.0', 'config_0.json', '10.00%']
        assert row_list[2] == ['model_1', '20.0', 'config_1.json', '20.00%']
