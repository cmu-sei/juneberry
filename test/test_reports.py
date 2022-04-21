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

"""
Unit tests for operations related to Juneberry Reports.
"""

from argparse import Namespace
import csv
import json
from pathlib import Path
from prodict import List, Prodict
import pytest
from unittest import TestCase

from juneberry.config.report import ReportConfig
import juneberry.reporting.utils as jb_report_utils
from juneberry.scripting import setup_workspace

from test_experiment_config import make_basic_config as make_experiment_config
from test_model_config import make_basic_config as make_model_config


def make_roc_report(description: str = "ROC Report", output_filename: str = "test_roc.png",
                    plot_title: str = "Test ROC", legend_scaling: float = 0.5, line_width: int = 2,
                    legend_font_size: int = 10,
                    curve_sources: Prodict = Prodict.from_dict({"models/eval/predictions.json": "0,1,2"})):

    return {
        "description": description,
        "fqcn": "juneberry.reporting.roc.ROCPlot",
        "kwargs": {
            "output_filename": output_filename,
            "plot_title": plot_title,
            "legend_scaling": legend_scaling,
            "line_width": line_width,
            "legend_font_size": legend_font_size,
            "curve_sources": curve_sources
        }
    }


def make_pr_report(description: str = "PR Report", output_dir: str = "example/dir", iou: float = 1.0,
                   tp_threshold: float = 0.8, stats_fqcn: str = "juneberry.metrics.metrics.Stats",
                   curve_sources: Prodict = Prodict.from_dict({"model_name": "data_sets/dataset_name.json"})):
    return {
        "description": description,
        "fqcn": "juneberry.reporting.pr.PRCurve",
        "kwargs": {
            "output_dir": output_dir,
            "iou": iou,
            "tp_threshold": tp_threshold,
            "stats_fqcn": stats_fqcn,
            "curve_sources": curve_sources
        }
    }


def make_summary_report(description: str = "Summary Report", md_filename: str = "summary.md",
                        csv_filename: str = "summary.csv", metrics_files: List = None, plot_files: List = None):
    if metrics_files is None:
        metrics_files = ["metrics1.json", "metrics2.json"]

    if plot_files is None:
        plot_files = ["plot1.png", "plot2.png"]

    return {
        "description": description,
        "fqcn": "juneberry.reporting.summary.Summary",
        "kwargs": {
            "md_filename": md_filename,
            "csv_filename": csv_filename,
            "metrics_files": metrics_files,
            "plot_files": plot_files
        }
    }


def make_generic_report(description: str = "Generic Report", fqcn: str = "custom.report", kwargs: dict = None):

    if kwargs is None:
        kwargs = {
            "arg1": "cat",
            "arg2": "apple",
            "arg3": "potato"
        }

    return {
        "description": description,
        "fqcn": fqcn,
        "kwargs": kwargs
    }


def make_dummy_model_dir(index: int, tmp_path, accuracy_value: float = 0.5, duration: float = 10.0):
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
                "accuracy": accuracy_value,
                "balanced_accuracy": accuracy_value
            }
        },
        "times": {
            "duration": duration
        }
    }

    file_path = tmp_path / f"metrics{index}.json"
    with open(file_path, 'w') as out_file:
        json.dump(metrics_data, out_file, indent=4)

    output_data = {
        "format_version": "0.2.0",
        "options": {
            "batch_size": 0,
            "epochs": 0,
            "model_architecture": {},
            "seed": 0,
            "training_dataset_config_path": "",
        },
        "results": {
            "accuracy": [],
            "loss": [],
            "model_name": "",
            "val_accuracy": [],
            "val_loss": [],
        },
        "times": {
            "duration": duration * 2
        }
    }

    output_file = tmp_path / f"models/model_{index}/train/output.json"
    output_file.parent.mkdir(parents=True)
    with open(output_file, 'w') as out_file:
        json.dump(output_data, out_file, indent=4)

    training_output_img = tmp_path / f"models/model_{index}/train/output.png"
    training_output_img.touch()


def create_report_obj(report_dict: dict):
    report = Prodict.from_dict(report_dict)
    return jb_report_utils.construct_report(report.fqcn, report.kwargs)


def report_config_checks(report_config):
    assert len(report_config.reports) == 4

    assert report_config.reports[0].description == "ROC Report"
    assert report_config.reports[0].fqcn == "juneberry.reporting.roc.ROCPlot"
    assert report_config.reports[0].kwargs.output_filename == "test_roc.png"
    assert report_config.reports[0].kwargs.plot_title == "Test ROC"
    assert report_config.reports[0].kwargs.legend_scaling == 0.5
    assert report_config.reports[0].kwargs.line_width == 2
    assert report_config.reports[0].kwargs.legend_font_size == 10
    assert report_config.reports[0].kwargs.curve_sources == {"models/eval/predictions.json": "0,1,2"}

    assert report_config.reports[1].description == "PR Report"
    assert report_config.reports[1].fqcn == "juneberry.reporting.pr.PRCurve"
    assert report_config.reports[1].kwargs.output_dir == "example/dir"
    assert report_config.reports[1].kwargs.iou == 1.0
    assert report_config.reports[1].kwargs.tp_threshold == 0.8
    assert report_config.reports[1].kwargs.stats_fqcn == "juneberry.metrics.metrics.Stats"
    assert report_config.reports[1].kwargs.curve_sources == {"model_name": "data_sets/dataset_name.json"}

    assert report_config.reports[2].description == "Generic Report"
    assert report_config.reports[2].fqcn == "custom.report"
    assert report_config.reports[2].kwargs.arg1 == "cat"
    assert report_config.reports[2].kwargs.arg2 == "apple"
    assert report_config.reports[2].kwargs.arg3 == "potato"

    assert report_config.reports[3].description == "Summary Report"
    assert report_config.reports[3].fqcn == "juneberry.reporting.summary.Summary"
    assert report_config.reports[3].kwargs.md_filename == "summary.md"
    assert report_config.reports[3].kwargs.csv_filename == "summary.csv"
    assert report_config.reports[3].kwargs.metrics_files == ["metrics1.json", "metrics2.json"]
    assert report_config.reports[3].kwargs.plot_files == ["plot1.png", "plot2.png"]


def test_config_construction(tmp_path):
    reports_list = [make_roc_report(), make_pr_report(), make_generic_report(), make_summary_report()]
    reports_stanza = {"reports": reports_list}

    report_config = ReportConfig.construct(reports_stanza)
    report_config_checks(report_config)


def test_experiment_reports(tmp_path):
    # Create workspace in tmp_path
    args = Namespace(workspace=tmp_path, dataRoot=None, tensorboard=None, silent=True, verbose=False,
                     profileName=None, logDir=None)
    setup_workspace(args, log_file=None)

    exp_config = make_experiment_config()
    exp_config_path = tmp_path / "experiments/test_experiment/config.json"
    model_dir_path = tmp_path / "models/imagenette_160x160_rgb_unit_test_pyt_resnet18"
    model_dir_path.mkdir(parents=True)
    dataset_config_path = tmp_path / "data_sets/imagenette_unit_test.json"
    dataset_config_path.parent.mkdir(parents=True)
    dataset_config_path.touch()
    exp_config_path.parent.mkdir(parents=True)
    with open(exp_config_path, 'w') as json_file:
        json.dump(exp_config, json_file)

    extracted_reports_list = jb_report_utils.extract_experiment_reports("test_experiment")

    assert extracted_reports_list[0].description == "basic description"
    assert extracted_reports_list[0].fqcn == "juneberry.reporting.roc.ROCPlot"
    assert extracted_reports_list[0].kwargs.output_filename == "sample_roc_1.png"
    assert extracted_reports_list[0].kwargs.plot_title == "Sample ROC Plot"
    assert extracted_reports_list[0].tests == [{"tag": "pyt50", "classes": "0"}]


def test_model_reports(tmp_path):
    args = Namespace(workspace=tmp_path, dataRoot=None, tensorboard=None, silent=True, verbose=False,
                     profileName=None, logDir=None)
    setup_workspace(args, log_file=None)

    reports_list = [make_roc_report(), make_pr_report(), make_generic_report(), make_summary_report()]
    model_config_dict = make_model_config()
    model_config_dict["reports"] = reports_list
    model_config_path = tmp_path / "models/test_model/config.json"
    model_config_path.parent.mkdir(parents=True)
    with open(model_config_path, "w") as json_file:
        json.dump(model_config_dict, json_file)

    extracted_reports_list = jb_report_utils.extract_model_reports("test_model")
    reports_stanza = {"reports": extracted_reports_list}

    report_config = ReportConfig.construct(reports_stanza)
    report_config_checks(report_config)


class TestROCReport(TestCase):

    def test_report_init_defaults(self):
        config_file_content = make_roc_report(output_filename="", plot_title="")
        del config_file_content["kwargs"]["legend_scaling"]
        del config_file_content["kwargs"]["legend_font_size"]
        del config_file_content["kwargs"]["line_width"]
        del config_file_content["kwargs"]["curve_sources"]

        roc_report_obj = create_report_obj(config_file_content)

        assert roc_report_obj.report_path == Path.cwd() / "ROC_curves.png"
        assert roc_report_obj.plot_title == "ROC Curve(s)"
        assert roc_report_obj.legend_scaling == 1.0
        assert roc_report_obj.line_width == 2
        assert roc_report_obj.legend_font_size == 10
        assert roc_report_obj.curve_sources is None

    def test_output_filename_as_dir(self):
        config_file_content = make_roc_report(output_filename="/test", plot_title="")
        roc_report_obj = create_report_obj(config_file_content)

        assert roc_report_obj.report_path == Path("/test") / "ROC_curves.png"


class TestPRReport(TestCase):

    def test_report_init_defaults(self):
        config_file_content = make_pr_report(output_dir="")
        del config_file_content["kwargs"]["iou"]
        del config_file_content["kwargs"]["tp_threshold"]
        del config_file_content["kwargs"]["stats_fqcn"]
        del config_file_content["kwargs"]["curve_sources"]

        pr_report_obj = create_report_obj(config_file_content)

        assert pr_report_obj.output_dir == Path.cwd()
        assert pr_report_obj.iou == 1.0
        assert pr_report_obj.tp_threshold == 0.8
        assert pr_report_obj.stats_fqcn == "juneberry.metrics.metrics.Stats"
        assert pr_report_obj.curve_sources is None

    def test_output_dir_as_file(self):
        config_file_content = make_pr_report(output_dir="test/test.png")
        pr_report_obj = create_report_obj(config_file_content)

        assert pr_report_obj.output_dir == Path("test/")


class TestSummaryReport(TestCase):

    def test_report_init_defaults(self):
        config_file_content = make_summary_report(md_filename="")
        del config_file_content["kwargs"]["csv_filename"]
        del config_file_content["kwargs"]["metrics_files"]
        del config_file_content["kwargs"]["plot_files"]

        summary_report_obj = create_report_obj(config_file_content)

        assert summary_report_obj.report_path == Path.cwd() / "summary.md"
        assert summary_report_obj.csv_path is None
        assert summary_report_obj.metrics_files is None
        assert summary_report_obj.plot_files is None
        assert summary_report_obj.metric is None

    def test_md_and_csv_output_paths(self):
        config_file_content = make_summary_report(md_filename="")
        del config_file_content["kwargs"]["metrics_files"]
        del config_file_content["kwargs"]["csv_filename"]
        summary_report_obj = create_report_obj(config_file_content)

        assert summary_report_obj.report_path == Path.cwd() / "summary.md"
        assert summary_report_obj.csv_path is None

        config_file_content["kwargs"]["md_filename"] = "test/"
        config_file_content["kwargs"]["csv_filename"] = "test/"
        summary_report_obj = create_report_obj(config_file_content)

        assert summary_report_obj.report_path == Path("test") / "summary.md"
        assert summary_report_obj.csv_path == Path("test") / "summary.csv"

        config_file_content["kwargs"]["md_filename"] = "test/file.md"
        config_file_content["kwargs"]["csv_filename"] = "test/file.csv"
        summary_report_obj = create_report_obj(config_file_content)

        assert summary_report_obj.report_path == Path("test") / "file.md"
        assert summary_report_obj.csv_path == Path("test") / "file.csv"

    @pytest.fixture(autouse=True)
    def init_tmp_path(self, tmp_path):
        self.tmp_path = tmp_path

    def test_summary_construction(self):
        args = Namespace(workspace=self.tmp_path, dataRoot=None, tensorboard=None, silent=True, verbose=False,
                         profileName=None, logDir=None)
        setup_workspace(args, log_file=None)

        metrics_files = []
        for i in range(2):
            file_path = self.tmp_path / f"metrics{i}.json"
            metrics_files.append(str(file_path))
            make_dummy_model_dir(i, self.tmp_path, accuracy_value=(0.1 * (i + 1)), duration=(5 * (i + 1)))

        config_file_content = make_summary_report(md_filename="summary.md", csv_filename="summary.csv",
                                                  metrics_files=metrics_files)
        del config_file_content["kwargs"]["plot_files"]

        report = Prodict.from_dict(config_file_content)
        summary_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)
        summary_report_obj.create_report()

        md_path = self.tmp_path / "summary.md"
        with open(md_path) as md_file:
            content = md_file.read().split("\n")

        assert content[0] == "# Experiment summary"
        assert content[1] == "Model | Duration (seconds) | Eval Dataset | Accuracy | Train Chart"
        assert content[2] == "--- | --- | --- | --- | ---"
        assert content[3] == f"model_0 | 10.0 | config_0.json | 10.00% | " \
                             f"[Training Chart](./report_files/model_0_output.png)"
        assert content[4] == f"model_1 | 20.0 | config_1.json | 20.00% | " \
                             f"[Training Chart](./report_files/model_1_output.png)"

        csv_path = self.tmp_path / "summary.csv"
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            row_list = list(reader)[0:]

        assert row_list[0] == ['model', 'duration', 'eval_dataset', "{'Accuracy'}"]
        assert row_list[1] == ['model_0', '10.0', 'config_0.json', '0.1']
        assert row_list[2] == ['model_1', '20.0', 'config_1.json', '0.2']
