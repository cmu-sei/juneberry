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

from pathlib import Path
from prodict import List, Prodict
from unittest import TestCase

from juneberry.config.report import ReportConfig
import juneberry.reporting.utils as jb_report_utils


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


def test_config_basics():
    reports_list = [make_roc_report(), make_pr_report(), make_generic_report(), make_summary_report()]
    config_file_content = {"reports": reports_list}

    report_config = ReportConfig.construct(config_file_content)

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


class TestROCReport(TestCase):

    def test_report_init_defaults(self):
        config_file_content = make_roc_report(output_filename="", plot_title="")
        del config_file_content["kwargs"]["legend_scaling"]
        del config_file_content["kwargs"]["legend_font_size"]
        del config_file_content["kwargs"]["line_width"]
        del config_file_content["kwargs"]["curve_sources"]

        report = Prodict.from_dict(config_file_content)
        roc_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert roc_report_obj.report_path == Path.cwd() / "ROC_curves.png"
        assert roc_report_obj.plot_title == "ROC Curve(s)"
        assert roc_report_obj.legend_scaling == 1.0
        assert roc_report_obj.line_width == 2
        assert roc_report_obj.legend_font_size == 10
        assert roc_report_obj.curve_sources is None

    def test_output_filename_as_dir(self):
        config_file_content = make_roc_report(output_filename="/test", plot_title="")
        report = Prodict.from_dict(config_file_content)
        roc_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert roc_report_obj.report_path == Path("/test") / "ROC_curves.png"


class TestPRReport(TestCase):

    def test_report_init_defaults(self):
        config_file_content = make_pr_report(output_dir="")
        del config_file_content["kwargs"]["iou"]
        del config_file_content["kwargs"]["tp_threshold"]
        del config_file_content["kwargs"]["stats_fqcn"]
        del config_file_content["kwargs"]["curve_sources"]

        report = Prodict.from_dict(config_file_content)
        pr_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert pr_report_obj.output_dir == Path.cwd()
        assert pr_report_obj.iou == 1.0
        assert pr_report_obj.tp_threshold == 0.8
        assert pr_report_obj.stats_fqcn == "juneberry.metrics.metrics.Stats"
        assert pr_report_obj.curve_sources is None

    def test_output_dir_as_file(self):
        config_file_content = make_pr_report(output_dir="test/test.png")
        report = Prodict.from_dict(config_file_content)
        pr_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert pr_report_obj.output_dir == Path("test/")


class TestSummaryReport(TestCase):

    def test_report_init_defaults(self):
        config_file_content = make_summary_report(md_filename="")
        del config_file_content["kwargs"]["csv_filename"]
        del config_file_content["kwargs"]["metrics_files"]
        del config_file_content["kwargs"]["plot_files"]

        report = Prodict.from_dict(config_file_content)
        summary_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert summary_report_obj.report_path == Path.cwd() / "summary.md"
        assert summary_report_obj.csv_path is None
        assert summary_report_obj.metrics_files is None
        assert summary_report_obj.plot_files is None
        assert summary_report_obj.metric is None

    def test_md_and_csv_output_paths(self):
        config_file_content = make_summary_report(md_filename="")
        del config_file_content["kwargs"]["metrics_files"]
        del config_file_content["kwargs"]["csv_filename"]
        report = Prodict.from_dict(config_file_content)
        summary_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert summary_report_obj.report_path == Path.cwd() / "summary.md"
        assert summary_report_obj.csv_path is None

        report.kwargs.md_filename = "test/"
        report.kwargs.csv_filename = "test/"
        summary_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert summary_report_obj.report_path == Path("test") / "summary.md"
        assert summary_report_obj.csv_path == Path("test") / "summary.csv"

        report.kwargs.md_filename = "test/file.md"
        report.kwargs.csv_filename = "test/file.csv"
        summary_report_obj = jb_report_utils.construct_report(report.fqcn, report.kwargs)

        assert summary_report_obj.report_path == Path("test") / "file.md"
        assert summary_report_obj.csv_path == Path("test") / "file.csv"

# TODO: A test to exercise construction of the Summary report tables.
# TODO: A test to confirm loading reports in an experiment config.
# TODO: A test to confirm loading reports in model config.