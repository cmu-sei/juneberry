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
from unittest import TestCase

from pathlib import Path
import pytest

from juneberry.config.experiment import ExperimentConfig
import utils


def make_basic_config():
    return {
        "description": "simple description",
        "models": [
            {
                "name": "tabular_binary_sample",
                "tests": [
                    {
                        "tag": "pyt50",
                        "dataset_path": "data_sets/train_data_config.json",
                        "classify": 3
                    }
                ]
            }
        ],
        "reports": [
            {
                "description": "basic description",
                "fqcn": 'juneberry.reporting.roc.ROCPlot',
                "kwargs": {
                    "output_filename": "sample_roc_1.png",
                    "plot_title": "Sample ROC Plot"
                },
                "tests": [
                    {
                        "tag": "pyt50",
                        "classes": "0"
                    }
                ],
            }
        ],
        "tuning": [
            {
                "model": "",
                "tuning_config": ""
            }

        ],
        "format_version": "1.5.0"
    }
    # NOTE: We provide the formatVersion manually to force an update of the unit test when
    # the version changes.


# TODO: Refactor for maintenance

def test_basic_construction(tmp_path):
    with utils.set_directory(tmp_path):
        experiment_config = make_basic_config()
        dc_loc = Path(experiment_config['models'][0]['tests'][0]['dataset_path'])
        dc_loc.parent.mkdir(parents=True)
        dc_loc.touch()

        model_dir = Path.cwd() / "models" / experiment_config['models'][0]['name']
        model_dir.mkdir(parents=True)

        # Most of the real functionality is in the checks
        ec = ExperimentConfig.construct(data=experiment_config)
        assert len(experiment_config['models']) == len(ec['models'])
        assert len(experiment_config['reports']) == len(ec['reports'])
        assert len(experiment_config['tuning']) == len(ec['tuning'])


class TestExperimentModels(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply to the "models" section of
    an experiment config.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, tmp_path, caplog):
        """
        The purpose of this function is to make the pytest tmp_path fixture available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path
        self.caplog = caplog

    def test_model_bad_name(self):
        config = make_basic_config()
        config['models'][0]['name'] = "bad name"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Model not found" in self.caplog.text

    def test_model_duplicate_tag(self):
        config = make_basic_config()
        config['models'][0]['tests'].append({
            "tag": "pyt50",
            "dataset_path": "data_sets/imagenette_unit_test.json",
        })

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Found duplicate tag" in self.caplog.text

    def test_model_duplicate_tag_2(self):
        config = make_basic_config()
        config['models'].append({
            "name": "tabular_binary_sample",
            "tests": [
                {
                    "tag": "pyt50",
                    "dataset_path": "data_sets/imagenette_unit_test.json",
                }
            ]
        })

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Found duplicate tag" in self.caplog.text

    def test_model_bad_dataset_path(self):
        config = make_basic_config()
        config['models'][0]['tests'][0]['dataset_path'] = "bad name"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Dataset not found" in self.caplog.text


class TestExperimentReports(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply to the "reports" section of
    an experiment config.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, tmp_path, caplog):
        """
        The purpose of this function is to make the pytest tmp_path fixture available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path
        self.caplog = caplog

    def test_report_bad_tag(self):
        config = make_basic_config()
        config['reports'][0]['tests'][0]['tag'] = "wrong tag"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Unknown report tag" in self.caplog.text


class TestExperimentTuning(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply to the "tuning" section of
    an experiment config.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, tmp_path, caplog):
        """
        The purpose of this function is to make the pytest tmp_path fixture available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path
        self.caplog = caplog

    def test_tuning_bad_model(self):
        config = make_basic_config()
        config['tuning'][0]['model'] = "unknown_model"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Model not found: " in self.caplog.text

    def test_tuning_bad_tuning_config(self):
        config = make_basic_config()
        config['tuning'][0]['tuning_config'] = "unknown_tuning_config.json"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Tuning config not found: " in self.caplog.text
