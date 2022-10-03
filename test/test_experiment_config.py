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
from unittest import TestCase

from pathlib import Path
import pytest

from juneberry.config.experiment import ExperimentConfig
import utils


class TestExperimentConfig(TestCase):
    """
    The purpose of this class is to organize any unit tests that apply to the "models" section of
    an experiment config.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, tmp_path, caplog):
        """
        The purpose of this function is to make the pytest tmp_path and caplog fixtures available
        inside of the unittest.TestCase.
        """
        self.tmp_path = tmp_path
        self.caplog = caplog

    def test_basic_construction(self):
        with utils.set_directory(self.tmp_path):
            experiment_config = utils.make_basic_experiment_config()
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

    def test_model_bad_name(self):
        config = utils.make_basic_experiment_config()
        config['models'][0]['name'] = "bad name"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Model not found" in self.caplog.text

    def test_model_duplicate_tag(self):
        config = utils.make_basic_experiment_config()
        config['models'][0]['tests'].append({
            "tag": "pyt50",
            "dataset_path": "data_sets/imagenette_unit_test.json",
        })

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Found duplicate tag" in self.caplog.text

    def test_model_duplicate_tag_2(self):
        config = utils.make_basic_experiment_config()
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
        config = utils.make_basic_experiment_config()
        config['models'][0]['tests'][0]['dataset_path'] = "bad name"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Dataset not found" in self.caplog.text

    def test_report_bad_tag(self):
        config = utils.make_basic_experiment_config()
        config['reports'][0]['tests'][0]['tag'] = "wrong tag"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Unknown report tag" in self.caplog.text

    def test_tuning_bad_model(self):
        config = utils.make_basic_experiment_config()
        config['tuning'][0]['model'] = "unknown_model"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Model not found: " in self.caplog.text

    def test_tuning_bad_tuning_config(self):
        config = utils.make_basic_experiment_config()
        config['tuning'][0]['tuning_config'] = "unknown_tuning_config.json"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Tuning config not found: " in self.caplog.text
