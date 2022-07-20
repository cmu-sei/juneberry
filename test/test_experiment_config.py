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
        "format_version": "1.5.0"
    }
    # NOTE: We provide the formatVersion manually to force an update of the unit test when
    # the version changes.


# TODO: Refactor for maintenance

def test_config_basics(tmp_path):
    with utils.set_directory(tmp_path):
        # TODO: Just creating the files in a fake workspace is a little heavy-handed. We need to have a better approach.
        utils.setup_test_workspace(tmp_path)
        utils.make_tabular_workspace(tmp_path)

        config = make_basic_config()

        # Most of the real functionality is in the checks
        exp_conf = ExperimentConfig.construct(config)
        assert len(config['models']) == len(exp_conf['models'])
        assert len(config['reports']) == len(exp_conf['reports'])


def test_model_bad_name(tmp_path, caplog):
    with utils.set_directory(tmp_path):
        # TODO: Just creating the files in a fake workspace is a little heavy-handed. We need to have a better approach.
        utils.setup_test_workspace(tmp_path)
        utils.make_tabular_workspace(tmp_path)

        config = make_basic_config()
        config['models'][0]['name'] = "bad name"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Model not found" in caplog.text


def test_model_duplicate_tag(tmp_path, caplog):
    with utils.set_directory(tmp_path):
        # TODO: Just creating the files in a fake workspace is a little heavy-handed. We need to have a better approach.
        utils.setup_test_workspace(tmp_path)
        utils.make_tabular_workspace(tmp_path)

        config = make_basic_config()
        config['models'][0]['tests'].append({
            "tag": "pyt50",
            "dataset_path": "data_sets/imagenette_unit_test.json",
        })

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Found duplicate tag" in caplog.text


def test_model_duplicate_tag_2(tmp_path, caplog):
    with utils.set_directory(tmp_path):
        # TODO: Just creating the files in a fake workspace is a little heavy-handed. We need to have a better approach.
        utils.setup_test_workspace(tmp_path)
        utils.make_tabular_workspace(tmp_path)

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

        assert "Found duplicate tag" in caplog.text


def test_model_bad_dataset_path(tmp_path, caplog):
    with utils.set_directory(tmp_path):
        # TODO: Just creating the files in a fake workspace is a little heavy-handed. We need to have a better approach.
        utils.setup_test_workspace(tmp_path)
        utils.make_tabular_workspace(tmp_path)

        config = make_basic_config()
        config['models'][0]['tests'][0]['dataset_path'] = "bad name"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Dataset not found" in caplog.text


def test_report_bad_tag(tmp_path, caplog):
    with utils.set_directory(tmp_path):
        # TODO: Just creating the files in a fake workspace is a little heavy-handed. We need to have a better approach.
        utils.setup_test_workspace(tmp_path)
        utils.make_tabular_workspace(tmp_path)

        config = make_basic_config()
        config['reports'][0]['tests'][0]['tag'] = "wrong tag"

        with pytest.raises(SystemExit) as exc_info:
            ExperimentConfig.construct(config)

        assert "Unknown report tag" in caplog.text
