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

import json
from pathlib import Path
from typing import List

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import juneberry.metrics.metrics_manager as mm
from juneberry.config.model import Plugin


test_data_dir = Path(__file__).resolve().parent / "data"

ground_truth_filename = test_data_dir / "ground_truth.json"
ground_truth_no_annos_filename = test_data_dir / "ground_truth_no_annos.json"
detections_filename = test_data_dir / "detections.json"
config_filename = test_data_dir / "config.json"
default_metrics_config_filename = test_data_dir / "default_metrics_with_formatter.json"

with open(ground_truth_filename, 'r') as f:
    gt_data = json.load(f)

with open(ground_truth_no_annos_filename, 'r') as f:
    gt_no_annos_data = json.load(f)

with open(detections_filename, 'r') as f:
    det_data = json.load(f)

with open(config_filename, 'r') as f:
    config_data = json.load(f)

with open(default_metrics_config_filename, 'r') as f:
    default_metrics_config_data = json.load(f)

evaluation_metrics_all: List[Plugin] = []
for cd in config_data["evaluation_metrics"]:
    evaluation_metrics_all.append(Plugin.from_dict(cd))

evaluation_metrics_default: List[Plugin] = []
for cd in default_metrics_config_data["evaluation_metrics"]:
    evaluation_metrics_default.append(Plugin.from_dict(cd))
evaluation_metrics_default_formatter = Plugin.from_dict(default_metrics_config_data["evaluation_metrics_formatter"])

metrics_mgr_all = mm.MetricsManager(evaluation_metrics_all)
metrics_all = metrics_mgr_all(gt_data, det_data)

metrics_mgr_default_formatted = mm.MetricsManager(evaluation_metrics_default, evaluation_metrics_default_formatter)
metrics_default_formatted = metrics_mgr_default_formatted(gt_data, det_data)

coco_metrics = metrics_all["juneberry.metrics.metrics.Coco"]
tide_metrics = metrics_all["juneberry.metrics.metrics.Tide"]
summary_metrics = metrics_all["juneberry.metrics.metrics.Summary"]

expected_coco_metrics = {
    "mAP_coco": 0.236,
    "mAP_50": 0.374,
    "mAP_75": 0.260,
    "mAP_small": 0.250,
    "mAP_medium": 0.323,
    "mAP_large": 0.101,
    "class_1": 0.376,
    "class_2": 0.144,
}

expected_tide_metrics = {
    "mAP": 0.374,
    "mdAP_localisation": 0.0,
    "mdAP_classification": 0.0,
    "mdAP_both": 0.0,
    "mdAP_duplicate": 0.0,
    "mdAP_background": 0.082,
    "mdAP_missed": 0.357,
    "mdAP_fp": 0.082,
    "mdAP_fn": 0.357,
}

expected_default_formatted_metrics = {
    "bbox": {
        "mAP": 0.236,
        "mAP_50": 0.374,
        "mAP_75": 0.260,
        "mAP_s": 0.250,
        "mAP_m": 0.323,
        "mAP_l": 0.101,
        "mdAP_localisation": 0.0,
        "mdAP_classification": 0.0,
        "mdAP_both": 0.0,
        "mdAP_duplicate": 0.0,
        "mdAP_background": 0.082,
        "mdAP_missed": 0.357,
        "mdAP_fp": 0.082,
        "mdAP_fn": 0.357,
    },
    "bbox_per_class": {
        "mAP_class_1": 0.376,
        "mAP_class_2": 0.144,
    },
    "summary": {
        "max_r": 0.467,
        "pr_auc": 0.246,
        "pc_auc": 0.414,
        "rc_auc": 0.353,
        "ap": 0.259,
        "tp": 4,
        "fp": 15,
        "fn": 11,
    }
}


def approx(expected_val):
    return pytest.approx(expected_val, abs=5e-3)


def _pytest_assert_frame_equal(frame1, frame2):
    try:
        assert_frame_equal(frame1, frame2)
        assert True
    except AssertionError:
        assert False


def test_formatted_default_metrics():
    # We need separate comparisons because pytest.approx doesn't support nested dictionaries
    assert metrics_default_formatted["bbox"] == approx(expected_default_formatted_metrics["bbox"])
    assert metrics_default_formatted["bbox_per_class"] == approx(expected_default_formatted_metrics["bbox_per_class"])
    assert metrics_default_formatted["summary"] == approx(expected_default_formatted_metrics["summary"])


def test_coco_metrics():
    assert coco_metrics == approx(expected_coco_metrics)


def test_tide_metrics():
    assert tide_metrics == approx(expected_tide_metrics)


def test_create_with_empty_annos():
    assert(metrics_mgr_all(gt_no_annos_data, det_data)) == {}


def test_prc_df():
    expected_result = pd.read_csv(test_data_dir / "prc.csv")
    actual_result = summary_metrics["prc_df"]
    _pytest_assert_frame_equal(expected_result, actual_result)
