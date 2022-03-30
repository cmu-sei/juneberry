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
from juneberry.config.model import Metrics


test_data_dir = Path(__file__).resolve().parent / "data"

ground_truth_filename = test_data_dir / "ground_truth.json"
ground_truth_no_annos_filename = test_data_dir / "ground_truth_no_annos.json"
detections_filename = test_data_dir / "detections.json"
config_filename = test_data_dir / "config.json"

with open(ground_truth_filename, 'r') as f:
    gt_data = json.load(f)

with open(ground_truth_no_annos_filename, 'r') as f:
    gt_no_annos_data = json.load(f)

with open(detections_filename, 'r') as f:
    det_data = json.load(f)

with open(config_filename, 'r') as f:
    config_data = json.load(f)

evaluation_metrics: List[Metrics] = []
for cd in config_data["evaluation_metrics"]:
    evaluation_metrics.append(Metrics.from_dict(cd))

metrics_mgr = mm.MetricsManager(evaluation_metrics)
metrics = metrics_mgr(gt_data, det_data)

coco_metrics = metrics["juneberry.metrics.metrics.Coco"]
tide_metrics = metrics["juneberry.metrics.metrics.Tide"]
stats_metrics = metrics["juneberry.metrics.metrics.Stats"]


def approx(expected_val):
    return pytest.approx(expected_val, abs=5e-3)


def _pytest_assert_frame_equal(frame1, frame2):
    try:
        assert_frame_equal(frame1, frame2)
        assert True
    except AssertionError:
        assert False


def test_formatted_coco_metrics():
    expected_result = {
        "bbox": {
            "mAP": 0.236,
            "mAP_50": 0.374,
            "mAP_75": 0.260,
            "mAP_s": 0.250,
            "mAP_m": 0.323,
            "mAP_l": 0.101,
        },
        "bbox_per_class": {
            'mAP_class_1': 0.376,
            'mAP_class_2': 0.144,
        }
    }
    # We need two comparisons because pytest.approx doesn't support nested dictionaries
    assert coco_metrics["bbox"] == approx(expected_result["bbox"])
    assert coco_metrics["bbox_per_class"] == approx(expected_result["bbox_per_class"])


def test_tide_metrics():
    expected_result = {
        'mAP': 0.374,
        'mdAP_localisation': 0.0,
        'mdAP_classification': 0.0,
        'mdAP_both': 0.0,
        'mdAP_duplicate': 0.0,
        'mdAP_background': 0.082,
        'mdAP_missed': 0.357,
        'mdAP_fp': 0.082,
        'mdAP_fn': 0.357,
    }
    assert tide_metrics == approx(expected_result)


def test_create_with_empty_annos():
    with pytest.raises(ValueError):
        _ = metrics_mgr(gt_no_annos_data, det_data)


def test_prc_df():
    expected_result = pd.read_csv(test_data_dir / "prc.csv")
    actual_result = stats_metrics["prc_df"]
    _pytest_assert_frame_equal(expected_result, actual_result)


def test_ap():
    assert stats_metrics["ap"] == approx(0.259)


def test_pr_auc():
    assert stats_metrics["pr_auc"] == approx(0.246)


def test_pc_auc():
    assert stats_metrics["pc_auc"] == approx(0.414)


def test_rc_auc():
    assert stats_metrics["rc_auc"] == approx(0.353)


def _test_prediction_types(tp: int, fp: int, fn: int):
    assert stats_metrics["prediction_types"]["tp"] == tp
    assert stats_metrics["prediction_types"]["fp"] == fp
    assert stats_metrics["prediction_types"]["fn"] == fn


def test_prediction_types():
    _test_prediction_types(4, 15, 11)
