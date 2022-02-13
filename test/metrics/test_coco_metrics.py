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

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

import juneberry.metrics.metrics as metrics


test_data_dir = Path(__file__).resolve().parent / "data"

ground_truth_filename = test_data_dir / "ground_truth.json"
ground_truth_no_annos_filename = test_data_dir / "ground_truth_no_annos.json"
detections_filename = test_data_dir / "detections.json"

with open(ground_truth_filename, 'r') as f:
    gt_data = json.load(f)

with open(detections_filename, 'r') as f:
    det_data = json.load(f)

coco_metrics = metrics.Metrics.create_with_data(gt_data,
                                                det_data,
                                                "test_metrics_model_name",
                                                "test_metrics_det_name",
                                                toolkit=metrics.MetricsToolkit.COCO)


def _pytest_assert_frame_equal(frame1, frame2):
    try:
        assert_frame_equal(frame1, frame2)
        assert True
    except AssertionError:
        assert False


def test_mAP():
    assert coco_metrics.mAP == 0.23531353135313532


def test_mAP_50():
    assert coco_metrics.mAP_50 == 0.372937293729373


def test_mAP_75():
    assert coco_metrics.mAP_75 == 0.2599009900990099


def test_mAP_small():
    assert coco_metrics.mAP_small == 0.2495049504950495


def test_mAP_medium():
    assert coco_metrics.mAP_medium == 0.3226072607260726


def test_mAP_large():
    assert coco_metrics.mAP_large == 0.100990099009901


def test_mAP_per_class():
    assert coco_metrics.mAP_per_class["class_1"] == 0.37013201320132016
    assert coco_metrics.mAP_per_class["class_2"] == 0.10049504950495049


def test_prc_df():
    prc_df = pd.read_csv(test_data_dir / "prc.csv")
    _pytest_assert_frame_equal(prc_df, coco_metrics._prc_df)


def test_fscore():
    fscore_df = pd.read_csv(test_data_dir / "fscore.csv")
    _pytest_assert_frame_equal(fscore_df, coco_metrics._fscore_df)


def test_ap():
    assert coco_metrics.ap == 0.2594444444444444


def test_pr_auc():
    assert coco_metrics.pr_auc == 0.2461111111111111


def test_pc_auc():
    assert coco_metrics.pc_auc == 0.4143311403508772


def test_rc_auc():
    assert coco_metrics.rc_auc == 0.3533333333333334


def test_as_dict():
    assert coco_metrics.as_dict() == {
        "mAP": coco_metrics.mAP,
        "mAP_50": coco_metrics.mAP_50,
        "mAP_75": coco_metrics.mAP_75,
        "mAP_s": coco_metrics.mAP_small,
        "mAP_m": coco_metrics.mAP_medium,
        "mAP_l": coco_metrics.mAP_large,
    }


def _test_prediction_types(tp_threshold: float, tp: int, fp: int, fn: int):
    assert coco_metrics.prediction_types(tp_threshold)["tp"] == tp
    assert coco_metrics.prediction_types(tp_threshold)["fp"] == fp
    assert coco_metrics.prediction_types(tp_threshold)["fn"] == fn


def test_prediction_types_high_threshold():
    _test_prediction_types(0.8, 4, 15, 11)


def test_prediction_types_low_threshold():
    _test_prediction_types(0.1, 9, 10, 6)
