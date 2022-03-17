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

import juneberry.metrics.common_metrics as metrics


test_data_dir = Path(__file__).resolve().parent / "data"

ground_truth_filename = test_data_dir / "ground_truth.json"
ground_truth_no_annos_filename = test_data_dir / "ground_truth_no_annos.json"
detections_filename = test_data_dir / "detections.json"

with open(ground_truth_filename, 'r') as f:
    gt_data = json.load(f)

with open(detections_filename, 'r') as f:
    det_data = json.load(f)

m = metrics.CommonMetrics(ground_truth_filename,
                          detections_filename,
                    "test_metrics_model_name",
                    "test_metrics_det_name")

def approx(val: float):
    return pytest.approx(val, abs=2.5e-3)


def _pytest_assert_frame_equal(frame1, frame2):
    try:
        assert_frame_equal(frame1, frame2)
        assert True
    except AssertionError:
        assert False


def test_create_with_empty_annos():
    with pytest.raises(ValueError):
        _ = metrics.CommonMetrics(ground_truth_no_annos_filename,
                                  detections_filename,
                            "test_create_with_empty_annos_model_name",
                            "test_create_with_empty_annos_det_name")


def test_create_with_data():
    m_with_data = metrics.CommonMetrics.create_with_data(gt_data, det_data)
    assert m_with_data.ap == m.ap


def test_prc_df():
    prc_df = pd.read_csv(test_data_dir / "prc.csv")
    _pytest_assert_frame_equal(prc_df, m.prc_df)


def test_fscore():
    fscore_df = pd.read_csv(test_data_dir / "fscore.csv")
    _pytest_assert_frame_equal(fscore_df, m._fscore_df)


def test_ap():
    assert m.ap == approx(0.259)


def test_pr_auc():
    assert m.pr_auc == approx(0.246)


def test_pc_auc():
    assert m.pc_auc == approx(0.414)


def test_rc_auc():
    assert m.rc_auc == approx(0.353)


def _test_prediction_types(tp_threshold: float, tp: int, fp: int, fn: int):
    assert m.prediction_types(tp_threshold)["tp"] == tp
    assert m.prediction_types(tp_threshold)["fp"] == fp
    assert m.prediction_types(tp_threshold)["fn"] == fn


def test_prediction_types_high_threshold():
    _test_prediction_types(0.8, 4, 15, 11)


def test_prediction_types_low_threshold():
    _test_prediction_types(0.1, 9, 10, 6)
