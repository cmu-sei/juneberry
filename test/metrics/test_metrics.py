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

import juneberry.metrics.metrics as metrics


test_data_dir = Path(__file__).resolve().parent / "data"

ground_truth_filename = test_data_dir / "ground_truth.json"
detections_filename = test_data_dir / "detections.json"

with open(ground_truth_filename, 'r') as f:
    gt_data = json.load(f)

with open(detections_filename, 'r') as f:
    det_data = json.load(f)

m = metrics.Metrics(ground_truth_filename,
                    detections_filename,
                    "test_metrics_model_name",
                    "test_metrics_det_name")


def _pytest_assert_frame_equal(frame1, frame2):
    try:
        assert_frame_equal(frame1, frame2)
        assert True
    except AssertionError:
        assert False


def test_create_with_data():
    m_with_data = metrics.Metrics.create_with_data(gt_data, det_data)
    assert m_with_data.mAP == m.mAP


def test_mAP():
    assert m.mAP == 0.23531353135313532


def test_mAP_50():
    assert m.mAP_50 == 0.372937293729373


def test_mAP_75():
    assert m.mAP_75 == 0.2599009900990099


def test_mAP_small():
    assert m.mAP_small == 0.2495049504950495


def test_mAP_medium():
    assert m.mAP_medium == 0.3226072607260726


def test_mAP_large():
    assert m.mAP_large == 0.100990099009901


def test_mAP_per_class():
    assert m.mAP_per_class["class_1"] == 0.37013201320132016
    assert m.mAP_per_class["class_2"] == 0.10049504950495049


def test_pr():
    pr = pd.read_csv(test_data_dir / "pr.csv")
    _pytest_assert_frame_equal(pr, m.pr)


def test_pc():
    pc = pd.read_csv(test_data_dir / "pc.csv")
    _pytest_assert_frame_equal(pc, m.pc)


def test_fscore():
    fscore = pd.read_csv(test_data_dir / "fscore.csv")
    _pytest_assert_frame_equal(fscore, m.fscore)


def test_ap():
    assert m.ap == 0.2594444444444444


def test_pr_auc():
    assert m.pr_auc == 0.2461111111111111


def test_pc_auc():
    assert m.pc_auc == 0.4143311403508772


def test_rc_auc():
    assert m.rc_auc == 0.3533333333333334
