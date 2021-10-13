#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ======================================================================================================================

import json
import juneberry.metrics.metrics as metrics
import pandas as pd
from pandas.util.testing import assert_frame_equal
from pathlib import Path

det_data = None  # detections data
gt_data = None  # ground truth data
m = None  # Metrics object


def _pytest_assert_frame_equal(frame1, frame2):
    try:
        assert_frame_equal(frame1, frame2)
        assert True
    except AssertionError:
        assert False


def setup_module():
    global det_data, gt_data, m

    gt_data = None
    det_data = None

    ground_truth_filename = "data/ground_truth.json"
    detections_filename = "data/detections.json"

    with open(ground_truth_filename, 'r') as f:
        gt_data = json.load(f)

    with open(detections_filename, 'r') as f:
        det_data = json.load(f)

    m = metrics.Metrics(Path(ground_truth_filename),
                        Path(detections_filename),
                        "test_metrics_model_name",
                        "test_metrics_det_name")


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
    pr = pd.read_csv("data/pr.csv")
    _pytest_assert_frame_equal(pr, m.pr)


def test_pc():
    pc = pd.read_csv("data/pc.csv")
    _pytest_assert_frame_equal(pc, m.pc)


def test_fscore():
    fscore = pd.read_csv("data/fscore.csv")
    _pytest_assert_frame_equal(fscore, m.fscore)


def test_ap():
    assert m.ap == 0.2594444444444444


def test_pr_auc():
    assert m.pr_auc == 0.2461111111111111


def test_pc_auc():
    assert m.pc_auc == 0.4143311403508772


def test_rc_auc():
    assert m.rc_auc == 0.3533333333333334