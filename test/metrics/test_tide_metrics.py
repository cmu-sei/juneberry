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

tide_metrics = metrics.Metrics.create_with_data(gt_data,
                                                det_data,
                                                "test_metrics_model_name",
                                                "test_metrics_det_name",
                                                toolkit=metrics.MetricsToolkit.TIDE)


def test_AP():
    assert tide_metrics.AP is None


def test_mAP():
    assert tide_metrics.mAP == 0.372937293729373


def test_dAP_localisation():
    assert tide_metrics.dAP_localisation is None


def test_mdAP_localization():
    assert tide_metrics.mdAP_localization is None


def test_dAP_classification():
    assert tide_metrics.dAP_classification is None


def test_mdAP_classification():
    assert tide_metrics.mdAP_classification is None


def test_dAP_both():
    assert tide_metrics.dAP_both is None


def test_mdAP_both():
    assert tide_metrics.mdAP_both is None


def test_dAP_duplicate():
    assert tide_metrics.dAP_duplicate is None


def test_mdAP_duplicate():
    assert tide_metrics.mdAP_duplicate is None


def test_dAP_background():
    assert tide_metrics.dAP_background is None


def test_mdAP_background():
    assert tide_metrics.mdAP_background is None


def test_dAP_missed():
    assert tide_metrics.dAP_missed is None


def test_mdAP_missed():
    assert tide_metrics.mdAP_missed is None


def test_dAP_fp():
    assert tide_metrics.dAP_fp is None


def test_mdAP_fp():
    assert tide_metrics.mdAP_fp is None


def test_dAP_fn():
    assert tide_metrics.dAP_fn is None


def test_mdAP_fn():
    assert tide_metrics.mdAP_fn is None
