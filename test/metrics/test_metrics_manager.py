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

import juneberry.metrics.metrics_manager as mm


test_data_dir = Path(__file__).resolve().parent / "data"

ground_truth_filename = test_data_dir / "ground_truth.json"
ground_truth_no_annos_filename = test_data_dir / "ground_truth_no_annos.json"
detections_filename = test_data_dir / "detections.json"
config_filename = test_data_dir / "config.json"

with open(ground_truth_filename, 'r') as f:
    gt_data = json.load(f)

with open(detections_filename, 'r') as f:
    det_data = json.load(f)

with open(config_filename, 'r') as f:
    config_data = json.load(f)

result = mm.MetricsManager(config_data)
metrics = result(gt_data, det_data)

def test_coco_metrics():
    expected_result = {
        'mAP_50': 0.3737623762376238,
        'mAP_75': 0.2599009900990099,
        'mAP_coco': 0.23648514851485147,
        'mAP_small': 0.2498019801980198,
        'mAP_medium': 0.3226072607260726,
        'mAP_large': 0.100990099009901,
        'AP_50_class_2': 0.14356435643564355,
        'AP_50_class_1': 0.603960396039604,
        'AP_75_class_2': 0.14356435643564355,
        'AP_75_class_1': 0.37623762376237624
    }
    assert metrics["juneberry.metrics.metrics.CocoMetrics"] == expected_result


def test_tide_metrics():
    expected_result = {
        'mAP': 0.3737623762376238,
        'mdAP_localisation': 0.0,
        'mdAP_classification': 0.0,
        'mdAP_both': 0.0,
        'mdAP_duplicate': 0.0,
        'mdAP_background': 0.08168316831683163,
        'mdAP_missed': 0.3567106710671066,
        'mdAP_fp': 0.08168316831683163,
        'mdAP_fn': 0.3567106710671066
    }
    assert metrics["juneberry.metrics.metrics.TideMetrics"] == expected_result


def test_common_metrics():
    pass
