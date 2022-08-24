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

import pytest
import torch

import juneberry.metrics.metrics_manager as mm
from juneberry.config.model import Plugin


test_data_dir = Path(__file__).resolve().parent / "data"

config_tm_filename = test_data_dir / "config_classification_torchmetrics.json"
config_nn_filename = test_data_dir / "config_classification_torchnn.json"
config_sk_filename = test_data_dir / "config_classification_sklearn.json"
output_data_filename = test_data_dir / "output.pt"
local_labels_data_filename = test_data_dir / "local_labels.pt"

with open(config_tm_filename, 'r') as f:
    config_tm_data = json.load(f)

with open(config_sk_filename, 'r') as f:
    config_sk_data = json.load(f)

with open(config_nn_filename, 'r') as f:
    config_nn_data = json.load(f)

evaluation_tm_metrics: List[Plugin] = []
for cd in config_tm_data["metrics"]:
    evaluation_tm_metrics.append(Plugin.from_dict(cd))

evaluation_sk_metrics: List[Plugin] = []
for cd in config_sk_data["metrics"]:
    evaluation_sk_metrics.append(Plugin.from_dict(cd))

evaluation_nn_metrics: List[Plugin] = []
for cd in config_nn_data["metrics"]:
    evaluation_nn_metrics.append(Plugin.from_dict(cd))

output = torch.load(output_data_filename)
local_labels = torch.load(local_labels_data_filename)

metrics_tm_mgr = mm.MetricsManager(evaluation_tm_metrics)
metrics_sk_mgr = mm.MetricsManager(evaluation_sk_metrics)
metrics_nn_mgr = mm.MetricsManager(evaluation_nn_metrics)

tm_metrics = metrics_tm_mgr(local_labels, output)
sk_metrics = metrics_sk_mgr(local_labels, output)
nn_metrics = metrics_nn_mgr(local_labels, output)

def approx(expected_val):
    return pytest.approx(expected_val, abs=5e-3)

def test_torchmetrics():
    assert tm_metrics["juneberry.metrics.torchmetrics.metrics.Metrics"]["torchmetrics.functional.accuracy"] == approx(0.6)

def test_torchnn():
    # not sure how to create correct tensor (tensor(2.1329, grad_fn=<NllLossBackward>) for testing
    # can compare data?
    assert nn_metrics["juneberry.metrics.torchnn.metrics.Metrics"]["torch.nn.CrossEntropyLoss"] == torch.tensor(2.1329)
    # assert nn_metrics["juneberry.metrics.torchnn.metrics.Metrics"]["torch.nn.CrossEntropyLoss"].data == torch.tensor(2.1329)
    # assert torch.equal(nn_metrics["juneberry.metrics.torchnn.metrics.Metrics"]["torch.nn.CrossEntropyLoss"].data, torch.tensor(2.1329))

def test_sklearn_metrics():
    assert sk_metrics["juneberry.metrics.sklearn.metrics.Metrics"]["sklearn.metrics.accuracy_score"] == approx(10)

