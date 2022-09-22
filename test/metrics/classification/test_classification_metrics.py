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

import juneberry.metrics.classification.metrics_manager as mm
from juneberry.config.model import Plugin


test_data_dir = Path(__file__).resolve().parent / "data"

config_filename = test_data_dir / "config_classification.json"
output_data_filename = test_data_dir / "output.pt"
local_labels_data_filename = test_data_dir / "local_labels.pt"

with open(config_filename, 'r') as f:
    config_data = json.load(f)

training_metrics: List[Plugin] = []
for cd in config_data["metrics"]:
    training_metrics.append(Plugin.from_dict(cd))

output = torch.load(output_data_filename)
local_labels = torch.load(local_labels_data_filename)

# lifted from pytorch.evaluation.utils.compute_accuracy
with torch.set_grad_enabled(False):
    output_np = output.cpu().numpy()
    local_labels_np = local_labels.cpu().detach().numpy()

metrics_mgr = mm.MetricsManager(training_metrics)

metrics = metrics_mgr(local_labels_np, output_np, binary = False)

def approx(expected_val):
    return pytest.approx(expected_val, abs=5e-3)

def test_torchmetrics_functional():
    assert metrics["func_accuracy"] == approx(0.6)

def test_torchmetrics_classbased():
    assert metrics["obj_accuracy"] == approx(0.6)

def test_torchnn():
    assert metrics["loss"] == approx(2.1329)

def test_sklearn_metrics():
    assert metrics["accuracy_score"] == approx(10)

def test_tensorflow_classbased():
    assert metrics["tf_accuracy"] == 0.5

def test_tensorflow_functional():
    assert metrics["tf_binary_accuracy"] == approx(0.1)
