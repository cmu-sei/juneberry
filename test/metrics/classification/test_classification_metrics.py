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

import numpy
import pytest
import torch

import juneberry.metrics.classification.metrics_manager as mm
from juneberry.config.model import Plugin

test_data_dir = Path(__file__).resolve().parent / "data"

config_filename = test_data_dir / "config_classification.json"

with open(config_filename, 'r') as f:
    config_data = json.load(f)

metrics_plugins: List[Plugin] = []
for cd in config_data["metrics"]:
    metrics_plugins.append(Plugin.from_dict(cd))

target = torch.tensor([0, 1, 2])
preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])

# metrics plugins take numpy inputs
with torch.set_grad_enabled(False):
    preds_np = preds.cpu().numpy()
    target_np = target.cpu().detach().numpy()

metrics_mgr = mm.MetricsManager(metrics_plugins)
metrics = metrics_mgr(target_np, preds_np, binary = False)

def approx(expected_val):
    return pytest.approx(expected_val, abs=5e-3)

def test_torchmetrics_functional():
    assert numpy.equal(metrics["func_accuracy"], numpy.array(0.6666667, dtype=numpy.float32))

def test_torchmetrics_classbased():
    assert numpy.equal(metrics["obj_accuracy"], numpy.array(0.6666667, dtype=numpy.float32))

def test_torchnn():
    assert numpy.equal(metrics["loss"], numpy.array(1.3038288, dtype=numpy.float32))

def test_sklearn_metrics():
    assert metrics["accuracy_score"] == 0

def test_tensorflow_classbased():
    assert metrics["tf_accuracy"] == 3.0

def test_tensorflow_functional():
    assert metrics["tf_binary_accuracy"] == approx(0.33333334)
