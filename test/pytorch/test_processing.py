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

import torch
import torch.distributed as dist

import juneberry.pytorch.processing as gpu_setup


class CaptureAllArgs:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self


class MockModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device


def test_determine_gpus():
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1

    assert gpu_setup.determine_gpus() == 1

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 5

    assert gpu_setup.determine_gpus() == 0


def test_setup_cuda_device():
    capture = CaptureAllArgs()

    torch.cuda.set_device = capture

    device = gpu_setup.setup_cuda_device(0, 0)
    assert device.type == 'cpu'

    device = gpu_setup.setup_cuda_device(5, 0)
    assert device.type == 'cuda'
    assert device.index == 0
    assert capture.args[0] == device

    device = gpu_setup.setup_cuda_device(5, 2)
    assert device.type == 'cuda'
    assert device.index == 2
    assert capture.args[0] == device


def test_setup_distributed():
    capture = CaptureAllArgs()
    dist.init_process_group = capture
    gpu_setup.setup_distributed(5, 2)
    assert capture.kwargs['world_size'] == 5
    assert capture.kwargs['rank'] == 2


def test_prepare_model():
    # The functions just return a dict of what was passed
    capture_dp = CaptureAllArgs()
    torch.nn.DataParallel = capture_dp
    capture_ddp = CaptureAllArgs()
    torch.nn.parallel.DistributedDataParallel = capture_ddp

    model = MockModel()
    device = "Device"
    # For CPU Mode we just get the model back
    result = gpu_setup.prepare_model(False, 0, 0, model, device)
    assert result == model

    # WARNING THIS IS FRAGILE
    result = gpu_setup.prepare_model(False, 1, 0, model, device)
    assert result == capture_dp
    assert result.args[0] == model
    assert result.kwargs.get('device_ids', None) is None
    assert model.device == device

    result = gpu_setup.prepare_model(True, 4, 2, model, device)
    assert result == capture_ddp
    assert result.args[0] == model
    assert result.kwargs['device_ids'] == [2]
    assert model.device == device
