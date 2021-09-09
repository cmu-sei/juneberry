#! /usr/bin/env bash

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn 
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow 
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
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
