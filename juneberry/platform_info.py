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

import importlib
import json
import os
import subprocess
import sys

from prodict import List, Prodict


class CUDA(Prodict):
    device_count: int
    is_available: bool
    nvidia_smi_version: str


class Report(Prodict):
    cuda: CUDA
    environment: Prodict
    gpu_list: List[str]
    pip_freeze: List[str]
    platform: str
    python_path: List[str]
    python_version: str


def locate_nvidia_smi():
    result = subprocess.run(['which', 'nvidia-smi'], capture_output=True, text=True)
    smi_path = "Not found"
    try:
        if result.returncode == 0:
            smi_path = result.stdout.strip()
    except FileNotFoundError:
        pass

    return smi_path


def get_cuda_devices():
    """:return: a list of devices using nvidia-smi """
    # Run "nvidia-smi -L" to get the list of GPUs. If the command fails we consider that 0.
    # Count the lines in the list for the number of GPUs. Each line comes with the following:
    # GPU 0: Tesla V100-PCIE-16GB (UUID: GPU-4b0a37d8-225a-b487-0ceb-a64003cea4c0)
    gpu_list = []
    try:
        smi_output = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        gpu_list = smi_output.stdout.rstrip().split('\n')
    except FileNotFoundError:
        pass

    return gpu_list


def capture_pip_freeze():
    output = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    return output.stdout.split('\n')


def capture_versions():
    versions = {}
#    for pkg in ['juneberry', 'detectron2', 'mmcv', 'mmdet', 'numpy', 'onnx', 'onnxruntime', 'onnxruntime-gpu',
#                'torch', 'torchvision', 'tensorflow', 'tensorrt']:
    for pkg in ['juneberry', 'detectron2', 'numpy', 'onnx', 'onnxruntime', 'onnxruntime-gpu',
                'torch', 'torchvision', 'tensorflow', 'tensorrt']:
        try:
            mod = importlib.import_module(pkg)
            if hasattr(mod, "__version__"):
                versions[pkg] = mod.__version__
            else:
                versions[pkg] = "No version"
        except ModuleNotFoundError:
            versions[pkg] = "Not found"
    return versions


def make_report(show_cuda=True):
    report = Report()
    report.cuda = CUDA()

    report.platform = sys.platform
    report.python_version = sys.version
    report.python_path = sys.path
    report.environment = dict(os.environ)

    # ============ Pytorch and cuda stuff
    if show_cuda:
        import torch.cuda

        report.cuda.is_available = torch.cuda.is_available()
        report.cuda.device_count = torch.cuda.device_count()

    # ============ Package versions

    # Do a pip freeze and get the list
    report['pip_freeze'] = capture_pip_freeze()

    # Because of the way things are installed, sometimes pip doesn't have a version, so we grab them manually
    report['versions'] = capture_versions()

    # Get nvidia settings
    report.cuda.nvidia_smi_version = locate_nvidia_smi()
    report.gpu_list = get_cuda_devices()

    return report


if __name__ == "__main__":
    print(json.dumps(make_report(), indent=4))
