#!/usr/bin/env python

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

import setuptools
import platform

install_requires = [
    "doit",
    "numpy",
    "pycocotools",
    "matplotlib",
    "pillow",
    "prodict",
    "hjson",
    "jsonschema",
    "sklearn",
    "tensorflow",
    "tensorflow-datasets",
    "tensorboard",
    "torch",
    "torchvision",
    "torch-summary>=1.4.5",
    "pandas",
    "brambox",
    "pyyaml",
    "hjson",
    "natsort",
    "opacus",
    "protobuf==3.16.0",
    "onnx",
    "onnxruntime",  # pip install onnxruntime-gpu if on cuda, otherwise onnxruntime is sufficient
    "tf2onnx",
    "tqdm"
]


def customize_requires(requires):
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        darwin_packages = {
            'tensorflow': 'tensorflow-macos'
        }

        # Rename stuff
        requires = [darwin_packages.get(x, x) for x in requires]

        # OSX doesn't have onnx runtime yet but tf2onnx will install!
        # https://github.com/microsoft/onnxruntime/issues/6633
        missing = ['onnxruntime']
        for item in missing:
            if item in requires
                del requires[item]

    return requires


bin_scripts = [
    'bin/jb_attack_to_rules',
    'bin/jb_clean_predictions',
    'bin/jb_evaluate',
    'bin/jb_experiment_to_rules',
    'bin/jb_generate_experiments',
    'bin/jb_gpu_runner',
    'bin/jb_plot_pr',
    'bin/jb_plot_roc',
    'bin/jb_rules_to_pydoit',
    'bin/jb_run_experiment',
    'bin/jb_run_plugin',
    'bin/jb_summary_report',
    'bin/jb_train'
]

setuptools.setup(
    name='Juneberry',
    version='0.5',
    description='Juneberry Machine Learning Experiment Manager',
    packages=setuptools.find_packages(),
    install_requires=customize_requires(install_requires),
    scripts=bin_scripts,
    python_requires='>=3.7',
    include_package_data=True
)
