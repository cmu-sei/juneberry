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

install_requires = [
    "doit",
    "numpy",
    "matplotlib",
    "pillow",
    "prodict",
    "hjson",
    "jsonschema",
    "sklearn",
    "tensorboard",
    "torch",
    "torchvision",
    "torch-summary>=1.4.5",
    "pandas",
    "brambox",
    "pyyaml",
    "hjson",
    "natsort"
]

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
    install_requires=install_requires,
    scripts=bin_scripts,
    python_requires='>=3.7',
)
