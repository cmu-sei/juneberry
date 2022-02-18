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

import csv
import json
import unittest
from pathlib import Path

import juneberry
from juneberry.lab import Lab
from juneberry.config.machine_specs import MachineSpecs


def make_sample_machine_specs(specs_path):
    specs_data = {
        "default": {
            "default": {
                "num_workers": 4
            },
            "cool": {
                "num_gpus": 12
            }
        },
        "pme": {
            "tabular": {
                "num_workers": 20
            }
        },
        "gpu9": {
            "od": {
                "num_workers": 16
            }
        }
        ,
        "gpu10": {
            "default": {
                "num_workers": 8
            },
            "object_det_.*": {
                "include": "gpu9:od"
            }
        }
    }

    with open(specs_path, 'w') as json_file:
        json.dump(specs_data, json_file)


def make_invalid_specs(specs_path):
    specs_data = {
        "pme": {
            "default": {
                "include": "1984"
            }
        }
    }

    with open(specs_path, 'w') as json_file:
        json.dump(specs_data, json_file)


def test_machine_specs(tmp_path):
    # Make invalid machine specs file
    specs_path = str(Path(tmp_path, "invalid_specs.json"))
    make_invalid_specs(specs_path)

    # Test invalid machine specs file case
    specs_data = MachineSpecs.load(data_path=specs_path, machine_class=None, model_name="not_good", test=True)
    assert specs_data is False

    # Make sample machine specs file
    specs_path = str(Path(tmp_path, "machine_specs.json"))
    make_sample_machine_specs(specs_path)

    # Test default:default case
    specs_data = MachineSpecs.load(data_path=specs_path, machine_class=None, model_name="not_good")
    assert specs_data == {"num_workers": 4, "num_gpus": None}

    # Test machine:default case
    specs_data = MachineSpecs.load(data_path=specs_path, machine_class="gpu10", model_name="not_great")
    assert specs_data == {"num_workers": 8, "num_gpus": None}

    # Test default:model case
    specs_data = MachineSpecs.load(data_path=specs_path, machine_class=None, model_name="cool4ever")
    assert specs_data == {"num_workers": 4, "num_gpus": 12}

    # Test machine:model case
    specs_data = MachineSpecs.load(data_path=specs_path, machine_class="pme", model_name="tabular_2")
    assert specs_data == {"num_workers": 20, "num_gpus": None}

    # Test include case
    specs_data = MachineSpecs.load(data_path=specs_path, machine_class="gpu10", model_name="object_det_vers3")
    assert specs_data == {"num_workers": 16, "num_gpus": None}
