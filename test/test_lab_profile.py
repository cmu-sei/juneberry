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
from juneberry.config.lab_profile import LabProfile


def make_sample_workspace_config(config_path):
    config_data = {
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
        },
        "gpu10": {
            "default": {
                "num_workers": 8
            },
            "object_det_.*": {
                "include": "gpu9:od"
            },
            "object_det_vers.*": {
                "include": "default:cool"
            }
        }
    }

    with open(config_path, 'w') as json_file:
        json.dump(config_data, json_file)


def make_invalid_workspace_config(config_path):
    config_data = {
        "pme": {
            "default": {
                "include": "1984"
            }
        }
    }

    with open(config_path, 'w') as json_file:
        json.dump(config_data, json_file)


def test_lab_profile_load(tmp_path):
    # Make invalid workspace config file
    config_path = str(Path(tmp_path, "inv_ws_config.json"))
    make_invalid_workspace_config(config_path)

    # Test invalid workspace config case
    ws_specs = LabProfile.load(data_path=config_path, profile_name=None, model_name="not_good", test=True)
    assert ws_specs is False

    # Make sample workspace config
    config_path = str(Path(tmp_path, "ws_config.json"))
    make_sample_workspace_config(config_path)

    # Test default:default case
    ws_specs = LabProfile.load(data_path=config_path, profile_name=None, model_name="not_good")
    assert ws_specs['num_workers'] == 4
    assert ws_specs['num_gpus'] is None

    # Test machine:default case
    ws_specs = LabProfile.load(data_path=config_path, profile_name="gpu10", model_name="not_great")
    assert ws_specs['num_workers'] == 8
    assert ws_specs['num_gpus'] is None

    # Test default:model case
    ws_specs = LabProfile.load(data_path=config_path, profile_name=None, model_name="cool4ever")
    assert ws_specs['num_workers'] == 4
    assert ws_specs['num_gpus'] == 12

    # Test machine:model case
    ws_specs = LabProfile.load(data_path=config_path, profile_name="pme", model_name="tabular_2")
    assert ws_specs['num_workers'] == 20
    assert ws_specs['num_gpus'] is None

    # Test include case
    ws_specs = LabProfile.load(data_path=config_path, profile_name="gpu10", model_name="object_det_vers3")
    assert ws_specs['num_workers'] == 16
    assert ws_specs['num_gpus'] == 12
