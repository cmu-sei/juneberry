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
import pytest

from juneberry.config.workspace import WorkspaceConfig


def make_sample_workspace_config(config_path):
    config_data = {
        "profiles": [
            {
                "include": "",
                "name": "",
                "model": "",
                "profile": {}
            },
            {
                "name": "default",
                "model": "default",
                "profile": {
                    "num_workers": 4
                }
            },
            {
                "name": "default",
                "model": "cool",
                "profile": {
                    "num_gpus": 12
                }
            },
            {
                "name": "pme",
                "model": "tabular",
                "profile": {
                    "num_workers": 20
                }
            },
            {
                "name": "gpu9",
                "model": "od",
                "profile": {
                    "num_workers": 16
                }
            },
            {
                "name": "gpu10",
                "model": "default",
                "profile": {
                    "num_workers": 8
                }
            },
            {
                "name": "gpu10",
                "model": "object_det_vers.*",
                "include": "default:cool",
                "profile": {}
            },
            {
                "name": "gpu10",
                "model": "object_det_.*",
                "include": "gpu9:od",
                "profile": {
                    "num_gpus": 6
                }
            },
        ]
    }

    with open(config_path, 'w') as json_file:
        json.dump(config_data, json_file)


def make_invalid_workspace_config(config_path):
    config_data = {
        "profiles": [
            {
                "include": "1984",
                "name": "pme",
                "model": "default",
                "profile": {
                }
            },
        ]
    }

    with open(config_path, 'w') as json_file:
        json.dump(config_data, json_file)


def test_invalid_profile_load(tmp_path):
    # Make invalid workspace config file
    config_path = str(Path(tmp_path, "inv_ws_config.json"))
    make_invalid_workspace_config(config_path)

    # Test invalid workspace config case
    # ws_specs = LabProfile.load(data_path=config_path, profile_name=None, model_name="not_good", test=True)
    with pytest.raises(RuntimeError) as exc_info:
        ws_config = WorkspaceConfig.load(config_path)
        lab_profile = ws_config.get_profile(profile_name=None, model_name="not_good", test=True)


def test_profile_load(tmp_path):
    # Make sample workspace config
    config_path = str(Path(tmp_path, "ws_config.json"))
    make_sample_workspace_config(config_path)

    # Load the good workspace config
    ws_config = WorkspaceConfig.load(config_path)

    # Test default:default case
    lab_profile = ws_config.get_profile(profile_name=None, model_name="not_good")
    assert lab_profile.num_workers == 4
    assert lab_profile.num_gpus is None

    # Test machine:default case
    lab_profile = ws_config.get_profile(profile_name="gpu10", model_name="not_great")
    assert lab_profile.num_workers == 8
    assert lab_profile.num_gpus is None

    # Test default:model case
    lab_profile = ws_config.get_profile(profile_name=None, model_name="cool4ever")
    assert lab_profile.num_workers == 4
    assert lab_profile.num_gpus == 12

    # Test machine:model case
    lab_profile = ws_config.get_profile(profile_name="pme", model_name="tabular_2")
    assert lab_profile.num_workers == 20
    assert lab_profile.num_gpus is None

    # Test more specific include case
    lab_profile = ws_config.get_profile(profile_name="gpu10", model_name="object_det_vers3")
    assert lab_profile.num_workers == 8
    assert lab_profile.num_gpus == 12

    # Test less specific include case
    lab_profile = ws_config.get_profile(profile_name="gpu10", model_name="object_det_other")
    assert lab_profile.num_workers == 16
    assert lab_profile.num_gpus == 6
