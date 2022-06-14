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

from pathlib import Path
import os

import juneberry.scripting.utils as jb_scripting_utils


class MockArgs:
    def __init__(self):
        self.workspace = None
        self.dataRoot = None
        self.tensorboard = None
        self.profileName = None


def test_defaults():
    vals = jb_scripting_utils.make_default_values("/foo")
    assert vals['workspace'] == "/foo"
    assert vals['data_root'] == "/dataroot"
    assert vals['tensorboard'] == "/tensorboard"
    assert vals['profile_name'] == "default"


def test_resolve_lab_args_ws():
    orig = os.environ.copy()

    os.environ.pop('JUNEBERRY_WORKSPACE', None)
    os.environ.pop('JUNEBERRY_DATA_ROOT', None)
    os.environ.pop('JUNEBERRY_TENSORBOARD', None)
    os.environ.pop('JUNEBERRY_PROFILE_NAME', None)

    args = MockArgs()
    args.workspace = "/fakeroot/ws"

    vals = jb_scripting_utils.resolve_lab_args(args)

    assert vals['workspace'] == "/fakeroot/ws"
    assert vals['data_root'] == "/fakeroot/dataroot"
    assert vals['tensorboard'] == "/fakeroot/tensorboard"
    assert vals['profile_name'] == "default"

    # Reset the environ so we don't impact other tests
    os.environ = orig.copy()


def test_env_variables():
    orig = os.environ.copy()
    os.environ['JUNEBERRY_WORKSPACE'] = "js_ws"
    os.environ['JUNEBERRY_DATA_ROOT'] = "js_dr"
    os.environ['JUNEBERRY_TENSORBOARD'] = "js_tb"
    os.environ['JUNEBERRY_PROFILE_NAME'] = "prof_name"

    args = MockArgs()
    vals = jb_scripting_utils.resolve_lab_args(args)

    assert vals['workspace'] == str((Path.cwd() / "js_ws").absolute())
    assert vals['data_root'] == str((Path.cwd() / "js_dr").absolute())
    assert vals['tensorboard'] == str((Path.cwd() / "js_tb").absolute())
    assert vals['profile_name'] == "prof_name"

    # Reset the environ so we don't impact other tests
    os.environ = orig.copy()


def test_overrides():
    orig = os.environ.copy()

    os.environ.pop('JUNEBERRY_TENSORBOARD', None)

    args = MockArgs()
    args.workspace = "/fakeroot/ws"
    args.dataRoot = "/some/hardcoded/path"

    vals = jb_scripting_utils.resolve_lab_args(args)

    assert vals['workspace'] == "/fakeroot/ws"
    assert vals['data_root'] == "/some/hardcoded/path"
    assert vals['tensorboard'] == "/fakeroot/tensorboard"
    assert vals['profile_name'] == "default"

    # Reset the environ so we don't impact other tests
    os.environ = orig.copy()
