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

import juneberry.transform_manager


def test_transform():
    config = [
        {
            'fqcn': 'moddir.simple_mod.ClassWithInitAndUnaryCall',
            'kwargs': {'name': 'frodo'}
        }
    ]

    ttm = juneberry.transform_manager.TransformManager(config)
    assert len(ttm) == 1
    assert ttm.get_fqn(0) == 'moddir.simple_mod.ClassWithInitAndUnaryCall'
    assert ttm.transform("baggins") == 'frodo baggins'


def test_opt_args():
    config = [
        {
            'fqcn': 'moddir.simple_mod.ClassWithInitAndUnaryCall'
        }
    ]

    opt_args = {"name": "frodo", "bar": 1234}
    ttm = juneberry.transform_manager.TransformManager(config, opt_args)
    assert len(ttm) == 1
    assert ttm.get_fqn(0) == 'moddir.simple_mod.ClassWithInitAndUnaryCall'
    assert ttm.transform("baggins") == 'frodo baggins'
