#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM22-0856
#
# ======================================================================================================================

import juneberry.utils as jb_utils


def setup_data():
    test_data = {
        "someKey": 1,
        "otherKey": 2,
        'nested': {
            'arrayKey': [1, 2, 3],
            'dictKey': {
                'subKeyA': 'Frodo',
                'subKeyB': 'Sam',
                'okay': 'Merry'
            }
        }
    }

    expected_data = {
        "some_key": 1,
        "other_key": 2,
        'nested': {
            'array_key': [1, 2, 3],
            'dict_key': {
                'sub_key_a': 'Frodo',
                'sub_key_b': 'Sam',
                'okay': 'Merry'
            }
        }
    }

    key_map = {
        'someKey': 'some_key',
        'otherKey': 'other_key',
        'arrayKey': 'array_key',
        'dictKey': 'dict_key',
        'subKeyA': 'sub_key_a',
        'subKeyB': 'sub_key_b',
    }

    return test_data, expected_data, key_map


def test_rekey():
    test_data, expected_data, key_map = setup_data()

    # Convert based on a known key_map
    jb_utils.rekey(test_data, key_map)
    assert expected_data == test_data


def test_snake_case():
    test_data, expected_data, key_map = setup_data()

    # Convert based on the algo
    new_map = jb_utils.mixed_to_snake_struct_keys(test_data)
    assert expected_data == test_data
    assert key_map == new_map
