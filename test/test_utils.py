#! /usr/bin/env python3

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

import juneberry.utils as jb_utils
from juneberry.filesystem import ModelManager


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


def test_get_label_mapping():
    # Binary sample
    model_name_bin = "tabular_binary_sample"
    expected_labels_bin = {0: "outer", 1: "inner"}
    model_manager_bin = ModelManager(model_name_bin)
    model_config_bin = model_manager_bin.get_model_config()
    train_config_bin = "models/tabular_binary_sample/train_data_config.json"

    # Output.json
    source = "training config 2"
    assert expected_labels_bin, source == jb_utils.get_label_mapping(model_manager_bin, show_source=True)

    # Model config
    source = "training config 2"
    assert expected_labels_bin, source == jb_utils.get_label_mapping(model_manager_bin, model_config=model_config_bin,
                                                                 show_source=True)

    # Training config
    source = "training config 1"
    assert expected_labels_bin, source == jb_utils.get_label_mapping(model_manager_bin, train_config=train_config_bin,
                                                                 show_source=True)

    # Multiclass sample
    model_name_multi = "tabular_multiclass_sample"
    model_manager_multi = ModelManager(model_name_multi)
    model_config_multi = model_manager_multi.get_model_config()
    expected_labels_multi = {0: "label_0", 1: "label_1", 2: "label_2"}
    train_config_multi = "models/tabular_multiclass_sample/train_data_config.json"
    eval_config_multi = "models/tabular_multiclass_sample/val_data_config.json"

    # Output.json
    source = "training config 2"
    assert expected_labels_multi, source == jb_utils.get_label_mapping(model_manager_multi, show_source=True)

    # Model config
    source = "training config 2"
    assert expected_labels_multi, source == jb_utils.get_label_mapping(model_manager_multi, model_config=model_config_multi,
                                                                   show_source=True)

    # Training config
    source = "training config 1"
    assert expected_labels_multi, source == jb_utils.get_label_mapping(model_manager_multi, train_config=train_config_multi,
                                                                   show_source=True)

    # Eval config
    source = "training config 2"
    assert expected_labels_multi, source == jb_utils.get_label_mapping(model_manager_multi, eval_config=eval_config_multi,
                                                                   show_source=True)


test_get_label_mapping()
