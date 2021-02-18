#! /usr/bin/env python3

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import csv
import random
import os

from pathlib import Path

from unittest import mock

import juneberry
import juneberry.data as jb_data
import juneberry.filesystem as jbfs
from juneberry.config.dataset import DatasetConfig
from juneberry.config.training import TrainingConfig
import juneberry.config.dataset as jb_dataset

import test_training_config


def make_data():
    data_list = [["a", 0],
                 ["b", 1],
                 ["c", 2],
                 ["d", 3],
                 ["e", 0],
                 ["f", 1],
                 ["g", 2],
                 ["h", 3],
                 ["i", 0],
                 ["j", 0],
                 ["k", 1],
                 ["l", 1],
                 ]
    data_dict = {0: ['a', 'e', 'i', 'j'],
                 1: ['b', 'f', 'k', 'l'],
                 2: ['c', 'g'],
                 3: ['d', 'h']}
    return data_list, data_dict


def check_allocation(good_dict, result_dict):
    for k, v in result_dict.items():
        for i in v:
            assert i in good_dict[k]


def test_listdir_no_hidden():
    with mock.patch('os.listdir') as mocked_listdir:
        mocked_listdir.return_value = ['thing1', '.myhidden', 'thing2']

        results = jb_data.listdir_nohidden('')
        assert len(results) == 2
        assert '.myhidden' not in results
        assert 'thing1' in results
        assert 'thing2' in results


#  _____
# |_   _|
#   | | _ __ ___   __ _  __ _  ___
#   | || '_ ` _ \ / _` |/ _` |/ _ \
#  _| || | | | | | (_| | (_| |  __/
#  \___/_| |_| |_|\__,_|\__, |\___|
#                        __/ |
#                       |___/


# This is a hard-code list dir that we use to test get images
def mock_list_image_dir(path):
    if str(path).endswith("frodo"):
        return [f'fr_{x}.png' for x in range(6)]
    elif str(path).endswith("sam"):
        return [f'sm_{x}.png' for x in range(6)]
    else:
        return []


def make_basic_data_set_image_config():
    return {
        "numModelClasses": 4,
        "timestamp": "never",
        "formatVersion": jb_dataset.FORMAT_VERSION,
        "labelNames": {"0": "frodo", "1": "sam"},
        "dataType": 'image',
        "imageData": {
            "taskType": "classification",
            "sources": [
                {
                    "directory": "frodo",
                    "label": 0,
                    # "samplingCount": "4",
                    # "samplingFraction": ""
                },
                {
                    "directory": "sam",
                    "label": 1,
                    # "samplingCount": "4",
                    # "samplingFraction": ""
                }
            ]
        }
    }


def make_sample_stanza(algorithm, args):
    return {
        "sampling": {
            "algorithm": algorithm,  # < 'none', 'randomFraction', 'randomQuantity', 'roundRobin' >
            "arguments": args  # < custom json structure depending on algorithm - see details >
        }
    }


def assert_correct_list(test_list, frodo_indexes, sam_indexes):
    correct_names = [str(Path('data_root', 'frodo', f'fr_{x}.png')) for x in frodo_indexes]
    correct_labels = [0] * len(frodo_indexes)
    correct_names.extend([str(Path('data_root', 'sam', f'sm_{x}.png')) for x in sam_indexes])
    correct_labels.extend([1] * len(sam_indexes))
    for idx, train in enumerate(test_list):
        assert train[0] == correct_names[idx]
        assert train[1] == correct_labels[idx]


def test_generate_image_list():
    # Just replace listdir
    os.listdir = mock_list_image_dir

    data_set_struct = make_basic_data_set_image_config()

    data_set_config = DatasetConfig(data_set_struct)
    dm = jbfs.DataManager({})

    train_list, val_list = jb_data.generate_image_list('data_root', data_set_config, None, dm)
    assert len(train_list) == 12
    assert len(val_list) == 0

    assert_correct_list(train_list, range(6), range(6))


def test_generate_image_sample_quantity():
    # If we pass in sampling count we should just get those
    # We know how the internal randomizer works.  We know it uses random.sample on both
    # sets in order.  This is a secret and fragile to this test.
    # With a seed of 1234 and two pulls of sampling with a count of 3, it pulls [3,0,4] and [0,4,5]
    os.listdir = mock_list_image_dir

    data_set_struct = make_basic_data_set_image_config()
    data_set_struct.update(make_sample_stanza("randomQuantity", {'seed': 1234, 'count': 3}))

    data_set_config = DatasetConfig(data_set_struct)
    dm = jbfs.DataManager({})

    train_list, val_list = jb_data.generate_image_list('data_root', data_set_config, None, dm)
    assert len(train_list) == 6
    assert len(val_list) == 0

    # Make sure they are in this order
    assert_correct_list(train_list, [3, 0, 4], [0, 4, 5])


def test_generate_image_sample_fraction():
    # If we pass in sampling count we should just get those
    # We know how the internal randomizer works.  We know it uses random.sample on both
    # sets in order.  This is a secret and fragile to this test.
    # With a seed of 1234 and two pulls of sampling with a count of 2, it pulls [3,0] and [0,5]
    os.listdir = mock_list_image_dir

    data_set_struct = make_basic_data_set_image_config()
    data_set_struct.update(make_sample_stanza("randomFraction", {'seed': 1234, 'fraction': 0.3333333333}))

    data_set_config = DatasetConfig(data_set_struct)
    dm = jbfs.DataManager({})

    train_list, val_list = jb_data.generate_image_list('data_root', data_set_config, None, dm)
    assert len(train_list) == 4
    assert len(val_list) == 0

    # Make sure they are in this order
    assert_correct_list(train_list, [3, 0], [0, 5])


def test_generate_image_validation_split():
    os.listdir = mock_list_image_dir

    data_set_struct = make_basic_data_set_image_config()
    data_set_config = DatasetConfig(data_set_struct)

    train_struct = test_training_config.make_basic_config()
    train_struct['validation'] = {
        "algorithm": "randomFraction",
        "arguments": {
            "seed": 1234,
            "fraction": 0.3333333
        }
    }
    train_config = TrainingConfig('', train_struct)
    dm = jbfs.DataManager({})

    train_list, val_list = jb_data.generate_image_list('data_root', data_set_config, train_config, dm)
    assert len(train_list) == 8
    assert len(val_list) == 4

    # NOTE: Another fragile secret we know is the order from the validation is is reversed
    assert_correct_list(train_list, [1, 2, 4, 5], [1, 2, 3, 4])
    assert_correct_list(val_list, [3, 0], [5, 0])


#  _____     _           _
# |_   _|   | |         | |
#   | | __ _| |__  _   _| | __ _ _ __
#   | |/ _` | '_ \| | | | |/ _` | '__|
#   | | (_| | |_) | |_| | | (_| | |
#   \_/\__,_|_.__/ \__,_|_|\__,_|_|
#

def make_basic_data_set_tabular_config():
    return {
        "numModelClasses": 4,
        "timestamp": "never",
        "formatVersion": jb_dataset.FORMAT_VERSION,
        "labelNames": {"0": "frodo", "1": "sam"},
        "dataType": 'tabular',
        "tabularData": {
            "sources": [
                {
                    "root": "dataroot",  # [ dataroot | workspace | relative ]
                    "path": "dr.csv",  # subdirectory
                },
                {
                    "root": "workspace",  # [ dataroot | workspace | relative ]
                    "path": "ws.csv",  # subdirectory
                },
                {
                    "root": "relative",  # [ dataroot | workspace | relative ]
                    "path": "re*.csv",  # subdirectory
                }
            ],
            "labelIndex": 2
        }
    }


def fill_tabular_tempdir(root_dir):
    """
    Creates the sample files to be read and returns the data we should find
    :param root_dir: The root directory
    :return: Good data in a dict of label -> dict of x -> y
    """
    results = {0: {}, 1: {}}

    # Directory, filename, val_range
    dir_struct = [
        ['myworkspace', 'ws.csv', list(range(0, 4))],
        ['mydataroot', 'dr.csv', list(range(4, 8))],
        ['myrelative', 'rel.csv', list(range(8, 12))]
    ]

    for dir_name, file_name, data in dir_struct:
        dir_path = Path(root_dir) / dir_name
        dir_path.mkdir()

        csv_path = dir_path / file_name
        with open(csv_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow('x,y,label')
            for idx, val in enumerate(data):
                results[idx % 2][val] = val + 10
                csv_writer.writerow([val, val + 10, idx % 2])

    return results


def test_load_tabular_data(tmp_path):
    correct = fill_tabular_tempdir(tmp_path)
    juneberry.WORKSPACE_ROOT = Path(tmp_path) / 'myworkspace'
    juneberry.DATA_ROOT = Path(tmp_path) / 'mydataroot'

    data_set_struct = make_basic_data_set_tabular_config()
    data_set_config = DatasetConfig(data_set_struct, Path(tmp_path) / 'myrelative')

    train_list, val_list = jb_data.load_tabular_data(None, data_set_config)

    # THe sample data is three files each with 4 sample with 2 in each class.
    # THe default validation split is 2.  So 3 * 4 / 2 = 6 per list
    assert len(train_list) == 12
    assert len(val_list) == 0

    # Make sure that evert returned value is in the results.
    for data, label in train_list:
        assert correct[int(label)][int(data[0])] == int(data[1])
        del correct[int(label)][int(data[0])]


def test_load_tabular_data_with_sampling(tmp_path):
    correct = fill_tabular_tempdir(tmp_path)
    juneberry.WORKSPACE_ROOT = Path(tmp_path) / 'myworkspace'
    juneberry.DATA_ROOT = Path(tmp_path) / 'mydataroot'

    # We only need to test one sample because the sampling core is tested elsewhere
    data_set_struct = make_basic_data_set_tabular_config()
    data_set_struct.update(make_sample_stanza("randomQuantity", {'seed': 1234, 'count': 3}))
    data_set_config = DatasetConfig(data_set_struct, Path(tmp_path) / 'myrelative')

    train_list, val_list = jb_data.load_tabular_data(None, data_set_config)

    # THe sample data is three files each with 4 sample with 2 in each class.
    # THe default validation split is 2.  So 3 * 4 / 2 = 6 per list
    assert len(train_list) == 6
    assert len(val_list) == 0

    # Now, make sure they are in each one, removing as we go
    for data, label in train_list:
        assert correct[int(label)][int(data[0])] == int(data[1])
        del correct[int(label)][int(data[0])]

    # At this point we should have three unused entries of each class
    assert len(correct[0]) == 3
    assert len(correct[1]) == 3


def test_load_tabular_data_with_validation(tmp_path):
    correct = fill_tabular_tempdir(tmp_path)
    juneberry.WORKSPACE_ROOT = Path(tmp_path) / 'myworkspace'
    juneberry.DATA_ROOT = Path(tmp_path) / 'mydataroot'

    data_set_struct = make_basic_data_set_tabular_config()
    data_set_config = DatasetConfig(data_set_struct, Path(tmp_path) / 'myrelative')

    train_struct = test_training_config.make_basic_config()
    train_config = TrainingConfig('', train_struct)

    train_list, val_list = jb_data.load_tabular_data(train_config, data_set_config)

    # THe sample data is three files each with 4 sample with 2 in each class.
    # THe default validation split is 2.  So 3 * 4 / 2 = 6 per list
    assert len(train_list) == 6
    assert len(val_list) == 6

    # Now, make sure they are in each one, removing as we go
    for data, label in train_list:
        assert correct[int(label)][int(data[0])] == int(data[1])
        del correct[int(label)][int(data[0])]

    assert len(correct[0]) == 3
    assert len(correct[1]) == 3

    for data, label in val_list:
        assert correct[int(label)][int(data[0])] == int(data[1])
        del correct[int(label)][int(data[0])]

    assert len(correct[0]) == 0
    assert len(correct[1]) == 0


#  _____                       _ _
# /  ___|                     | (_)
# \ `--.  __ _ _ __ ___  _ __ | |_ _ __   __ _
#  `--. \/ _` | '_ ` _ \| '_ \| | | '_ \ / _` |
# /\__/ / (_| | | | | | | |_) | | | | | | (_| |
# \____/ \__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
#                       | |               __/ |
#                       |_|              |___/


def test_sampling_random_quantity():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(6))
    sampled = jb_data.sample_data_list(data_list, "randomQuantity", {"count": 3}, randomizer)
    for correct, test in zip([3, 0, 4], sampled):
        assert correct == test


def test_sampling_random_fraction():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(6))
    sampled = jb_data.sample_data_list(data_list, "randomFraction", {"fraction": 0.3333333333}, randomizer)
    for correct, test in zip([3, 0], sampled):
        assert correct == test


def test_sampling_round_robin():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(9))
    sampled = jb_data.sample_data_list(data_list, "roundRobin", {"groups": 3, "position": 1}, randomizer)
    for correct, test in zip([3, 4, 1], sampled):
        assert correct == test


def test_sampling_none():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(8))
    sampled = jb_data.sample_data_list(data_list, "none", {}, randomizer)
    for correct, test in zip(range(8), sampled):
        assert correct == test


# ___  ____
# |  \/  (_)
# | .  . |_ ___  ___
# | |\/| | / __|/ __|
# | |  | | \__ \ (__
# \_|  |_/_|___/\___|

def test_flatten_dict_to_pairs():
    data_list, data_dict = make_data()
    result_pairs = jb_data.flatten_dict_to_pairs(data_dict)

    # Order doesn't matter. Just check to make sure that the entries are in the original dict
    assert len(result_pairs) == len(data_list)
    for v, k in result_pairs:
        assert v in data_dict[k]


def test_labeled_pairs_to_labeled_dict():
    data_list, data_dict = make_data()
    result_dict = jb_data.labeled_pairs_to_labeled_dict(data_list)

    assert len(result_dict) == len(data_dict)
    for k, v in result_dict.items():
        assert len(v) == len(data_dict[k])
        for i in v:
            assert i in data_dict[k]


def test_make_balanced_list():
    data_list, data_dict = make_data()
    result = jb_data.make_balanced_labeled_list(data_list, -1, random.Random())

    result_dict = jb_data.labeled_pairs_to_labeled_dict(result)

    assert len(result_dict) == 4
    for k, v in result_dict.items():
        assert len(v) == 2
    check_allocation(data_dict, result_dict)


def test_make_balanced_dict():
    data_list, data_dict = make_data()
    result_dict = jb_data.make_balanced_labeled_dict(data_dict, -1, random.Random())

    assert len(result_dict) == 4
    for k, v in result_dict.items():
        assert len(v) == 2
    check_allocation(data_dict, result_dict)
