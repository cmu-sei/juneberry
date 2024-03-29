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

import csv
import json
from pathlib import Path
import random
from unittest import mock, TestCase

import juneberry
from juneberry.config.dataset import DatasetConfig, SamplingAlgo
from juneberry.config.model import ModelConfig
from juneberry.config.training_output import TrainingOutput
import juneberry.data as jb_data
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
from juneberry.transforms.transform_manager import TransformManager
from test_coco_utils import make_sample_coco
import utils


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


# Replace our data utility function which underlies all the various pathing and globbing.
def mock_list_or_glob_dir(data_root: Path, path: str):
    if str(path).endswith("frodo"):
        return [data_root / path / f'fr_{x}.png' for x in range(6)]
    elif str(path).endswith("sam"):
        return [data_root / path / f'sm_{x}.png' for x in range(6)]
    else:
        return []


def make_image_classification_config():
    return {
        "num_model_classes": 4,
        "timestamp": "never",
        "format_version": DatasetConfig.FORMAT_VERSION,
        "label_names": {"0": "frodo", "1": "sam"},
        "data_type": 'image',
        "image_data": {
            "task_type": "classification",
            "sources": [
                {
                    "directory": "frodo",
                    "label": 0,
                    # "sampling_count": "4",
                    # "sampling_fraction": ""
                },
                {
                    "directory": "sam",
                    "label": 1,
                    # "sampling_count": "4",
                    # "sampling_fraction": ""
                }
            ]
        }
    }


def make_sample_stanza(algorithm, args):
    return {
        "sampling": {
            "algorithm": algorithm,  # < 'none', 'random_fraction', 'random_quantity', 'round_robin' >
            "arguments": args  # < custom json structure depending on algorithm - see details >
        }
    }


def assert_correct_list(test_list, frodo_indexes, sam_indexes, data_root='data_root'):
    correct_names = [str(Path(data_root, 'frodo', f'fr_{x}.png')) for x in frodo_indexes]
    correct_labels = [0] * len(frodo_indexes)
    correct_names.extend([str(Path(data_root, 'sam', f'sm_{x}.png')) for x in sam_indexes])
    correct_labels.extend([1] * len(sam_indexes))
    assert len(test_list) == len(correct_names)
    for idx, train in enumerate(test_list):
        assert str(train[0]) == correct_names[idx]
        assert train[1] == correct_labels[idx]


def test_generate_image_list(monkeypatch):
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_list_or_glob_dir)

    dataset_struct = make_image_classification_config()

    lab = Lab(workspace='.', data_root='data_root')
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_image_manifests(lab, dataset_config)
    assert len(train_list) == 12
    assert len(val_list) == 0

    assert_correct_list(train_list, range(6), range(6))


def test_generate_image_sample_quantity(monkeypatch):
    # If we pass in sampling count we should just get those.
    # We know how the internal randomizer works.  We know it uses random.sample on both
    # sets in order.  This is a secret and fragile to this test.
    # With a seed of 1234 and two pulls of sampling with a count of 3, it pulls [3,0,4] and [0,4,5]
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_list_or_glob_dir)

    dataset_struct = make_image_classification_config()
    dataset_struct.update(make_sample_stanza("random_quantity", {'seed': 1234, 'count': 3}))

    lab = Lab(workspace='.', data_root='data_root')
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_image_manifests(lab, dataset_config)
    assert len(train_list) == 6
    assert len(val_list) == 0

    # Make sure they are in this order
    assert_correct_list(train_list, [0, 3, 4], [0, 4, 5])


def test_generate_image_sample_fraction(monkeypatch):
    # If we pass in sampling count we should just get those.
    # We know how the internal randomizer works.  We know it uses random.sample on both
    # sets in order.  This is a secret and fragile to this test.
    # With a seed of 1234 and two pulls of sampling with a count of 2, it pulls [3,0] and [0,5]
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_list_or_glob_dir)

    dataset_struct = make_image_classification_config()
    dataset_struct.update(make_sample_stanza("random_fraction", {'seed': 1234, 'fraction': 0.3333333333}))

    lab = Lab(workspace='.', data_root='data_root')
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_image_manifests(lab, dataset_config)
    assert len(train_list) == 4
    assert len(val_list) == 0

    # Make sure they are in this order.
    assert_correct_list(train_list, [0, 3], [0, 5])


def test_generate_image_validation_split(monkeypatch, tmp_path):
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_list_or_glob_dir)

    lab = Lab(workspace='.', data_root='data_root')
    dataset_struct = make_image_classification_config()
    dataset_config = DatasetConfig.construct(dataset_struct)

    config = utils.make_basic_model_config()
    model_config = ModelConfig.construct(data=config)
    model_config.validation = {
        "algorithm": "random_fraction",
        "arguments": {
            "seed": 1234,
            "fraction": 0.3333333
        }
    }

    split_config = model_config.get_validation_split_config()

    train_list, val_list = jb_data.generate_image_manifests(lab, dataset_config, splitting_config=split_config)
    assert len(train_list) == 8
    assert len(val_list) == 4

    # NOTE: Another fragile secret we know is the order from the validation is reversed.
    assert_correct_list(train_list, [1, 2, 4, 5], [1, 2, 3, 4])
    assert_correct_list(val_list, [3, 0], [5, 0])

    # Now test the "from_file" validation algorithm.
    # Create a dataset config file.
    dataset_path = Path(tmp_path / "dataset_config.json")
    dataset_config.save(dataset_path)

    # Adjust the validation stanza to the "from_file" algorithm.
    model_config.validation = {
        "algorithm": "from_file",
        "arguments": {
            "file_path": str(dataset_path)
        }
    }

    # Get the train and val lists.
    split_config = model_config.get_validation_split_config()
    train_list, val_list = jb_data.generate_image_manifests(lab, dataset_config, splitting_config=split_config)

    # Check the train list for correctness. We expect 12 images; 6 frodo and 6 sam.
    assert len(train_list) == 12
    assert_correct_list(train_list, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])

    # The val list should be identical to the train list, since it's the same dataset config just loaded from file.
    assert train_list == val_list


#  __  __      _            _       _
# |  \/  | ___| |_ __ _  __| | __ _| |_ __ _
# | |\/| |/ _ \ __/ _` |/ _` |/ _` | __/ _` |
# | |  | |  __/ || (_| | (_| | (_| | || (_| |
# |_|  |_|\___|\__\__,_|\__,_|\__,_|\__\__,_|


def mock_data_listdir_metadata(data_root: Path, path: str):
    p = Path(path)
    if str(path).startswith("frodo"):
        return [data_root / p.parent / f'metadata_1.json']
    elif str(path).startswith("sam"):
        return [data_root / p.parent / f'metadata_2.json']
    else:
        return []


def make_image_object_detection_config(include_image_removal=False):
    """
    This creates a simple dataset config for an objectDetection task.
    :return: A dataset config.
    """
    return {
        "task": "objectDetection:",
        "num_model_classes": 4,
        "timestamp": "never",
        "format_version": DatasetConfig.FORMAT_VERSION,
        "label_names": {"0": "frodo", "1": "sam"},
        "data_type": 'image',
        "image_data": {
            "task_type": "object_detection",
            "sources": [
                {
                    "directory": "frodo/*.json",
                    "remove_image_ids": [1] if include_image_removal else []
                },
                {
                    "directory": "sam/*.json",
                }
            ]
        }
    }


def make_ids(idx):
    # Make two images, and then 2 and 3 annotations.
    return [idx, idx + 10], [[idx, idx + 10], [idx + 20, idx + 30, idx + 40]]


def create_meta_file(dir_path: Path, idx):
    file_path = dir_path / f"metadata_{idx}.json"
    data = make_sample_coco(*make_ids(idx))
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)

    return data


def setup_metadata_files(tmp_path: Path):
    # Create two metadata directories, each with one metadata file.
    anno_files = []
    (tmp_path / 'frodo').mkdir()
    anno_files.append(create_meta_file(tmp_path / 'frodo', 1))
    (tmp_path / 'sam').mkdir()
    anno_files.append(create_meta_file(tmp_path / 'sam', 2))

    return anno_files


def assert_correct_metadata_list(test_list, id_map):
    assert len(test_list) == len(id_map)
    for item in test_list:
        assert item['id'] in id_map
        anno_ids = id_map.get(item['id'], {})
        assert sorted(anno_ids) == sorted([x['id'] for x in item['annotations']])


# ==============================================================================

def make_dir_and_items(the_dir: Path, items: list):
    the_dir.mkdir()
    for item in items:
        (the_dir / item).touch()


def test_list_or_glob_dir(tmp_path):
    print("Running test")
    # First we need to make a sample structure
    plain_items = ['one', 'two', 'three']
    make_dir_and_items(Path(tmp_path) / "plain", plain_items)

    items = jb_data.list_or_glob_dir(Path(tmp_path), "plain")
    assert len(items) == len(plain_items)
    for item in plain_items:
        assert Path(tmp_path) / "plain" / item in items

    # =====

    all_items = ['item1.txt', 'item2.txt', 'item3.png']
    text_items = ['item1.txt', 'item2.txt']
    make_dir_and_items(Path(tmp_path) / "glob", all_items)

    items = jb_data.list_or_glob_dir(Path(tmp_path), "glob/*.txt")
    assert len(items) == len(text_items)
    for item in text_items:
        assert Path(tmp_path) / "glob" / item in items


def test_add_image_data_sources(monkeypatch, tmp_path):
    # This gives use two directories of six items each
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_list_or_glob_dir)

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_struct = make_image_classification_config()
    dataset_config = DatasetConfig.construct(dataset_struct)
    sources_list = []
    jb_data.add_image_data_sources(lab, dataset_config, sources_list, 'train')

    assert len(sources_list) == 2
    assert len(sources_list[0]['train']) == 6
    assert len(sources_list[1]['train']) == 6

# ==============================================================================


def test_generate_metadata_list(monkeypatch, tmp_path):
    """
    This test is responsible for checking that lists of metadata files are being created properly.
    """
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_data_listdir_metadata)
    setup_metadata_files(tmp_path)

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_struct = make_image_object_detection_config()
    assert dataset_struct['image_data']['sources'][0]['remove_image_ids'] == []
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config)
    # At this point we should have the stanzas refactored
    assert_correct_metadata_list(train_list, {1: [1, 11], 11: [21, 31, 41], 2: [2, 12], 12: [22, 32, 42]})


def test_generate_metadata_list_with_image_removal(monkeypatch, tmp_path):
    """
    This test is responsible for checking that lists of metadata files are being created properly
    when we have specified image removal.
    """
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_data_listdir_metadata)
    setup_metadata_files(tmp_path)

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_struct = make_image_object_detection_config(include_image_removal=True)
    assert dataset_struct['image_data']['sources'][0]['remove_image_ids'] == [1]
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config)

    # At this point we should have the stanzas refactored.
    assert_correct_metadata_list(train_list, {11: [21, 31, 41], 2: [2, 12], 12: [22, 32, 42]})


def test_generate_image_metadata_sample_quantity(monkeypatch, tmp_path):
    # The purpose of this test is to confirm the correct metadata file lists are created when
    # the sampling type is a random quantity from each source.
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_data_listdir_metadata)
    setup_metadata_files(tmp_path)

    dataset_struct = make_image_object_detection_config()
    dataset_struct.update(make_sample_stanza("random_quantity", {'seed': 1234, 'count': 1}))

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config)

    # This seed should consistently produce this order.
    assert_correct_metadata_list(train_list, {11: [21, 31, 41], 2: [2, 12]})


def test_generate_image_metadata_sample_fraction(monkeypatch, tmp_path):
    # The purpose of this test is to confirm the correct metadata file lists are created when
    # the sampling type is a random fraction from each source.
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_data_listdir_metadata)
    setup_metadata_files(tmp_path)

    dataset_struct = make_image_object_detection_config()
    dataset_struct.update(make_sample_stanza("random_fraction", {'seed': 7890, 'fraction': 0.3333333333}))

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_config = DatasetConfig.construct(dataset_struct)

    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config)

    # Make sure they are in this order.
    assert_correct_metadata_list(train_list, {1: [1, 11], 12: [22, 32, 42]})


def test_generate_image_metadata_validation_split(monkeypatch, tmp_path):
    # The purpose of this test is to confirm the correct metadata file lists are created when
    # a random fraction validation split is defined in the training config.
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_data_listdir_metadata)
    setup_metadata_files(tmp_path)

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_struct = make_image_object_detection_config()
    dataset_config = DatasetConfig.construct(dataset_struct)

    model_config_dict = utils.make_basic_model_config(add_transforms=True)
    model_config = ModelConfig.construct(data=model_config_dict)
    model_config.validation = {
        "algorithm": "random_fraction",
        "arguments": {
            "seed": 1234,
            "fraction": 0.3333333
        }
    }
    split_config = model_config.get_validation_split_config()

    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config, splitting_config=split_config)

    assert_correct_metadata_list(train_list, {1: [1, 11], 12: [22, 32, 42]})
    assert_correct_metadata_list(val_list, {2: [2, 12], 11: [21, 31, 41]})

    # Now test the "from_file" validation algorithm.
    # Create a dataset config file.
    dataset_path = Path(tmp_path / "dataset_config.json")
    dataset_config.save(dataset_path)

    # Adjust the validation stanza to the "from_file" algorithm.
    model_config.validation = {
        "algorithm": "from_file",
        "arguments": {
            "file_path": str(dataset_path)
        }
    }

    # Get the train and val lists.
    split_config = model_config.get_validation_split_config()
    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config, splitting_config=split_config)

    # Check the train list for correctness.
    assert_correct_metadata_list(train_list, {1: [1, 11], 2: [2, 12], 11: [21, 31, 41], 12: [22, 32, 42]})

    # The val list should be identical to the train list, since it's the same dataset config just loaded from file.
    assert train_list == val_list


def get_labels(meta):
    objects = meta.get('imageProperties', {}).get('objectDetection', {}).get('objects', [])
    return [x['label'] for x in objects]


# The original categories in the sample coco file are as follows.
# LABEL_MAP = {0: 'zero', 1: 'one', 2: 'two', 3: 'three'}

# Since we are doubling, inject the unused ones in the middle.
DOUBLED_LABEL_MAP = {'0': 'zero', '1': 'unused1', '2': 'one', '3': 'unused2', '4': 'two', '5': 'unused3', '6': 'three'}


class LabelDoubler:
    def __call__(self, meta):
        for anno in meta['annotations']:
            anno['category_id'] *= 2

        meta["categories"] = [{"id": str(k), "name": v} for k, v in DOUBLED_LABEL_MAP.items()]

        return meta


def test_generate_image_metadata_preprocessing(monkeypatch, tmp_path):
    # This test checks to see if the metadata preprocessor can modify the labels.
    monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_data_listdir_metadata)
    coco_files = setup_metadata_files(tmp_path)

    lab = Lab(workspace='.', data_root=tmp_path)
    dataset_struct = make_image_object_detection_config()
    dataset_config = DatasetConfig.construct(dataset_struct)

    preprocessors = TransformManager([{"fqcn": "test_data.LabelDoubler"}])

    train_list, val_list = jb_data.generate_metadata_manifests(lab, dataset_config, preprocessors=preprocessors)
    assert len(train_list) == 4
    assert len(val_list) == 0

    # They should all be there and match.
    assert_correct_metadata_list(train_list, {1: [1, 11], 11: [21, 31, 41], 2: [2, 12], 12: [22, 32, 42]})

    # Walk all the raw annotations and make a map of id to new label.
    doubled = {}
    for coco_file in coco_files:
        for anno in coco_file['annotations']:
            doubled[anno['id']] = anno['category_id'] * 2

    for img in train_list:
        for anno in img['annotations']:
            assert anno['category_id'] == doubled[anno['id']]

    assert dataset_config.label_names == DOUBLED_LABEL_MAP


#  _____     _           _
# |_   _|   | |         | |
#   | | __ _| |__  _   _| | __ _ _ __
#   | |/ _` | '_ \| | | | |/ _` | '__|
#   | | (_| | |_) | |_| | | (_| | |
#   \_/\__,_|_.__/ \__,_|_|\__,_|_|
#

def make_basic_data_set_tabular_config():
    return {
        "num_model_classes": 4,
        "timestamp": "never",
        "format_version": DatasetConfig.FORMAT_VERSION,
        "label_names": {"0": "frodo", "1": "sam"},
        "data_type": 'tabular',
        "tabular_data": {
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
            "label_index": 2
        }
    }


def fill_tabular_tempdir(root_dir):
    """
    Creates the sample files to be read and returns the data we should find.
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

    lab = Lab(workspace=Path(tmp_path) / 'myworkspace', data_root=Path(tmp_path) / 'mydataroot')
    dataset_struct = make_basic_data_set_tabular_config()
    dataset_config = DatasetConfig.construct(dataset_struct, Path(tmp_path) / 'myrelative')

    train_list, val_list = jb_data.load_tabular_data(lab, dataset_config)

    # THe sample data is three files each with 4 sample with 2 in each class.
    # THe default validation split is 2.  So 3 * 4 / 2 = 6 per list
    assert len(train_list) == 12
    assert len(val_list) == 0

    # Make sure that every returned value is in the results.
    for data, label in train_list:
        assert correct[int(label)][int(data[0])] == int(data[1])
        del correct[int(label)][int(data[0])]


def test_load_tabular_data_with_sampling(tmp_path):
    correct = fill_tabular_tempdir(tmp_path)
    lab = Lab(workspace=Path(tmp_path) / 'myworkspace', data_root=Path(tmp_path) / 'mydataroot')

    # We only need to test one sample because the sampling core is tested elsewhere.
    dataset_struct = make_basic_data_set_tabular_config()
    dataset_struct.update(make_sample_stanza("random_quantity", {'seed': 1234, 'count': 3}))
    dataset_config = DatasetConfig.construct(dataset_struct, Path(tmp_path) / 'myrelative')

    train_list, val_list = jb_data.load_tabular_data(lab, dataset_config)

    # THe sample data is three files each with 4 sample with 2 in each class.
    # THe default validation split is 2.  So 3 * 4 / 2 = 6 per list
    assert len(train_list) == 6
    assert len(val_list) == 0

    # Now, make sure they are in each one, removing as we go.
    for data, label in train_list:
        assert correct[int(label)][int(data[0])] == int(data[1])
        del correct[int(label)][int(data[0])]

    # At this point we should have three unused entries of each class.
    assert len(correct[0]) == 3
    assert len(correct[1]) == 3


def test_load_tabular_data_with_validation(tmp_path):
    correct = fill_tabular_tempdir(tmp_path)
    lab = Lab(workspace=Path(tmp_path) / 'myworkspace', data_root=Path(tmp_path) / 'mydataroot')

    dataset_struct = make_basic_data_set_tabular_config()
    dataset_config = DatasetConfig.construct(dataset_struct, Path(tmp_path) / 'myrelative')

    model_config_dict = utils.make_basic_model_config()
    model_config = ModelConfig.construct(data=model_config_dict)
    train_list, val_list = jb_data.load_tabular_data(lab,
                                                     dataset_config,
                                                     splitting_config=model_config.get_validation_split_config())

    # THe sample data is three files each with 4 sample with 2 in each class.
    # THe default validation split is 2.  So 3 * 4 / 2 = 6 per list
    assert len(train_list) == 6
    assert len(val_list) == 6

    # Now, make sure they are in each one, removing as we go.
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

def test_sample_list():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(6))
    sampled = jb_data.sample_list(data_list, 2, randomizer)
    for correct, test in zip([0, 3], sampled):
        assert correct == test


def test_sample_list_omit():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(6))
    sampled = jb_data.sample_list(data_list, 4, randomizer, omit=True)
    for correct, test in zip([1, 2, 4, 5], sampled):
        assert correct == test


def test_sampling_random_quantity():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(6))
    sampled = jb_data.sample_data_list(data_list, "random_quantity", {"count": 3}, randomizer)
    for correct, test in zip([0, 3, 4], sampled):
        assert correct == test


def test_sampling_random_fraction():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(6))
    sampled = jb_data.sample_data_list(data_list, "random_fraction", {"fraction": 0.3333333333}, randomizer)
    for correct, test in zip([0, 3], sampled):
        assert correct == test


def test_sampling_round_robin():
    randomizer = random.Random()
    randomizer.seed(1234)
    data_list = list(range(9))
    sampled = jb_data.sample_data_list(data_list, "round_robin", {"groups": 3, "position": 1}, randomizer)
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

    # Order doesn't matter. Just check to make sure that the entries are in the original dict.
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


def mock_generate_image_list(splitting_config=None):
    if splitting_config is not None:
        return ['image_train'], ['image_val']
    else:
        return ['image_train'], []


def mock_load_tabular_data(mc):
    if mc is not None:
        return ['tab_train'], ['tab_val']
    else:
        return ['tab_train'], []


def test_get_label_mapping(tmp_path):
    with utils.set_directory(tmp_path):
        # Binary sample files
        model_name = "tabular_binary_sample"
        model_config = utils.tabular_model_config
        model_manager = ModelManager(model_name)
        model_manager.get_model_config().parent.mkdir(parents=True)

        mc = ModelConfig.construct(data=model_config)
        mc.save(data_path=model_manager.get_model_config())

        dataset_config = utils.tabular_dataset_config
        ds = DatasetConfig.construct(data=dataset_config)
        Path(mc.training_dataset_config_path).parent.mkdir(parents=True)
        ds.save(mc.training_dataset_config_path)

        test_labels = {0: "outer", 1: "inner"}
        test_stanza = {"0": "outer", "1": "inner"}
        assert isinstance(jb_data.convert_dict(test_stanza), dict)

        # Unit tests
        test_source = "training dataset config via model config via model manager"
        func_labels, func_source = jb_data.get_label_mapping(model_manager=model_manager, show_source=True)
        TestCase().assertDictEqual(func_labels, test_labels)
        assert test_source == func_source

        # Test the training output case.

        # Create the training output file.
        to_data = utils.training_output
        training_output = TrainingOutput.construct(data=to_data)
        training_output.options.label_mapping = {str(k): v for k, v in test_stanza.items()}
        model_manager.get_training_out_file().parent.mkdir(parents=True)
        training_output.save(model_manager.get_training_out_file())

        test_source = "training output"
        func_labels, func_source = jb_data.get_label_mapping(model_manager=model_manager, show_source=True)
        TestCase().assertDictEqual(func_labels, test_labels)
        assert test_source == func_source
        model_manager.get_training_out_file().unlink()

        func_labels = jb_data.get_label_mapping(model_config=mc, show_source=True)
        assert func_labels is None

        test_source = "training dataset config"
        func_labels, func_source = jb_data.get_label_mapping(train_config=ds, show_source=True)
        TestCase().assertDictEqual(func_labels, test_labels)
        assert test_source == func_source

        # Now include a model_config with no label mapping and confirm the label mapping still
        # comes from the training dataset config.
        test_source = "training dataset config"
        func_labels, func_source = jb_data.get_label_mapping(model_config=mc, train_config=ds, show_source=True)
        TestCase().assertDictEqual(func_labels, test_labels)
        assert test_source == func_source

        func_labels = jb_data.get_label_mapping(model_config=mc, train_config=ds, show_source=False)
        TestCase().assertDictEqual(func_labels, test_labels)

        # Test retrieving the label mapping from a model config that contains a label mapping.
        test_source = "model config"
        mc.label_mapping = test_stanza
        func_labels, func_source = jb_data.get_label_mapping(model_config=mc, train_config=ds, show_source=True)
        TestCase().assertDictEqual(func_labels, test_labels)
        assert test_source == func_source


def make_sample_manifest(manifest_path, category_list):
    if not manifest_path.exists():
        # With a clean checkout there is no training directory.
        if not manifest_path.parent.exists():
            manifest_path.parent.mkdir(parents=True)

        # Currently, for these tests we really only need categories.
        manifest_data = {
            "images": [],
            "annotations": [],
            "categories": category_list
        }
        with open(manifest_path, 'w') as json_file:
            json.dump(manifest_data, json_file)
        return True
    return False


def test_get_category_list(monkeypatch, tmp_path):
    with utils.set_directory(tmp_path):

        # Grab args
        model_name = "text_detect/dt2/ut"
        model_manager = ModelManager(model_name)

        # Construct a dataset config and make sure the file exists in the tmp_path.
        ds = DatasetConfig.construct(data=utils.text_detect_dataset_config, file_path='test.json')
        ds.save(data_path=ds.file_path)

        data_root = Path(tmp_path)
        test_list_1 = [{'id': 0, 'name': 'HINDI'},
                       {'id': 1, 'name': 'ENGLISH'},
                       {'id': 2, 'name': 'OTHER'}]
        test_list_2 = [{'id': 0, 'name': 'zero'},
                       {'id': 1, 'name': 'one'},
                       {'id': 2, 'name': 'two'},
                       {'id': 3, 'name': 'three'}]

        # Make sample manifest files (if not instantiated already)
        train_manifest_path = model_manager.get_training_data_manifest_path()
        temp_train_manifest = make_sample_manifest(train_manifest_path, test_list_1)
        val_manifest_path = model_manager.get_validation_data_manifest_path()
        temp_val_manifest = make_sample_manifest(val_manifest_path, test_list_1)

        # Make sample coco data file
        monkeypatch.setattr(juneberry.data, 'list_or_glob_dir', mock_list_or_glob_dir)
        coco_data = make_sample_coco([], [])
        coco_path = Path(data_root / 'detectron2-text-detection/val/')
        Path(coco_path).mkdir(parents=True, exist_ok=True)
        with open(coco_path / 'coco_annotations.json', 'w') as json_file:
            json.dump(coco_data, json_file)

        # Train config case
        with TestCase().assertLogs(level='WARNING') as cm:
            category_list, source = jb_data.get_category_list(eval_manifest_path=train_manifest_path,
                                                              train_config=ds,
                                                              data_root=data_root,
                                                              show_source=True)
        assert test_list_2 == category_list
        assert source == "train config"

        # Check for warning message
        TestCase().assertIn(
            "WARNING:juneberry.data:The evaluation category list does not match that of the eval_manifest:",
            cm.output[0])

        # Train manifest case
        category_list, source = jb_data.get_category_list(eval_manifest_path=train_manifest_path,
                                                          model_manager=model_manager,
                                                          train_config=ds,
                                                          data_root=data_root,
                                                          show_source=True)
        assert test_list_1 == category_list
        assert source == "train manifest"

        # Test no mappings case
        try:
            jb_data.get_category_list(eval_manifest_path=Path(""))
            assert False
        except SystemExit:
            assert True

        # Test ModelConfig function
        model_config = utils.make_basic_model_config()
        model_config["preprocessors"] = [{"fqcn": "juneberry.transforms.metadata_preprocessors.ObjectRelabel",
                                          "kwargs": {
                                              "key": "orig",
                                              "labels": {
                                                  "0": "HINDI",
                                                  "1": "ENGLISH",
                                                  "2": "OTHER"
                                              }
                                          }
                                          }]

        model_config_path = Path(tmp_path, "config.json")
        with open(model_config_path, 'w') as out_file:
            json.dump(model_config, out_file, indent=4)
        category_list = jb_data.categories_in_model_config(model_config_path=model_config_path)

        assert test_list_1 == category_list

        # Remove sample manifests from filesystem
        if temp_train_manifest:
            train_manifest_path.unlink()
        if temp_val_manifest:
            val_manifest_path.unlink()
