#! /usr/bin/env python3

"""
Contains utilities for setting up data sets, sampling data, splitting data and making labels.
"""

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

import os
import csv
import sys
import json
import random
import logging

from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import juneberry
import juneberry.pytorch.util as pyt_utils


def setup_data_loaders(train_config, data_set_config, data_manager, no_paging, collate_fn=None):
    """
    Creates the appropriate data loaders based on the training and data set configurations.
    :param train_config: The training config that has transforms and validation options.
    :param data_set_config: The data set config that describes the data.
    :param data_manager: A filesystem data manager that provides paths to data files.
    :param no_paging: Set to true to read all the data at once. Good for small data sets or large memory.
    :param collate_fn: Function that controls how samples are collated into batches.
    :return: training_iterable, prediction_iterable
    """
    if data_set_config.is_image_type():
        logging.info(f"Generating Image file lists...")
        train_list, val_list = generate_image_list(juneberry.DATA_ROOT, data_set_config, train_config,
                                                   data_manager)
    elif data_set_config.is_tabular_type():
        logging.info("Loading tabular data...")
        train_list, val_list = load_tabular_data(train_config, data_set_config)
    else:
        logging.error("We currently do NOT support any data type but IMAGE or TABULAR")
        sys.exit(-1)

    logging.info("Constructing TRAINING data loader.")
    training_loader = pyt_utils.make_data_loader(data_set_config,
                                                 train_list,
                                                 train_config.training_transforms,
                                                 train_config.batch_size,
                                                 no_paging, collate_fn)

    logging.info("Constructing VALIDATION data loader.")
    validation_loader = pyt_utils.make_data_loader(data_set_config,
                                                   val_list,
                                                   train_config.prediction_transforms,
                                                   train_config.batch_size,
                                                   no_paging, collate_fn)

    return training_loader, validation_loader


def listdir_nohidden(path):
    """
    Acts like os.listdir except all hidden files are omitted. 
    :param path: The path to the directory you to be listed out
    :return: array of files paths that do not start with '.' 
    """
    files = []
    for file in os.listdir(path):
        if not file.startswith('.'):
            files.append(file)
    return files


def generate_image_list(data_root, data_set_config, train_config, data_manager):
    """
    Produces the file lists from the data set including sampling and validation splitting.
    Each list is an array of pairs of path and label. So [[<path>, int-label]*]
    NOTE: This can also be used to load sampled testing data.
    :param data_root: The root of the data directory.
    :param data_set_config: The data_set configuration
    :param train_config: OPTIONAL training config for validation split information or NONE for no split.
    :param data_manager: A data manager used to provide the data files.
    :return: train_list, val_list
    """

    logger = logging.getLogger()

    # We store the file lists into the data set structure
    source_list = data_set_config.get_image_sources()

    # Step 1, load all the files into a track
    for data_set in source_list:
        full_path = os.path.join(data_root, data_set['directory'])
        data_set['train'] = listdir_nohidden(full_path)
        data_set['valid'] = []

    # Step 2, perform sampling if they want that on each set
    if data_set_config.has_sampling():
        sample_data_sets(source_list, *data_set_config.get_sampling_config())

    # Step 3, do the validation split on each set
    if train_config is not None and train_config.has_validation_split():
        split_data_sets(source_list, *train_config.get_validation_split_config())

    # Step 4: Aggregate everything into file and label pairs
    # This also looks in the cache for the image size and color space
    train_list = []
    val_list = []
    train_class_counts = defaultdict(int)
    validation_class_counts = defaultdict(int)

    cache_hits = 0
    cache_misses = 0
    for data_set in source_list:
        for filename in data_set['train']:
            relative_path, used_cache = data_manager.check_cache(data_root, data_set['directory'],
                                                                 os.path.join(data_set['directory'], filename))
            train_list.append([os.path.join(data_root, relative_path), data_set['label']])

            # This will be used to establish how many classes we loaded data for
            train_class_counts[data_set['label']] += 1

            if used_cache:
                cache_hits += 1
            else:
                cache_misses += 1

        for filename in data_set['valid']:
            relative_path, used_cache = data_manager.check_cache(data_root, data_set['directory'],
                                                                 os.path.join(data_set['directory'], filename))
            val_list.append([os.path.join(data_root, relative_path), data_set['label']])
            if used_cache:
                cache_hits += 1
            else:
                cache_misses += 1

            validation_class_counts[data_set['label']] += 1

    logger.info(f"Training labels and counts: " +
                ", ".join([f"{k}: {train_class_counts[k]}" for k in sorted(train_class_counts.keys())]))
    logger.info(f"Validation labels and counts: " +
                ", ".join([f"{k}: {validation_class_counts[k]}" for k in sorted(validation_class_counts.keys())]))
    logger.info(f"Total Image count: {len(train_list)} training images, {len(val_list)} validation images")
    logger.info(f"Cache results: cache hits {cache_hits} cache misses {cache_misses}")

    return train_list, val_list


def generate_metadata_list(data_root, data_set_config, train_config):
    """
    Produces file lists of metadata from a glob style path; performs sampling and splitting. Each
    element in the list will be a PosixPath.
    :param data_root: The root of the data directory.
    :param data_set_config: The data_set configuration.
    :param train_config: OPTIONAL training config for validation split information or NONE for no split.
    :return:
    """
    logger = logging.getLogger()

    logger.info(f"Creating metadata file lists...")

    # Retrieve the sources from the data set structure
    source_list = data_set_config.get_image_sources()

    # Step 1, load all the files into a track
    for data_set in source_list:
        data_set['train'] = list(Path(data_root).glob(data_set['directory']))
        data_set['valid'] = []

    # Step 2, perform sampling if they want that on each set
    if data_set_config.has_sampling():
        sample_data_sets(source_list, *data_set_config.get_sampling_config())

    # Step 3, do the validation split on each set
    if train_config is not None and train_config.has_validation_split():
        split_data_sets(source_list, *train_config.get_validation_split_config())

    # Step 4, combine the training / validation contributions from each directory into
    # one list of all training files and one list of all validation files
    train_list = list()
    val_list = list()

    for data_set in source_list:
        train_list.extend(data_set['train'])
        val_list.extend(data_set['valid'])

    logger.info(f"Loaded metadata files from {len(source_list)} directories.")
    logger.info(f"File count: {len(train_list)} training files, {len(val_list)} validation files")

    return train_list, val_list


def load_tabular_data(train_config, data_set_config):
    # TODO: Change this to take validation split config not full training config - COR-384
    logging.info(f"Identifying CSV sources...")
    file_list, label_index = data_set_config.get_resolved_tabular_source_paths_and_labels()

    logging.info(f"Loading data from {len(file_list)} sources...")
    train_dict = load_labeled_csvs(file_list, label_index)

    if data_set_config.has_sampling():
        logging.info(f"Sampling data...")
        sample_labeled_tabular_data(train_dict, *data_set_config.get_sampling_config())

    val_dict = {}
    if train_config is not None:
        logging.info(f"Pulling out validation split...")
        val_dict = split_labeled_tabular_data(train_dict, *train_config.get_validation_split_config())

    # Summarize what we have

    logging.info("Training labels and counts: " +
                 ", ".join([f"{k}: {len(train_dict[k])}" for k in sorted(train_dict.keys())]))
    logging.info("Validation labels and counts: " +
                 ", ".join([f"{k}: {len(val_dict[k])}" for k in sorted(val_dict.keys())]))

    # Now, turn this into flat list of data and labels.
    train_list = flatten_dict_to_pairs(train_dict)
    val_list = flatten_dict_to_pairs(val_dict)

    return train_list, val_list


def get_num_classes(data_config_files):
    """
    Determines the number of classes in a model from the Data Config files.
    If the number of classes is missing from the config, an error will be logged and the program will exit.
    If the config files disagree on the number of classes, an warning will be logged and the highest number of classes
    will be returned.
    :param data_config_files: Array of data set config files
    """
    num_classes = []
    for config_file in data_config_files:
        with open(config_file) as file:
            config = json.load(file)
        if 'numModelClasses' not in config.keys():
            logging.error(f"Config file {config_file} does not specify the number of classes.")
            exit(-1)
        else:
            num_classes.append(config['numModelClasses'])
            # Checks if config is a data config and if max label value in the dataset is not smaller than num_classes
            if 'data' in config.keys() and num_classes[-1] <= max([data_set['label'] for data_set in config['data']]):
                # remember that labels are indexed at 0, so if largest label == num_classes, the label is too large
                logging.error(f"Config file {config_file} has a label beyond the number of classes specified.")
    # Checks if all config files specify the same number of classes
    if num_classes.count(num_classes[0]) != len(num_classes):
        logging.warning(f"Config files do NOT specify the same number of classes. The largest number of "
                        f"{max(num_classes)} will be used.")

    return max(num_classes)


#  ____                        _ _
# / ___|  __ _ _ __ ___  _ __ | (_)_ __   __ _
# \___ \ / _` | '_ ` _ \| '_ \| | | '_ \ / _` |
#  ___) | (_| | | | | | | |_) | | | | | | (_| |
# |____/ \__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
#                       |_|              |___/


def sample_data_sets(sources_list, sampling_algo: str, sampling_args, randomizer):
    for data_source in sources_list:
        data_source['train'] = sample_data_list(data_source['train'], sampling_algo, sampling_args, randomizer)


def sample_labeled_tabular_data(labeled_tabular, sampling_algo: str, sampling_args, randomizer):
    for key in labeled_tabular:
        labeled_tabular[key] = sample_data_list(labeled_tabular[key], sampling_algo, sampling_args, randomizer)


def sample_data_list(data_list, sampling_algo: str, sampling_args, randomizer):
    """
    Returns a sample (subset) of the source list.
    :param data_list: The data list to sample.
    :param sampling_algo: The sample algorithm.
    :param sampling_args: The arguments to the sampling algorithm.
    :param randomizer: Randomizer if needed
    :return: The sampled (reduced) data list.
    """
    if sampling_algo == "randomFraction":
        count = round(len(data_list) * float(sampling_args['fraction']))
        if count == 0:
            logging.error("Fraction is less than 1, setting quantity to 1")
            count = 1
        data_list = randomizer.sample(data_list, count)
    elif sampling_algo == "randomQuantity":
        count = sampling_args['count']
        data_list = randomizer.sample(data_list, count)
    elif sampling_algo == "roundRobin":  # Randomize list, split list into groups, return position item from each group
        groups = sampling_args["groups"]
        position = sampling_args["position"]
        if position < 0 or position > (groups - 1):
            logging.error("position is less than zero or larger than the number size of a group")
            sys.exit(-1)
        randomized_indexs = [x for x in range(len(data_list))]
        select_indexs = [y for y in range(position, len(data_list), groups)]
        randomizer.shuffle(randomized_indexs)
        data_list = [data_list[randomized_indexs[i]] for i in select_indexs]
    elif sampling_algo == "none":
        pass
    else:
        logging.error("Unknown sampling algorithm. Adding ZERO files.")
        data_list = []

    return data_list


def make_balanced_labeled_list(data_set: List[Tuple[Any, int]], max_count: int, randomizer: random.Random) -> list:
    """
    Takes a list of data with labels (list(Any, int)) and returns a list where each
    class is balanced size-wise where the number of elements in each class does not
    exceed max_count. If max_count is -1, then the smallest class element count found
    is used. Returns the updated set.
    :param data_set: Input data which is list of (str,int)
    :param max_count: The maximum elements per class or -1 to use existing size.
    :param randomizer: Randomizer used when sampling the categories.
    :return: List[Tuple[Any, int]]
    """
    labeled_dict = labeled_pairs_to_labeled_dict(data_set)
    balanced_dict = make_balanced_labeled_dict(labeled_dict, max_count, randomizer)
    return flatten_dict_to_pairs(balanced_dict)


def make_balanced_labeled_dict(data_set: Dict[int, List[Any]], max_count: int, randomizer: random.Random) -> dict:
    """
    Takes a dict of data with labels (int:list()) and returns a dict where each
    class is balanced size-wise where the number of elements in each class does not
    exceed max_count. If max_count is -1, then the smallest class element count found
    is used.  The number of elements per class is returned.
    :param data_set: Input data which is list of (str,int)
    :param max_count: The maximum elements per class or -1 to use existing size.
    :param randomizer: Randomizer used when sampling the categories.
    :return: Dict[int, List[Any]]
    """

    # Find smallest count
    smallest_count = min([len(v) for k, v in data_set.items()])
    if max_count == -1:
        max_count = smallest_count

    if smallest_count < max_count:
        max_count = smallest_count

    # Make a new dict with that subset
    new_dict = {}
    for k, v in data_set.items():
        new_dict[k] = randomizer.sample(v, max_count)

    return new_dict


#  ____        _ _ _   _   _
# / ___| _ __ | (_) |_| |_(_)_ __   __ _
# \___ \| '_ \| | | __| __| | '_ \ / _` |
#  ___) | |_) | | | |_| |_| | | | | (_| |
# |____/| .__/|_|_|\__|\__|_|_| |_|\__, |
#       |_|                        |___/


def split_labeled_tabular_data(labeled_tabular, splitter_algo: str, splitter_args, randomizer):
    split_data = {}
    for key in labeled_tabular:
        split_data[key] = split_list(labeled_tabular[key], splitter_algo, splitter_args, randomizer)
    return split_data


def split_data_sets(data_set_list, splitter_algo: str, splitter_args, randomizer):
    for data_set in data_set_list:
        data_set['valid'] = split_list(data_set['train'], splitter_algo, splitter_args, randomizer)


def split_list(source_list, splitter_algo: str, splitter_args, randomizer):
    """
    Removes a set of items from the source list and returns them in a new list based on the
    splitter algorithm and splitter_args;
    :param source_list: The source list to split.  It is MODIFIED.
    :param splitter_algo: The algorithm to use to split out the data
    :param splitter_args: The arguments to the splitting algorithm.
    :param randomizer: Randomizer if needed
    :return: The elements removed out of the source list.
    """
    shard = []
    if splitter_algo == 'randomFraction':
        count = round(len(source_list) * float(splitter_args['fraction']))
        val_indexes = randomizer.sample(range(0, len(source_list)), count)
        val_indexes = sorted(val_indexes, reverse=True)

        for i in val_indexes:
            shard.append(source_list[i])
            del (source_list[i])
    else:
        logging.error(f"Unknown validation algorithm '{splitter_algo}', not splitting list.")

    return shard


# ___  ____
# |  \/  (_)
# | .  . |_ ___  ___
# | |\/| | / __|/ __|
# | |  | | \__ \ (__
# \_|  |_/_|___/\___|

def flatten_dict_to_pairs(labeled_data) -> List[list]:
    """
    Takes as input a dict of key -> list and turns it into a single unified list of pairs
    of entry and label.  Thus
    { 'a': ['frodo', 'sam'], 'b': ['merry', 'pippin' ] }
    becomes
    [ ['frodo', 'a], ['sam', 'a'], ['merry', 'b'], ['pippin', 'b'] ]

    :param labeled_data: The dict of labeled data.
    :return: Combined list of all values with parallel list of all matching labels.
    """
    result = []
    for label, values in labeled_data.items():
        for item in values:
            result.append([item, label])
    return result


def labeled_pairs_to_labeled_dict(pair_list):
    """
    Converts a list of pairs of [value,key] to a dict of key -> list(value)
    :param pair_list: List of pairs of [value,key]
    :return: Dict of key -> list(values)
    """
    labeled_dict = defaultdict(list)
    for v, k in pair_list:
        labeled_dict[k].append(v)
    return labeled_dict


def load_labeled_csvs(file_list, label_index):
    """
    Reads a series of CSVs into a dict of label -> data rows
    :param file_list: A list of files to read
    :param label_index: The index that contains the label
    :return: Dict of label -> data rows
    """
    data = defaultdict(list)
    first_header = None
    for file_path in file_list:
        logging.info(f"Loading - {file_path}... ")
        with open(file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            if first_header is None:
                first_header = next(reader)
            else:
                header = next(reader)
                # All headers must be identical as we don't do ANY mapping.
                if header != first_header:
                    logging.error(f"Header fields of CSV entries do NOT match. EXITING! First header from "
                                  f"{file_list[0]} doesn't match header in {file_path}")
                    sys.exit(-1)

            row_idx = 0
            for row in reader:
                if len(row) <= label_index:
                    logging.error(f"Row has length {len(row)} but we asked for labels that are "
                                  f"column {label_index} row {row_idx}")
                else:
                    label = int(row[label_index])
                    del row[label_index]
                    data[label].append(row)
                row_idx += 1

    return data
