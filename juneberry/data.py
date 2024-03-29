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

"""
Contains utilities for setting up data sets, sampling data, splitting data and making labels.
"""

from collections import defaultdict
import copy
import csv
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple, Union

from juneberry.config.coco_anno import CocoAnnotations
import juneberry.config.coco_utils as coco_utils
from juneberry.config.dataset import DatasetConfig, SamplingAlgo
from juneberry.config.model import ModelConfig, SplittingConfig
from juneberry.config.training_output import TrainingOutput
import juneberry.filesystem as jb_fs
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
from juneberry.transforms.transform_manager import TransformManager, LabeledTransformManager

logger = logging.getLogger(__name__)


# NOTE Dataspec is the new name for Dataset
def dataspec_to_manifests(lab: Lab, dataset_config, *,
                          splitting_config: SplittingConfig = SplittingConfig(None, None, None),
                          preprocessors: TransformManager = None):
    """
    Creates the appropriate data manifests (file path/label) based on the training and dataset configurations.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The dataset config that describes the data.
    :param splitting_config: OPTIONAL The SplittingConfig that is used to split the data into
    a separate subset. Specify None for no splitting.
    :param preprocessors: OPTIONAL - A TransformManager to be applied to each entry before returning.
    :return: This function returns data manifest and validation manifest
    """
    # In case folks pass in None for the config
    if splitting_config is None:
        splitting_config = SplittingConfig(None, None, None)

    if dataset_config.is_image_type():
        # If it's an object detection task, build lists of metadata files
        if dataset_config.is_object_detection_task():
            logger.info(f"Generating metadata lists...")
            loader = CocoMetadataMarshal(lab, dataset_config, splitting_config=splitting_config,
                                         preprocessors=preprocessors)
            return loader.load()

        # If it's not object detection but a list of images
        else:
            logger.info(f"Generating image lists...")
            loader = ListMarshall(lab, dataset_config, splitting_config=splitting_config, preprocessors=preprocessors)
            return loader.load()

    elif dataset_config.is_tabular_type():
        logger.info("Loading tabular data...")
        return load_tabular_data(lab, dataset_config, splitting_config=splitting_config)

    else:
        logger.error("We currently do NOT support any data type but IMAGE or TABULAR.")
        sys.exit(-1)


def generate_image_manifests(lab: Lab, dataset_config: DatasetConfig, *,
                             splitting_config: SplittingConfig = SplittingConfig(None, None, None),
                             preprocessors: TransformManager = None):
    """
    Produces the file lists from the dataset, including sampling and validation splitting.
    Each list is an array of pairs of path and label. So [[<path>, int-label]*]
    NOTE: This can also be used to load sampled testing data.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The dataset configuration
    :param splitting_config: OPTIONAL The SplittingConfig that is used to split the data into
    a separate subset. Specify None for no splitting.
    :param preprocessors: OPTIONAL - A TransformManager to be applied to each entry before returning.
    :return: train_list, val_list
    """
    if splitting_config is None:
        splitting_config = SplittingConfig(None, None, None)

    loader = ListMarshall(lab, dataset_config, splitting_config=splitting_config, preprocessors=preprocessors)
    loader.load()
    return loader.train, loader.val


def generate_metadata_manifests(lab: Lab, dataset_config: DatasetConfig, *,
                                splitting_config: SplittingConfig = SplittingConfig(None, None, None),
                                preprocessors: TransformManager = None):
    """
    Produces file lists of metadata from a glob style path; performs sampling and splitting. Each
    element in the list will be a PosixPath.
    NOTE: The dataset labels are updated from the metadata
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The dataset configuration.
    :param splitting_config: OPTIONAL The SplittingConfig that is used to split the data into
    a separate subset. Specify None for no splitting.
    :param preprocessors: OPTIONAL - A TransformManager to be applied to each entry before returning.
    :return: Training file list and validation list
    """
    if splitting_config is None:
        splitting_config = SplittingConfig(None, None, None)

    loader = CocoMetadataMarshal(lab, dataset_config, splitting_config=splitting_config, preprocessors=preprocessors)
    loader.load()
    return loader.train, loader.val


def make_split_metadata_manifest_files(lab: Lab,
                                       dataset_config: DatasetConfig,
                                       model_config: ModelConfig,
                                       model_manager: ModelManager,
                                       ):
    """
    Some object detection systems want coco style annotation files to ingest so we use our
    manifests to create some temporary coco annotation files that are the nice and accurate lists of the
    images and annotations.
    :param lab: The lab in which we are executing.
    :param dataset_config: The dataset config.
    :param model_config: The model config with split config and transformers.
    :param model_manager: The model manager of where to save the files.
    :return: The manifests that were converted to coco style annotations
    """
    # Make the file lists.
    train_meta, split_meta = generate_metadata_manifests(
        lab,
        dataset_config,
        splitting_config=model_config.get_validation_split_config(),
        preprocessors=TransformManager(model_config.preprocessors))

    # Convert out COCO like intermediate list format into pure coco file.
    label_names = get_label_mapping(model_manager=model_manager, model_config=model_config, train_config=dataset_config)
    train_coco_meta = coco_utils.convert_jbmeta_to_coco(train_meta, label_names)
    split_coco_meta = coco_utils.convert_jbmeta_to_coco(split_meta, label_names)

    # Serialize
    train_path = model_manager.get_training_data_manifest_path()
    jb_fs.save_json(train_coco_meta, train_path)

    split_path = model_manager.get_validation_data_manifest_path()
    jb_fs.save_json(split_coco_meta, split_path)

    # TODO: JB should output manifests
    logger.info(f"Saving training data manifest: {train_path}")
    logger.info(f"Saving validation data manifest: {split_path}")

    return train_meta, split_meta


def make_eval_manifest_file(lab: Lab, dataset_config: DatasetConfig,
                            model_config: ModelConfig,
                            model_manager: ModelManager,
                            use_train_split=False, use_val_split=False):
    """
    Creates a single manifest file based on the transforms.
    :param lab: The lab in which we are executing.
    :param dataset_config: The dataset config.
    :param model_config: The model config with split config and transformers.
    :param model_manager: The model manager of where to save the files.
    :param use_train_split: Boolean indicating if the training portion of the split
    dataset should be used for evaluation.
    :param use_val_split: Boolean indicating if the validation portion of the split
    dataset should be used for evaluation.
    :return: The evaluation list and the coco formatted file.
    """

    splitting_config = None
    if use_train_split or use_val_split:
        logger.info(f"Splitting the dataset according to the model's validation split instructions.")
        splitting_config = model_config.get_validation_split_config()

    eval_list, split = generate_metadata_manifests(
        lab,
        dataset_config,
        splitting_config=splitting_config,
        preprocessors=TransformManager(model_config.preprocessors))

    if use_train_split:
        logger.info("Evaluating using ONLY the training portion of the split data.")

    elif use_val_split:
        logger.info("Evaluating using ONLY the validation portion of the split data.")
        eval_list = split

    output_path = model_manager.get_eval_dir_mgr(str(dataset_config.file_path)).get_manifest_path()

    label_names = get_label_mapping(model_manager=model_manager, model_config=model_config, train_config=dataset_config)
    coco_style = coco_utils.convert_jbmeta_to_coco(eval_list, label_names)
    jb_fs.save_json(coco_style, output_path)

    logger.info(f"Saving evaluation data manifest: {output_path}")

    return eval_list, coco_style


#  ____        _                 _   __  __                _           _
# |  _ \  __ _| |_ __ _ ___  ___| |_|  \/  | __ _ _ __ ___| |__   __ _| |
# | | | |/ _` | __/ _` / __|/ _ \ __| |\/| |/ _` | '__/ __| '_ \ / _` | |
# | |_| | (_| | || (_| \__ \  __/ |_| |  | | (_| | |  \__ \ | | | (_| | |
# |____/ \__,_|\__\__,_|___/\___|\__|_|  |_|\__,_|_|  |___/_| |_|\__,_|_|

"""
Sources List
The sources_list is a copy of the "image_sources" entry from the dataset config with the addition 
of two fields: "train" and "valid". During loading the directory is iterated or globbed and the contents
are placed into the "train" field. Later, during splitting, some of the actual contents are moved 
into the val field.

So after load the sources list (with a data root of /var/dataroot might look like:

[
    {
        "directory": "my_images",
        "label": 42
        "train": [ '/var/dataroot/image1.png', '/var/dataroot/image3.png' ]
        "valid": [ '/var/dataroot/image2.png' ]
    }
]

"""


class DatasetMarshal:
    """
    Dataset Marshals are used to load and process various types of data configs into "datasets"
    that are basically lists of data items (pytorch Datasets that support __len__ and __getitem__)
    that can be fed to data loaders.
    """

    def __init__(self, lab, dataset_config: DatasetConfig, *,
                 splitting_config: SplittingConfig = SplittingConfig(None, None, None),
                 preprocessors: TransformManager = None):
        """
        Constructs the base object for getting the dataset specified by the data source list
        and loading it into memory.
        :param lab: The Juneberry Lab in which this operation occurs.
        :param dataset_config: The dataset config that describes the data to load.
        :param splitting_config: OPTIONAL The SplittingConfig that is used to split the data into a separate
        subset. Specify None for no splitting.
        :param preprocessors: OPTIONAL - A TransformManager to be applied to each entry before returning.
        """
        self.lab = lab
        self.ds_config = dataset_config

        # Arrays the individual entries
        self.train = []
        self.val = []

        self._splitting_config = splitting_config
        self._preprocessors = preprocessors

        self._source_list = []
        self._sampling_configs = [dataset_config.get_sampling_config(), dataset_config.get_sampling_config()]

    def load(self):
        """
        Besides construction this is the primary method. This invokes all the steps to load, preprocess,
        sample, split, etc. the data and returns two "datasets" that support __len__ and __getitem__.
        :return: The training dataset and the validation dataset.
        """
        self.setup()
        self.make_source_list()
        self.validation_from_file()
        self.expand()
        # TODO: Preprocessing happens IN expand for metadata. Do we really want this step?
        #       Maybe preprocessing IS expand??
        self.preprocess()
        self.sample()
        self.split()
        self.merge_sources()
        if len(self.train) == 0:
            logger.warning("Training dataset is length 0.")
        if len(self.val) == 0:
            logger.warning("Validation dataset is length 0.")
        return self.train, self.val

    #  _____      _                 _               ____       _       _
    # | ____|_  _| |_ ___ _ __  ___(_) ___  _ __   |  _ \ ___ (_)_ __ | |_ ___
    # |  _| \ \/ / __/ _ \ '_ \/ __| |/ _ \| '_ \  | |_) / _ \| | '_ \| __/ __|
    # | |___ >  <| ||  __/ | | \__ \ | (_) | | | | |  __/ (_) | | | | | |_\__ \
    # |_____/_/\_\\__\___|_| |_|___/_|\___/|_| |_| |_|   \___/|_|_| |_|\__|___/

    def setup(self):
        pass

    def make_source_list(self):
        add_image_data_sources(self.lab, self.ds_config, self._source_list, 'train')

    def validation_from_file(self):
        if self._splitting_config.algo == 'from_file':
            logger.info(f"Found 'fromFile' validation splitter.  Loading {self._splitting_config.args['file_path']}.")
            new_sampling = validation_from_file(self.lab, self._source_list, self._splitting_config)

            # Now update our configs.  We have already split so we set it to none,
            # but now get the new sampling config.
            self._splitting_config = SplittingConfig(None, None, None)
            self._sampling_configs[1] = new_sampling

    def expand(self):
        pass

    def preprocess(self):
        if self._preprocessors is not None:
            logger.info(f"Applying ({len(self._preprocessors)}) preprocessors...")
            apply_function(self._source_list, self._preprocessors)

    def sample(self):
        if self._sampling_configs[0].algo is not None:
            logger.info(f"SAMPLING {self._sampling_configs[0]}")
            sample_data_sets(self._source_list, *self._sampling_configs[0], 'train')
        if self._sampling_configs[1].algo is not None:
            logger.info(f"SAMPLING {self._sampling_configs[1]}")
            sample_data_sets(self._source_list, *self._sampling_configs[1], 'valid')

    def split(self):
        if self._splitting_config.algo is not None:
            logger.info(f"Applying splitting {self._splitting_config}.")
            split_data_sets(self._source_list, *self._splitting_config)

    def merge_sources(self):
        pass


class ListMarshall(DatasetMarshal):
    def __init__(self, lab, dataset_config: DatasetConfig, *,
                 splitting_config: SplittingConfig = SplittingConfig(None, None, None),
                 preprocessors: SplittingConfig = None):
        super().__init__(lab, dataset_config, splitting_config=splitting_config, preprocessors=preprocessors)

    def merge_sources(self):
        train_class_counts = defaultdict(int)
        validation_class_counts = defaultdict(int)

        for data_set in self._source_list:
            for filename in data_set['train']:
                self.train.append([str(filename), data_set['label']])

                # This will be used to establish how many classes we loaded data for
                train_class_counts[data_set['label']] += 1

            for filename in data_set['valid']:
                self.val.append([str(filename), data_set['label']])

                validation_class_counts[data_set['label']] += 1

        logger.info(f"Labels and counts in main dataset: " +
                    ", ".join([f"{k}: {train_class_counts[k]}" for k in sorted(train_class_counts.keys())]))
        logger.info(f"Labels and counts in split dataset: " +
                    ", ".join([f"{k}: {validation_class_counts[k]}" for k in sorted(validation_class_counts.keys())]))
        logger.info(f"Total Image count: {len(self.train)} main dataset images, {len(self.val)} split dataset images")


class CocoMetadataMarshal(DatasetMarshal):
    """
    A data marshall that loads data from coco files and places them into image-centric lists of entries.
    """

    def __init__(self, lab, dataset_config: DatasetConfig, *,
                 splitting_config: SplittingConfig = SplittingConfig(None, None, None),
                 preprocessors: TransformManager = None):
        super().__init__(lab, dataset_config, splitting_config=splitting_config, preprocessors=preprocessors)

    def expand(self):
        # Expand the files names into actual metadata files
        logger.info(f"Expanding metadata files...")

        for source in self._source_list:
            for dataset_type in ['train', 'valid']:
                new_values = []
                for x in source[dataset_type]:
                    self._load_metadata_file(x, new_values, source.remove_image_ids)
                source[dataset_type] = new_values
        logger.info("...done expanding metadata.")

    def preprocess(self):
        # We do our pre-processing during loading so we have nothing to do.
        pass

    def merge_sources(self):
        for data_set in self._source_list:
            self.train.extend(data_set['train'])
            self.val.extend(data_set['valid'])

        logger.info(f"Loaded metadata files from {len(self._source_list)} sources.")
        self._summarize_set("train", self.train)
        self._summarize_set("val", self.val)
        logger.info(f"LabelNames: {self.ds_config.label_names}")

    def _load_metadata_file(self, filepath: str, new_values: list, remove_image_ids: list):
        # NOTE on categories
        # Priority is preprocessors supplied ones > dataset > annotations
        # Preprocessor - if supplied we just use those totally.  They win
        # Data set is then used if NO preprocessor configs
        # Then we add any more file annotation files

        logger.info(f"...loading: {filepath}...")
        data = CocoAnnotations.load(filepath)
        if self._preprocessors is not None:
            logger.info(f"...applying ({len(self._preprocessors)}) preprocessors...")
            orig_cats = copy.deepcopy(data.categories)
            data = self._preprocessors(data)
            if orig_cats != data.categories:
                logger.info(f"......changing categories because preprocessor changed them.")
                # The categories were changed by the preprocessors. They win out over all others
                # Categories is a list of id, name
                # Last preprocessor of all files wins
                # NOTE: This might not be a good idea to do it every time depending on how the
                # preprocessors are written, but we'll try this for a while.
                # TODO: Fix the dichotomy with str vs int labels
                str_labels = {str(x['id']): x['name'] for x in data.categories}
                int_labels = {int(x['id']): x['name'] for x in data.categories}
                self.ds_config.label_names = str_labels
                self.ds_config.update_label_names(int_labels)

        # Now that we have the file loaded let's load the values
        helper = coco_utils.COCOImageHelper(data)
        if remove_image_ids:
            [helper.remove_image(img_id) for img_id in remove_image_ids]
        new_values.extend(helper.to_image_list())

    @staticmethod
    def _summarize_set(label_name, data_list):
        counts = defaultdict(int)
        anno_count = 0
        for item in data_list:
            anno_count += len(item['annotations'])
            for anno in item['annotations']:
                counts[anno['category_id']] += 1

        msg = ", ".join([f"{k}:{counts[k]}" for k in sorted(counts.keys())])
        logger.info(f"Metadata set '{label_name}' has {len(data_list)} images with "
                    f"{anno_count} annotations distributed as {msg}")


#  _____     _           _
# |_   _|_ _| |__  _   _| | __ _ _ __
#   | |/ _` | '_ \| | | | |/ _` | '__|
#   | | (_| | |_) | |_| | | (_| | |
#   |_|\__,_|_.__/ \__,_|_|\__,_|_|

def load_tabular_data_binned(lab, dataset_config: DatasetConfig, *, splitting_config: SplittingConfig = None):
    """
    Loads the tabular data set into a dict of label -> rows
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The data set config.
    :param splitting_config: Optional splitting config.
    :return: train_list, val_list
    """
    logger.info(f"Identifying CSV sources...")
    file_list, label_index = dataset_config.get_resolved_tabular_source_paths_and_labels(lab)

    logger.info(f"Loading data from {len(file_list)} sources...")
    binned_data = load_labeled_csvs(file_list, label_index)

    if dataset_config.has_sampling():
        logger.info(f"Sampling data...")
        sample_labeled_tabular_data(binned_data, *dataset_config.get_sampling_config())

    return binned_data


# TODO: Convert the tabular loader to use the new parser/loader
def load_tabular_data(lab, dataset_config: DatasetConfig, *, splitting_config: SplittingConfig = None):
    """
    Loads the tabular data set
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The data set config.
    :param splitting_config: Optional splitting config.
    :return: train_list, val_list
    """
    train_dict = load_tabular_data_binned(lab, dataset_config, splitting_config=splitting_config)

    val_dict = {}
    if splitting_config is not None:
        if splitting_config.algo == 'from_file':
            # For from_file it should point to an entire dataset config. We then load it (without splitting)
            # as the val_dict
            file_path = splitting_config.args['file_path']
            logger.info(f"Loading validation from file {file_path}...")
            val_config = DatasetConfig.load(file_path, Path(file_path).parent)
            val_dict = load_tabular_data_binned(lab, val_config, splitting_config=None)
        else:
            logger.info(f"Pulling out validation split...")
            val_dict = split_labeled_tabular_data(train_dict, *splitting_config)

    # Summarize what we have
    logger.info("Training labels and counts: " +
                ", ".join([f"{k}: {len(train_dict[k])}" for k in sorted(train_dict.keys())]))
    logger.info("Validation labels and counts: " +
                ", ".join([f"{k}: {len(val_dict[k])}" for k in sorted(val_dict.keys())]))

    # Now, turn this into flat list of data and labels.
    train_list = flatten_dict_to_pairs(train_dict)
    val_list = flatten_dict_to_pairs(val_dict)

    if len(train_list) == 0:
        logger.error("Training dataset is length 0.")
        sys.exit(1)
    if len(val_list) == 0:
        logger.warning("Validation dataset is length 0.")

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
        config = jb_fs.load_file(config_file)
        if 'num_model_classes' not in config.keys():
            logger.error(f"Config file {config_file} does not specify the number of classes.")
            exit(-1)
        else:
            num_classes.append(config['num_model_classes'])
            # Checks if config is a data config and if max label value in the dataset is not smaller than num_classes
            if 'data' in config.keys() and num_classes[-1] <= max([data_set['label'] for data_set in config['data']]):
                # remember that labels are indexed at 0, so if largest label == num_classes, the label is too large
                logger.error(f"Config file {config_file} has a label beyond the number of classes specified.")
    # Checks if all config files specify the same number of classes
    if num_classes.count(num_classes[0]) != len(num_classes):
        logger.warning(f"Config files do NOT specify the same number of classes. The largest number of "
                       f"{max(num_classes)} will be used.")

    return max(num_classes)


#  ____
# |  _ \ _ __ ___ _ __  _ __ ___   ___ ___  ___ ___
# | |_) | '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|
# |  __/| | |  __/ |_) | | | (_) | (_|  __/\__ \__ \
# |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/
#                |_|

def apply_function(source_list, fn):
    """
    Applies the function to each entry in the list. If the function returns None,
    the item is removed. Meaning, we don't return None elements.
    :param source_list: The source list
    :param fn: The function to apply
    :return:
    """
    # TODO: This doesn't pass in the labels and can't relabel the data because the label
    #  is associated with the source. We would need to flatten the entire list then relabel
    #  so the preprocess would need to happen after merge and this is NOT the right place for
    #  that.
    for source in source_list:
        for dataset_type in ['train', 'valid']:
            new_list = [fn(x) for x in source[dataset_type]]
            source[dataset_type] = [x for x in new_list if x is not None]


# TODO: Nothing calls this, but the functionality is in the "expand" call COCO call.
def load_preprocess_metadata(source_list, preprocessors=None):
    """
    This function loads the metadata file (as a file) and then performs preprocessing
    on the metadata.  The data is then stored in the source list instead of the path.
    :param source_list: The sources list to preprocess.
    :param preprocessors: The preprocessors to use.
    :return:
    """

    # Construct a transform manager
    transformers = TransformManager(preprocessors)

    # Reset the data list, load each file, preprocess and add if not None.
    for source in source_list:
        for dataset_type in ['train', 'valid']:
            input_paths = source[dataset_type]
            source[dataset_type] = []
            for entry in input_paths:
                data = jb_fs.load_file(entry)

                if preprocessors is not None:
                    data = transformers(data)

                if data is not None:
                    source[dataset_type].append(data)


#  ____                        _ _
# / ___|  __ _ _ __ ___  _ __ | (_)_ __   __ _
# \___ \ / _` | '_ ` _ \| '_ \| | | '_ \ / _` |
#  ___) | (_| | | | | | | |_) | | | | | | (_| |
# |____/ \__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
#                       |_|              |___/


def sample_data_sets(sources_list, sampling_algo: str, sampling_args, randomizer, dataset_type='train'):
    before = []
    after = []
    for data_source in sources_list:
        before.append(len(data_source[dataset_type]))
        data_source[dataset_type] = sample_data_list(data_source[dataset_type], sampling_algo, sampling_args,
                                                     randomizer)
        after.append(len(data_source[dataset_type]))

    logger.info(f"Sampled {dataset_type} data {sampling_algo}, {sampling_args}, sizes {before} -> {after}")


def sample_labeled_tabular_data(labeled_tabular, sampling_algo: str, sampling_args, randomizer):
    for key in labeled_tabular:
        labeled_tabular[key] = sample_data_list(labeled_tabular[key], sampling_algo, sampling_args, randomizer)


def sample_list(data_list, count: int, randomizer, omit: bool = False):
    # Generate a list of indexes and use to keep or omit
    indexes = list(range(len(data_list)))

    # NOTE: Sample does NOT provide them in order, so we use a set to identify
    # the indexes then remove or keep accordingly
    indexes = set(randomizer.sample(indexes, count))
    result = []
    if omit:
        # OMIT case - scan and discard things from index list
        for idx, item in enumerate(data_list):
            if idx in indexes:
                indexes.remove(idx)
            else:
                result.append(item)
        logger.info(f"OMIT case: {len(data_list)}, {len(result)}")
    else:
        # KEEP case - scan and keep things in the index list
        for idx, item in enumerate(data_list):
            if idx in indexes:
                indexes.remove(idx)
                result.append(item)
        logger.info(f"KEEP case: {len(data_list)}, {len(result)}")
    return result


def sample_data_list(data_list, algo: str, args, randomizer):
    """
    Returns a sample (subset) of the source list.
    :param data_list: The data list to sample.
    :param algo: The sample algorithm.
    :param args: The arguments to the sampling algorithm.
    :param randomizer: Randomizer if needed
    :return: The sampled (reduced) data list.
    """
    if algo == SamplingAlgo.RANDOM_FRACTION:
        fraction = args['fraction']
        if fraction > 1:
            logger.error(f"The random fraction must be less than 1. It is {fraction}. Exiting.")
            sys.exit(-1)
        count = round(len(data_list) * float(args['fraction']))
        if count == 0:
            logger.error("Fraction is less than 1, setting quantity to 1")
            count = 1
        if count < len(data_list):
            data_list = sample_list(data_list, count, randomizer, args.get('omit', False))

    elif algo == SamplingAlgo.RANDOM_QUANTITY:
        count = args['count']
        if count < len(data_list):
            data_list = sample_list(data_list, count, randomizer, args.get('omit', False))

    # Randomize list, split list into groups, return position item from each group
    elif algo == SamplingAlgo.ROUND_ROBIN:
        groups = args["groups"]
        position = args["position"]
        if position < 0 or position > (groups - 1):
            logger.error("position is less than zero or larger than the number size of a group")
            sys.exit(-1)
        randomized_indexs = [x for x in range(len(data_list))]
        select_indexs = [y for y in range(position, len(data_list), groups)]
        randomizer.shuffle(randomized_indexs)
        data_list = [data_list[randomized_indexs[i]] for i in select_indexs]
    elif algo == SamplingAlgo.NONE or algo is None:
        pass
    else:
        logger.error(f"Unknown sampling algorithm: {algo} Returning all files.")

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


def split_data_sets(dataset_list, splitter_algo: str, splitter_args, randomizer):
    for dataset in dataset_list:
        before = [len(dataset['train']), len(dataset['valid'])]
        dataset['valid'] = split_list(dataset['train'], splitter_algo, splitter_args, randomizer)
        after = [len(dataset['train']), len(dataset['valid'])]
        logger.info(f"...split (train/valid) for {dataset['label']} from {before} to {after}")


def split_list(source_list, algo: str, args, randomizer):
    """
    Removes a set of items from the source list and returns them in a new list based on the
    splitter algorithm and splitter_args;
    :param source_list: The source list to split.  It is MODIFIED.
    :param algo: The algorithm to use to split out the data.
    :param args: The arguments to the splitting algorithm.
    :param randomizer: Randomizer if needed.
    :return: The elements removed out of the source list.
    """
    shard = []
    if algo == 'random_fraction':
        count = round(len(source_list) * float(args['fraction']))
        val_indexes = randomizer.sample(range(0, len(source_list)), count)
        val_indexes = sorted(val_indexes, reverse=True)

        for i in val_indexes:
            shard.append(source_list[i])
            del (source_list[i])
    else:
        logger.error(f"Unknown validation algorithm '{algo}', not splitting list.")

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
        logger.info(f"Loading - {file_path}... ")
        with open(file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            if first_header is None:
                first_header = next(reader)
            else:
                header = next(reader)
                # All headers must be identical as we don't do ANY mapping.
                if header != first_header:
                    logger.error(f"Header fields of CSV entries do NOT match. EXITING! First header from "
                                 f"{file_list[0]} doesn't match header in {file_path}")
                    sys.exit(-1)

            row_idx = 0
            for row in reader:
                if len(row) <= label_index:
                    logger.error(f"Row has length {len(row)} but we asked for labels that are "
                                 f"column {label_index} row {row_idx}")
                else:
                    label = int(row[label_index])
                    del row[label_index]
                    data[label].append(row)
                row_idx += 1

    return data


def add_image_data_sources(lab: Lab, dataset_config: DatasetConfig, source_list, set_type) -> None:
    """
    This call creates an entry in the source_list for each image source and adds all the items to the
    entry for that set type.
    :param lab: The lab from which to gather data
    :param dataset_config: The dataset config to load.
    :param source_list: The list of sources receiving the new image source.
    :param set_type: The data set type. "train" or "valid".
    :return:
    """
    for source in dataset_config.image_data.sources:
        source_list.append(source)
        source['train'] = []
        source['valid'] = []
        root = source.get("root", "dataroot")
        source_path = source['directory']
        if root == 'dataroot':
            root_path = lab.data_root()
        elif root == 'workspace':
            root_path = lab.workspace()
        elif root == 'relative':
            if dataset_config.relative_path is None:
                logger.error(f"No relative_path set in dataset for 'relative' data source. {source_path}. EXITING")
                sys.exit(-1)
            root_path = dataset_config.relative_path
        else:
            logger.error(f"Unknown source root '{root}' for source path {source_path}")
            sys.exit(-1)

        # Now load up the sources
        source[set_type] = list_or_glob_dir(root_path, source_path)


def validation_from_file(lab, source_list, splitting_config: SplittingConfig):
    # Read the indicated dataset config file
    validation_dataset_config = DatasetConfig.load(splitting_config.args['file_path'])
    add_image_data_sources(lab, validation_dataset_config, source_list, 'valid')

    # Return how they want the validation data sampled
    return validation_dataset_config.get_sampling_config()


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


def list_or_glob_dir(data_root: Path, path: str):
    """
    Returns the items as Paths INCLUDING the data_root.
    :param data_root: Path to the data root in which to look
    :param path: glob or directory
    :return: The items at the path
    """
    if '*' in path:
        return sorted(list(data_root.glob(path)))
    else:
        source_item = data_root / path
        if source_item.is_dir():
            # Iterdir skips hidden directories
            return sorted(list((data_root / path).iterdir()))
        else:
            return [source_item]


def load_coco_json(filepath, output: list) -> None:
    """
    Loads the metadata json file. Validates during load.
    :param filepath: The filepath to load.
    :param output: The output list in which to add out content.
    :return: None
    """
    data = CocoAnnotations.load(filepath)
    helper = coco_utils.COCOImageHelper(data)
    output.extend(helper.to_image_list())


def get_label_dict(label_val: Union[dict, str], key: str = 'labelNames'):
    """
    Helper function for converting a stanza in a specified json file into a Python dictionary of integer keys and
    string values.
    :param label_val: The value associated with the label mapping key. May be a dictionary of label mappings or a path
        to a label mappings dictionary.
    :param key: The key associated with the stanza of interest.
    :return: Returns a dictionary of integer keys mapped to string values.
    """
    if label_val:
        if isinstance(label_val, str) or isinstance(label_val, Path):
            file_content = jb_fs.load_json(str(label_val))
            if key in file_content:
                stanza = file_content[key]
                return convert_dict(stanza)

        elif isinstance(label_val, dict):
            return convert_dict(label_val)

        else:
            logger.error(f"get_label_dict received a label_val that wasn't a str or dict. EXITING.")
            sys.exit(-1)


def convert_dict(stanza):
    """
    Converts a json stanza into a dictionary of integer key and string values.
    :param stanza: The json stanza that will be converted.
    """
    # Return dictionary if stanza is not empty.
    if stanza:
        return {int(k): v for (k, v) in stanza.items()}


def return_label_mapping(show_source: bool, source: str, source_path, label_dict: dict):
    """
    Logs information about the source of the label mapping and returns the label dictionary and (optional) source.
    :param show_source: Boolean indicating if the source should be returned (for testing purposes).
    :param source: String indicating the type of file from which the label mapping was retrieved.
    :param source_path: The path to the file from which the label mapping was retrieved.
    :param label_dict: A dictionary containing an int -> string mapping of labels.
    :return: The dictionary of label names and the source from which that dictionary was retrieved (optional).
    """
    logger.info(f"Using label mapping from"
                f"     Source: {source}"
                f"     Path: {source_path}")
    if show_source:
        return label_dict, source
    else:
        return label_dict


def get_label_mapping(model_manager: ModelManager = None, model_config: ModelConfig = None,
                      train_config: DatasetConfig = None, eval_config: DatasetConfig = None,
                      show_source=False) -> Union[Tuple[Dict[int, str], str], Dict[int, str]]:
    """
    Checks a hierarchy of files to determine the set of label names used by the trained model. The order of precedence
    is as follows: training output.json file, specified model config file, specified training config file, default model
    config file, default training config file, and specified eval config file.
    :param model_manager: The ModelManager object for the model.
    :param model_config: The ModelConfig object for the model configuration.
    :param train_config: The DatasetConfig object for model training.
    :param eval_config: The DatasetConfig object for model evaluation.
    :param show_source: Set to True to return the source from which the label names were extracted.
    :return: The label names as a dict of int -> string.
    """
    # If the model manager was provided, check output.json file (if one exists) for label names.
    if model_manager:
        if model_manager.get_training_out_file().exists():
            training_output = TrainingOutput.load(model_manager.get_training_out_file())
            label_val = training_output.options.label_mapping
            label_dict = get_label_dict(label_val)
            if label_dict:
                return return_label_mapping(show_source=show_source,
                                            source="training output",
                                            source_path=model_manager.get_training_out_file(),
                                            label_dict=label_dict)

    # If a model config was provided...
    if model_config:
        # Check the model config for label names.
        label_val = model_config.label_mapping
        label_dict = get_label_dict(label_val)
        if label_dict:
            return return_label_mapping(show_source=show_source,
                                        source="model config",
                                        source_path=model_config.file_path,
                                        label_dict=label_dict)

    # If a training config was provided...
    if train_config:
        label_dict = train_config.retrieve_label_names()
        if label_dict:
            return return_label_mapping(show_source=show_source,
                                        source="training dataset config",
                                        source_path=train_config.file_path,
                                        label_dict=label_dict)

    # If the model manager was provided, check the default model config followed by the default training config.
    if model_manager:
        mc = ModelConfig.load(model_manager.get_model_config())
        label_val = mc.label_mapping

        # If the model config has labels, use those.
        if label_val is not None:
            label_dict = get_label_dict(label_val)
            if label_dict:
                return return_label_mapping(show_source=show_source,
                                            source="model config via model manager",
                                            source_path=model_manager.get_model_config(),
                                            label_dict=label_dict)

        # If the model config didn't have labels, get them from the training config.
        else:
            dc = DatasetConfig.load(mc.training_dataset_config_path)
            label_dict = dc.retrieve_label_names()
            if label_dict:
                return return_label_mapping(show_source=show_source,
                                            source="training dataset config via model config via model manager",
                                            source_path=dc.file_path,
                                            label_dict=label_dict)

    # If an eval config was provided, check this as a last resort.
    if eval_config:
        label_dict = eval_config.retrieve_label_names()
        if label_dict:
            return return_label_mapping(show_source=show_source,
                                        source="eval dataset config",
                                        source_path=eval_config.file_path,
                                        label_dict=label_dict)


def categories_in_dataset_config(config_path: Path, data_root: Path) -> list:
    """
    Pulls and returns the list of categories from the coco data.
    :param config_path: The path to the dataset config file.
    :param data_root: The path to the data root.
    :return: The list of categories.
    """
    dataset_config = DatasetConfig.load(str(config_path))
    # TODO: eliminate use of snake case
    if 'imageData' in dataset_config:
        coco_path = dataset_config.imageData['sources'][0]['directory']
    else:
        coco_path = dataset_config.image_data['sources'][0]['directory']

    coco_data = CocoAnnotations.load(data_root / coco_path)
    return coco_data.categories


def categories_in_model_config(model_config_path: Path) -> list:
    """
    Pulls, reformats, and returns the list of categories from the model config preprocessors.
    :param model_config_path: The path to the model config file.
    :returns: The list of categories.
    """
    model_config_data = ModelConfig.load(str(model_config_path))
    preprocessor_dict = model_config_data.preprocessors
    object_relabel_dict = next((item for item in preprocessor_dict if item["fqcn"] == "juneberry.transforms"
                                                                                      ".metadata_preprocessors"
                                                                                      ".ObjectRelabel"), None)
    if object_relabel_dict:
        category_dict = object_relabel_dict.kwargs.labels
        if category_dict:
            category_list = []
            for k, v in category_dict.items():
                category_list.append({'id': int(k), 'name': v})
            return category_list

    return []


def check_category_list(category_list: list, eval_manifest_path: Path, show_source: bool, source: str,
                        source_path: str) -> Union[List, Tuple[List, str]]:
    """
    Checks that the extracted category list matches the evaluation manifest category list, and logs a warning if there
    is a mismatch. Logs information about the source of the label mapping. Returns the category list and (optional)
    source.
    :param category_list: The list of categories.
    :param eval_manifest_path: The path to the evaluation manifest file.
    :param show_source: Boolean indicating if the source should be returned (for testing purposes).
    :param source: String indicating the type of file from which the category list was retrieved.
    :param source_path: The path to the file from which the category list was retrieved.
    :return: The list of categories and the source from which the category list was retrieved (optional).
    """
    eval_manifest = jb_fs.load_file(str(eval_manifest_path))
    eval_manifest_categories = eval_manifest["categories"]
    if category_list != eval_manifest_categories:
        logger.warning("The evaluation category list does not match that of the eval_manifest:"
                       f"  category_list: {category_list}"
                       f"  eval_manifest list: {eval_manifest_categories}")
    logger.info(f"Using category list from"
                f"     Source: {source}"
                f"     Path: {source_path}")
    if show_source:
        return category_list, source
    else:
        return category_list


def get_category_list(eval_manifest_path: Path, model_manager: ModelManager = None,
                      train_config: DatasetConfig = None,
                      eval_config: DatasetConfig = None, data_root: Path = None,
                      show_source: bool = False) -> Union[List, Tuple[List, str]]:
    """
    Checks a hierarchy of files to determine the set of categories for use in evaluation. The order of precedence
    is as follows: training manifest, validation manifest, model config, train config, and eval config.
    :param eval_manifest_path: The path to the model's evaluation manifest file.
    :param model_manager: The ModelManager object for the model.
    :param train_config: The DatasetConfig object for model training.
    :param eval_config: The DatasetConfig object for model evaluation.
    :param data_root: The path to the data root of source images.
    :param show_source: Set to True to return the source from which the label names were extracted.
    :return: A dictionary containing the category mapping.
    """
    model_config_path = None
    if model_manager:
        # Check the training manifest
        training_manifest_path = model_manager.get_training_data_manifest_path()
        if training_manifest_path.exists():
            train_manifest_data = jb_fs.load_file(str(training_manifest_path))
            category_list = train_manifest_data.get('categories', None)
            if category_list:
                return check_category_list(category_list, eval_manifest_path, show_source, "train manifest",
                                           str(training_manifest_path))

        # Check the validation manifest
        validation_manifest_path = model_manager.get_validation_data_manifest_path()
        if validation_manifest_path.exists():
            val_manifest_data = jb_fs.load_file(str(validation_manifest_path))
            category_list = val_manifest_data.get('categories', None)
            if category_list:
                return check_category_list(category_list, eval_manifest_path, show_source, "val manifest",
                                           str(validation_manifest_path))

        # Check the model config
        model_config_path = model_manager.get_model_config()
        if model_config_path.exists():
            category_list = categories_in_model_config(model_config_path)
            if category_list:
                return check_category_list(category_list, eval_manifest_path, show_source, "model config",
                                           model_config_path)

    # Check the training config
    train_config_path = None
    if train_config:
        train_config_path = train_config.file_path
        if train_config_path.exists():
            category_list = categories_in_dataset_config(train_config_path, data_root)
            if category_list:
                return check_category_list(category_list, eval_manifest_path, show_source, "train config",
                                           str(train_config_path))

    # Check the evaluation config
    eval_config_path = None
    if eval_config:
        eval_config_path = eval_config.file_path
        if eval_config_path.exists():
            category_list = categories_in_dataset_config(eval_config_path, data_root)
            if category_list:
                return check_category_list(category_list, eval_manifest_path, show_source, "eval config",
                                           str(eval_config_path))

    # Log detailed error message and exit
    logger.error(f"Failed to retrieve a category list via the following args:\n"
                 f"  model_manager: {model_manager}\n"
                 f"  model_config: {model_config_path}\n"
                 f"  train_config: {train_config_path}\n"
                 f"  eval_config: {eval_config_path}\n"
                 f"EXITING.")
    sys.exit(-1)


def check_num_classes(args: dict, num_model_classes: int) -> None:
    """
    Checks that num_model_classes is in the args dictionary and if not
    adds it. Also checks that if one exists it matches the data.
    :param args: The args structure to check/modify from the model architecture.
    :param num_model_classes: The number of model classes to look for. Usually from the dataset.
    :return: None
    """
    if 'num_classes' not in args:
        logger.warning(f"The 'model_architecture' 'args' do not contain 'num_classes' for validation. "
                       f"Using '{num_model_classes}' from the dataset config.")
        args['num_classes'] = num_model_classes
    else:
        if args['num_classes'] != num_model_classes:
            logger.error(f"The number of classes in the training config: '{args['num_classes']}' "
                         f"does not match the number of classes in the dataset: '{num_model_classes}'. EXITING.")
            sys.exit(-1)


def shuffle_manifests(seed: int, *data_lists: list) -> None:
    """
    Shuffles the specified data lists based on the random seed. NOTE: The random seed is NOT reset
    between data lists, so that the selection is not weighted similarly (e.g. early, late) in case
    lists are correlated.
    :param seed: The random seed to use for shuffling.
    :param data_lists: The data lists to shuffle.
    :return: None
    """
    randomizer = random.Random(seed)
    for data_list in data_lists:
        randomizer.shuffle(data_list)


def save_path_label_manifest(data_list, filename, relative_to: Path = None) -> None:
    """
    Save the image datalist of pairs (image path, label) to a json file. The structure is:
    [ { "path":str, "label":int }, ... ] }
    :param data_list: The list of pairs.
    :param filename: A filename to save to.
    :param relative_to: (Optional) Path the filename should be relative to.
    :return: None
    """
    # Convert to json
    rows = []
    for row in data_list:
        path = Path(row[0])
        if relative_to is not None:
            path = path.relative_to(relative_to)
        rows.append({"path": str(path), "label": int(row[1])})

    with open(filename, "w") as out_file:
        json.dump(rows, out_file, indent=4)


def load_path_label_manifest(filename, relative_to: Path = None):
    """
    Loads a list of path, label pairs from json file.
    [ { "path":str, "label":int }, ... ] }
    :param filename: The file to load.
    :param relative_to: (Optional) Path the filename should be relative to.
    :return: A list of pairs of [path, label].
    """
    pairs = []
    data = jb_fs.load_file(filename)

    for row in data:
        path = row['path']
        if relative_to is not None:
            path = str(relative_to / path)
        pairs.append([path, row['label']])
    return pairs


def make_transform_manager(model_cfg: ModelConfig, ds_cfg: DatasetConfig, set_size: int,
                           opt_args: dict, stage_transform_class, eval_mode: bool):
    """
    Constructs the appropriate transform manager for this data.
    :param model_cfg: The model config.
    :param ds_cfg: The dataset config.
    :param set_size: The size of the data set.
    :param opt_args: Optional args to pass into the construction of the plugin.
    :param eval_mode: Are we in train (False) or eval mode (True).
    :param stage_transform_class: The class of a staged transform manager to construct
                                  for when we have both dataset and model transforms.
                                  These class should be StagedTransformManager or look like it.
    :return: Transform manager.
    """
    assert model_cfg is not None
    model_transforms = None
    if model_cfg.seed is not None:
        model_seed = model_cfg.seed
    else:
        logger.warning("Creating a dataset with a random seed!!!")
        model_seed = random.randint(0, 2 ** 32 - 1)

    if eval_mode:
        if model_cfg.evaluation_transforms is not None:
            logger.info(f"Making labeled transform manager from model config EVALUATION transforms.")
            model_transforms = LabeledTransformManager(model_cfg.evaluation_transforms, opt_args)
    else:
        if model_cfg.training_transforms is not None:
            logger.info(f"Making labeled transform manager from model config TRAINING transforms.")
            model_transforms = LabeledTransformManager(model_cfg.training_transforms, opt_args)

    if ds_cfg.data_transforms is not None:
        logger.info(f"Making labeled transform manager from DATASET transforms.")
        ds_transforms = LabeledTransformManager(ds_cfg.data_transforms.transforms, opt_args)
        if ds_cfg.data_transforms.seed is not None:
            data_seed = ds_cfg.data_transforms.seed
        else:
            data_seed = model_seed + set_size
    else:
        ds_transforms = None
        data_seed = 0

    if model_transforms and ds_transforms:
        if stage_transform_class is not None:
            logger.info(f"Making STAGED transform manager with seed {data_seed}.")
            return stage_transform_class(data_seed, ds_transforms, model_seed, model_transforms)
        else:
            logger.error("This platform does not support making BOTH dataset AND model transforms. EXITING.")
            sys.exit(-1)

    elif model_transforms:
        return model_transforms
    elif ds_transforms:
        return ds_transforms
    else:
        return None
