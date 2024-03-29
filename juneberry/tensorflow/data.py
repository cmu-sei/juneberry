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

from functools import partial
import logging
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

import juneberry.config.dataset as jb_dataset
from juneberry.config.dataset import DatasetConfig
import juneberry.config.model as jb_model
from juneberry.config.model import ModelConfig, ShapeHWC
import juneberry.data as jb_data
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
import juneberry.transforms.transform_manager as jb_tm
from juneberry.transforms.transform_manager import TransformManager

logger = logging.getLogger(__name__)


class TFImageDataSequence(tf.keras.utils.Sequence):
    """
    A keras data sequence to be used directly as a data source for training on tensorflow.
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, data_list, batch_size, transforms, shape_hwc: ShapeHWC):
        """
        Construct a keras sequence that delivers the images and labels as batches
        :param data_list: List of pairs of [filepath, label]
        :param batch_size: The size of each batch
        :param transforms: Optional callable to be applied to each image
        :param shape_hwc: Image shape as height, width, channels
        """
        # TODO: Consider preload, caching, trunc-oversample
        self.data_list = data_list
        self.batch_size = batch_size
        self.transforms = transforms
        self.shape_hwc = shape_hwc
        # This give us floor, so we lose some of the last entries.  We could implement an oversample.
        self.len = len(self.data_list) // self.batch_size
        self.epoch = 0

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        start = index * self.batch_size
        images = []
        labels = []

        for i in range(start, start + self.batch_size):
            image, label = self._load_image(i)
            images.append(image)
            labels.append(label)

        # REMEMBER TensorFlow is all HWC
        np_images = np.array(images).reshape(-1, self.shape_hwc.height, self.shape_hwc.width, self.shape_hwc.channels)
        np_labels = np.array(labels)
        return np_images, np_labels

    def on_epoch_end(self):
        # TODO: We want the random seed to be different every epoch.
        #       We need to check to see what is going on with the random numbers here...
        self.epoch += 1

    def _load_image(self, index: int) -> Image:
        image = Image.open(self.data_list[index][0])
        image.load()
        label = self.data_list[index][1]
        if self.transforms is not None:
            args = {'label': label, 'index': index, 'epoch': self.epoch}
            image, label = self.transforms(image, **args)

        return np.array(image), label


#  _____                     __                        ____                               _
# |_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   / ___| _   _ _ __  _ __   ___  _ __| |_
#   | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \  \___ \| | | | '_ \| '_ \ / _ \| '__| __|
#   | || | | (_| | | | \__ \  _| (_) | |  | | | | | |  ___) | |_| | |_) | |_) | (_) | |  | |_
#   |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_| |____/ \__,_| .__/| .__/ \___/|_|   \__|
#                                                                 |_|   |_|


class TFTorchStagedTransform(jb_tm.StagedTransformManager):
    def __init__(self, consistent_seed: int, consistent, per_epoch_seed: int, per_epoch):
        super().__init__(consistent_seed, consistent, per_epoch_seed, per_epoch)
        self.numpy_state = None
        self.python_state = None

    def save_random_state(self):
        self.numpy_state = np.random.get_state()
        self.python_state = random.getstate()

    def restore_random_state(self):
        np.random.set_state(self.numpy_state)
        random.setstate(self.python_state)

    def set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)


def make_transform_manager(model_cfg: ModelConfig, ds_cfg: DatasetConfig, set_size: int,
                           opt_args: dict, eval_mode: bool = False):
    """
    Constructs the appropriate transform manager for the this data.
    :param model_cfg: The model config.
    :param ds_cfg: The dataset config.
    :param set_size: The size of the data set.
    :param opt_args: Optional args to pass into the construction of the plugin.
    :param eval_mode: Are we in train (False) or eval mode (True).
    :return: Transform manager.
    """
    # Convenience to call the base on with our custom stage transform
    return jb_data.make_transform_manager(model_cfg, ds_cfg, set_size, opt_args,
                                          TFTorchStagedTransform, eval_mode)


def _call_transforms_numpy(np_image, y, transforms):
    # Convert to image, apply transforms, and convert back.
    # print(f"+++++++++ type={type(np_image)} shape={np_image.shape}")
    # PIL doesn't like grayscale things to have 3 dimensions, so we have
    # to flatten it, and add it back out.
    shape = np_image.shape
    if shape[2] == 1:
        np_image = np_image.reshape(shape[0], shape[1])
    img = Image.fromarray(np_image)
    img, new_label = transforms(img, label=y)
    if shape[2] == 1:
        return np.array(img).reshape(shape[0], shape[1], 1), new_label
    return np.array(img), np.array(new_label)


# TODO: Should this be a tf.function?
def _transform_magic(image, y, transforms):
    # Set up a numpy function wrapper and then call it.
    # Based on the explanation of py_function we need to explicitly set the shape so the
    # graph code can figure out the size
    # https://www.tensorflow.org/guide/data#applying_arbitrary_python_logic
    # So get the shape, run the function, and set it back.
    im_shape = image.shape
    partial_func = partial(_call_transforms_numpy, transforms=transforms)
    new_image, new_y = tf.numpy_function(partial_func, [image, y], Tout=[tf.uint8, tf.int64])
    new_image.set_shape(im_shape)
    new_y.set_shape(y.shape)
    return new_image, new_y


def _add_transforms_and_batching(dataset, model_cfg, ds_cfg, batch_size, eval_mode):
    """
    Adds transformers and batching instructions to this datasets
    :param dataset: The dataset to augment
    :param model_cfg: The model config
    :param ds_cfg: The dataset config
    :param batch_size: The batch size
    :param eval_mode: Are we in eval mode
    :return: The augmented dataset
    """
    # TODO: Figure out how to do BOTH dataset and model transforms
    # transforms = jb_data.make_transform_manager(
    #     model_cfg, ds_cfg, len(dataset), {},
    #     TFTorchStagedTransform, eval)
    transforms = jb_data.make_transform_manager(
        model_cfg, ds_cfg, len(dataset), {},
        None, eval_mode)

    if transforms is not None:
        dataset = dataset.map(lambda x, y: _transform_magic(x, y, transforms))

    # Apply batching
    return dataset.batch(batch_size)


#   ____
#  / ___|___  _ __ ___  _ __ ___   ___  _ __
# | |   / _ \| '_ ` _ \| '_ ` _ \ / _ \| '_ \
# | |__| (_) | | | | | | | | | | | (_) | | | |
#  \____\___/|_| |_| |_|_| |_| |_|\___/|_| |_|

def _prep_tfds_load_args(tf_stanza):
    # NOTE: Prodict, so can't use "get" with defaults so we check for None.
    load_args = {}
    if tf_stanza.load_kwargs is not None:
        load_args = tf_stanza.load_kwargs

    # We always need supervised so we get a tuple instead of dict as the transform (map) is
    # designed to get the tuple not a dict.
    load_args['as_supervised'] = True

    # Remove some other stuff we don't want
    for unwanted in ['batch_size', 'shuffle']:
        if unwanted in load_args:
            del load_args[unwanted]

    return load_args


#  _____          _
# |_   _| __ __ _(_)_ __
#   | || '__/ _` | | '_ \
#   | || | | (_| | | | | |
#   |_||_|  \__,_|_|_| |_|


def _make_image_dataset(model_config: ModelConfig, ds_cfg: DatasetConfig, data_list,
                        val_set: bool) -> TFImageDataSequence:
    """
    Creates a TF data sequence wrapping for the prepared data list. This is used to provide identical
    inputs like Juneberry does for PyTorch.
    :param model_config: The model config
    :param ds_cfg: The dataset config
    :param data_list: The data list
    :param val_set: True if the validation set, else the training set
    :return: The constructed data sequence
    """
    # TODO: Check sampling and validation split seeds

    shape_hwc = model_config.model_architecture.get_shape_hwc()
    opt_args = {'path_label_list': list(data_list)}
    transform_manager = make_transform_manager(model_config, ds_cfg, len(data_list), opt_args, val_set)
    image_ds = TFImageDataSequence(data_list, model_config.batch_size, transform_manager, shape_hwc)
    logging_name = "validation" if val_set else "training"
    logger.info(f"Constructed {logging_name} dataset with {len(image_ds)} items.")

    tmp_data, tmp_labels = image_ds[0]
    logging_name = "Val" if val_set else "Train"
    logger.info(f"{logging_name}: data.shape={tmp_data.shape}, label.shape={tmp_labels.shape}")

    return image_ds


def _make_tfds_split_args(val_stanza, load_args):
    # Custom args based on algorithm.
    if val_stanza.algorithm == jb_model.SplittingAlgo.TENSORFLOW:
        if 'split' not in load_args:
            logger.error(f"'tensorflow' was specified as a validation algorithm but the required 'split' was "
                         f"not found. load_kwargs={load_args}. See documentation for details. EXITING.")
            sys.exit(-1)
        else:
            split_arg = load_args['split']
            if len(split_arg) != 2 or not isinstance(split_arg[0], str) or not isinstance(split_arg[1], str):
                logger.error(f"The validation algorithm is set to 'tensorflow' which requires that the "
                             f"'split' value be and array of two strings. load_kwargs={load_args}. EXITING.")
                sys.exit(-1)

    elif val_stanza.algorithm == jb_model.SplittingAlgo.RANDOM_FRACTION:
        split_name = 'train'
        if 'split' in load_args:
            split_name = load_args['split']
            if not isinstance(split_name, str):
                logger.error(f"'random_fraction' was specified as a validation algorithm and the optional "
                             "'split' was not a single string. load_kwargs={load_args}. EXITING.")
                sys.exit(-1)

        # Create a training fraction of he low percent and the validation of the high part
        # TODO: How does NOT shuffling affect this???
        train_frac = 100 - int(val_stanza.arguments.fraction * 100)
        train_str = f"{split_name}[:{train_frac}%]"
        test_str = f"{split_name}[{train_frac}%:]"
        load_args['split'] = [train_str, test_str]

    else:
        logger.error(f"TensorFlow does not support the '{val_stanza.algorithm}' validation split algorithm. "
                     "Please use 'random_fraction' or 'tensorflow'. EXITING.")
        sys.exit(-1)


def _load_tfds_split_dataset(ds_config: DatasetConfig, model_config: ModelConfig):
    tf_stanza = ds_config.tensorflow_data
    load_args = _prep_tfds_load_args(tf_stanza)

    # Based on the validation split we will either use the native random fraction
    # or combine the sets and then split.
    _make_tfds_split_args(model_config.validation, load_args)

    # Now load them
    logger.info(f"Loading TFDS name='{tf_stanza.name}' with args='{load_args}'")
    train_ds, val_ds = tfds.load(tf_stanza.name, **load_args)

    # Do this BEFORE batching so that the size shows the full length, not number of batches
    logger.info(f"Loaded dataset with size train={len(train_ds)}, val={len(val_ds)}")

    train_ds = _add_transforms_and_batching(train_ds, model_config, ds_config, model_config.batch_size, False)
    val_ds = _add_transforms_and_batching(val_ds, model_config, ds_config, model_config.batch_size, True)

    return train_ds, val_ds


def load_split_datasets(lab: Lab, ds_config: DatasetConfig, model_config: ModelConfig, model_manager: ModelManager):
    """
    Returns the training and evaluation datasets based on the dataset config and the model config.
    :param lab: The lab in which this occurs.
    :param ds_config: The dataset config that describes the dataset
    :param model_config: A model config that describes validation split.
    :param model_manager: A model manager in which to place the manifest files.
    :return:
    """
    if ds_config.data_type == jb_dataset.DataType.IMAGE:
        train_list, val_list = jb_data.dataspec_to_manifests(
            lab,
            dataset_config=ds_config,
            splitting_config=model_config.get_validation_split_config(),
            preprocessors=TransformManager(model_config.preprocessors))

        # Shuffle the datasets so that the manifests can be used directly.
        logger.info(f"...shuffling manifests with seed {model_config.seed}...")
        jb_data.shuffle_manifests(model_config.seed, train_list, val_list)

        # Save the manifests
        logger.info(f"...saving manifests...")
        jb_data.save_path_label_manifest(train_list, model_manager.get_training_data_manifest_path(), lab.data_root())
        jb_data.save_path_label_manifest(val_list, model_manager.get_validation_data_manifest_path(), lab.data_root())

        # Now make the loaders
        train_ds = _make_image_dataset(model_config, ds_config, train_list, False)
        val_ds = _make_image_dataset(model_config, ds_config, val_list, True)
        return train_ds, val_ds

    elif ds_config.data_type == jb_dataset.DataType.TABULAR:
        logger.error("TensorFlow is currently not ready to support tabular data sets. Exiting.")
        sys.exit(-1)

    elif ds_config.data_type == jb_dataset.DataType.TENSORFLOW:
        return _load_tfds_split_dataset(ds_config, model_config)

    elif ds_config.data_type == jb_dataset.DataType.TORCHVISION:
        logger.error("Torchvision datasets cannot be used with tensorflow. Exiting.")
        sys.exit(-1)


#  _____            _
# | ____|_   ____ _| |
# |  _| \ \ / / _` | |
# | |___ \ V / (_| | |
# |_____| \_/ \__,_|_|

def _extract_labels(eval_ds):
    # HACK: This is HORRIBLY inefficient!!!
    labels_iter = eval_ds.map(lambda x, y: y)
    return [int(x.numpy()) for x in labels_iter]


def _make_image_eval_dataset(model_config: ModelConfig, ds_cfg: DatasetConfig, eval_list):
    shape_hwc = model_config.model_architecture.get_shape_hwc()

    opt_args = {'path_label_list': list(eval_list)}
    transforms = make_transform_manager(model_config, ds_cfg, len(eval_list), opt_args, True)
    eval_ds = TFImageDataSequence(eval_list, model_config.batch_size, transforms, shape_hwc)
    logger.info(f"Constructed evaluation dataset with {len(eval_ds)} items.")

    tmp_data, tmp_labels = eval_ds[0]
    logger.info(f"Eval: data.shape={tmp_data.shape}, label.shape={tmp_labels.shape}")

    return eval_ds


def _make_tfds_eval_args(load_args):
    if 'split' in load_args:
        if not isinstance(load_args['split'], str):
            logger.error(f"During evaluation 'split' was specified for load args, but is not a single string. "
                         f"load_kwargs={load_args}. EXITING.")
            sys.exit(-1)
    else:
        load_args['split'] = "test"


def _load_tfds_eval_dataset(ds_config: DatasetConfig, model_config: ModelConfig, use_train_split, use_val_split):
    if use_train_split and use_val_split:
        logger.error("When constructing eval dataset use_train_split and use_val_split were both specified."
                     "This is not supported. To use the entire dataset, simply don't specify either. EXITING.")
        sys.exit(-1)

    # Now, construct the dataset based on
    tf_stanza = ds_config.tensorflow_data
    load_args = _prep_tfds_load_args(tf_stanza)

    if use_val_split or use_train_split:
        # If use train or use val, make them both but then return the right one
        _make_tfds_split_args(model_config.validation, load_args)
        logger.info(f"Loading TFDS name='{tf_stanza.name}' with args='{load_args}'")
        train_ds, val_ds = tfds.load(tf_stanza.name, **load_args)
        eval_ds = train_ds if use_train_split else val_ds
    else:
        _make_tfds_eval_args(load_args)
        logger.info(f"Loading TFDS name='{tf_stanza.name}' with args='{load_args}'")
        eval_ds = tfds.load(tf_stanza.name, **load_args)

    # Do this BEFORE batching so that the size shows the full length, not number of batches
    logger.info(f"Loaded dataset of size {len(eval_ds)}")

    # Now, get the labels
    labels = _extract_labels(eval_ds)

    # Add transforms
    eval_ds = _add_transforms_and_batching(eval_ds, model_config, ds_config, model_config.batch_size, True)

    return eval_ds, labels


def load_eval_dataset(lab: Lab, ds_config: DatasetConfig, model_config: ModelConfig, eval_dir_mgr,
                      use_train_split, use_val_split):
    """
    Loads the tensorflow evaluation dataset based on the dataset config, model config and splits.
    NOTE. For TFDS the 'split' value comes from the dataset potentially combined with the random fraction.
    :param lab: The lab in which this occurs.
    :param ds_config: The dataset config that describes the dataset
    :param model_config: A model config that describes split style for use_train_split and use_val_split.
    :param eval_dir_mgr: The directory of the evaluation in which to place the manifest file
    :param use_train_split: True to provide the training fraction of the data.
    :param use_val_split: True to provide the validation fraction of the data.
    :return: The evaluation dataset along with the labels.
    """
    if ds_config.data_type == jb_dataset.DataType.IMAGE:
        splitting_config = None
        if use_train_split or use_val_split:
            logger.info(f"Splitting the dataset according to the model's validation split instructions.")
            splitting_config = model_config.get_validation_split_config()

        eval_list, split = jb_data.dataspec_to_manifests(
            lab,
            dataset_config=ds_config,
            splitting_config=splitting_config,
            preprocessors=TransformManager(model_config.preprocessors))

        if use_train_split:
            logger.info("Evaluating using ONLY the training portion of the split data.")

        elif use_val_split:
            logger.info("Evaluating using ONLY the validation portion of the split data.")
            eval_list = split

        # Save the manifest
        jb_data.save_path_label_manifest(eval_list, eval_dir_mgr.get_manifest_path(), lab.data_root())

        # Now make the loaders returning the loader and labels
        return _make_image_eval_dataset(model_config, ds_config, eval_list), [x[1] for x in eval_list]
    elif ds_config.data_type == jb_dataset.DataType.TABULAR:
        logger.error("TensorFlow is currently not ready to support tabular data sets. EXITING.")
        sys.exit(-1)
    elif ds_config.data_type == jb_dataset.DataType.TENSORFLOW:
        return _load_tfds_eval_dataset(ds_config, model_config, use_train_split, use_val_split)
    elif ds_config.data_type == jb_dataset.DataType.TORCHVISION:
        logger.error("Torchvision datasets cannot be used with tensorflow. EXITING.")
        sys.exit(-1)


#  _   _ _   _ _
# | | | | |_(_) |___
# | | | | __| | / __|
# | |_| | |_| | \__ \
#  \___/ \__|_|_|___/

def save_sample_images(tf_ds, image_dir, label_map: dict = None, max_images: int = 5):
    # Reset the random seed so we get different images each dry run.
    random.seed()

    # Walk through the loader batches, sampling one from each.
    idx = 0
    for images, labels in iter(tf_ds):
        # Only put out the number they asked for.
        if idx >= max_images:
            return

        if isinstance(images, tf.Tensor):
            images = images.numpy()
        if isinstance(labels, tf.Tensor):
            labels = labels.numpy()

        # Get a random item from the batch and save it. Note, we need to make sure it is uint8.
        rand_idx = random.randrange(0, len(images))
        np_image = images[rand_idx]
        np_image = np_image.astype(np.uint8)

        # The labels come back as an array of arrays but it is one element
        # label_num = labels[rand_idx][0]
        label_num = labels[rand_idx]

        # PIL GRAYSCALE HACK does not like grayscale images as a dimension of three it wants them as HxW only.
        shape = np_image.shape
        if shape[2] == 1:
            np_image = np_image.reshape(shape[0], shape[1])

        # Convert and save
        img = Image.fromarray(np_image)
        label = label_num
        if label_map is not None:
            label = label_map[label_num]
        path = Path(image_dir) / f"{idx:03}_{label}.png"
        img.save(str(path))

        # Next image
        idx += 1
