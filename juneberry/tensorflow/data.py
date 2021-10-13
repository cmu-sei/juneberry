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

from functools import partial
import logging
from pathlib import Path
import random
import sys

import numpy as np

from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig, ShapeHWC
import juneberry.data as jb_data
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
from juneberry.transform_manager import TransformManager

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

        # This give us floor, so we lose some of the last entries.  We could implement an oversample
        self.len = len(self.data_list) // self.batch_size

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        start = index * self.batch_size
        images = []
        labels = []

        for i in range(start, start + self.batch_size):
            images.append(self._load_image(i))
            labels.append(self.data_list[i][1])

        # REMEMBER Tensorflow is all HWC
        np_images = np.array(images).reshape(-1, self.shape_hwc.height, self.shape_hwc.width, self.shape_hwc.channels)
        np_labels = np.array(labels)
        return np_images, np_labels

    def on_epoch_end(self):
        # TODO: We want the random seed to be different every epoch
        #       We need to check to see what is going on with the random numbers here...
        pass

    def _load_image(self, index: int) -> Image:
        image = Image.open(self.data_list[index][0])
        image.load()
        if self.transforms is not None:
            image = self.transforms(image)
        return np.array(image)


def construct_image_datasets(model_config: ModelConfig, train_list, val_list):
    shape_hwc = model_config.model_architecture.get_shape_hwc()

    transform_manager = TransformManager(model_config.training_transforms)
    train_ds = TFImageDataSequence(
        train_list,
        model_config.batch_size,
        transform_manager,
        shape_hwc)
    logger.info(f"Constructed training dataset with {len(train_ds)} items")

    transform_manager = TransformManager(model_config.evaluation_transforms)
    val_ds = TFImageDataSequence(
        val_list,
        model_config.batch_size,
        transform_manager,
        shape_hwc)
    logger.info(f"Constructed validation dataset with {len(val_ds)} items")

    tmp_data, tmp_labels = train_ds[0]
    logger.info(f"Train: data.shape={tmp_data.shape}, label.shape={tmp_labels.shape}")
    tmp_data, tmp_labels = val_ds[0]
    logger.info(f"Val: data.shape={tmp_data.shape}, label.shape={tmp_labels.shape}")

    return train_ds, val_ds


def call_transforms_numpy(np_image, transforms):
    # Convert to image, apply transforms and convert back
    # print(f"+++++++++ type={type(np_image)} shape={np_image.shape}")
    # PIL doesn't like grayscale things to have 3 dimensions, so we have
    # to flatten in, and add it back out.
    shape = np_image.shape
    if shape[2] == 1:
        np_image = np_image.reshape(shape[0], shape[1])
    img = Image.fromarray(np_image)
    img = transforms(img)
    if shape[2] == 1:
        return np.array(img).reshape(shape[0], shape[1], 1)
    return np.array(img)


# TODO: Should this be a tf.function?
def transform_image(image, transforms):
    # Set up a numpy function wrapper and then call it.
    partial_func = partial(call_transforms_numpy, transforms=transforms)
    return tf.numpy_function(partial_func, [image], Tout=tf.uint8)


def load_tf_dataset(ds_config: DatasetConfig, model_config: ModelConfig):
    tf_stanza = ds_config.tensorflow_data

    # Based on the validation split we will either use the native random fraction
    # or combine the sets and then split

    val_stanza = model_config.validation

    # We always need supervised so we get a tuple insted of dict as the transform (map) is
    # designed to get the tuple not a dict.
    load_args = {}
    if tf_stanza.load_args is not None:
        load_args = {}
    load_args['as_supervised'] = True

    # Customer args based on algorithm
    if val_stanza.algorithm == "tensorflow":
        if 'split' not in load_args:
            load_args['split'] = ["train", "test"]
    elif val_stanza.algorithm == "random_fraction":
        train_frac = 100 - int(val_stanza.arguments.fraction * 100)
        train_str = f"train[:{train_frac}%]+test[:{train_frac}%]"
        test_str = f"train[{train_frac}%:]+test[{train_frac}%:]"
        load_args['split'] = [train_str, test_str]
    else:
        logger.error(f"TensorFlow does not support the '{val_stanza.algorithm}' validation split algorithm. "
                     "Please use 'random_fraction' or 'tensorflow'. EXITING")
        sys.exit(-1)

    # Now load it
    train_ds, val_ds = tfds.load(tf_stanza.name, **load_args)

    # TODO: Turn on shuffling
    # If they have specified a shuffle size, then add that
    # train_ds = train_ds.shuffle(shuffle_buffer_size, seed)

    # TRANSFORMS - THIS IS BEFORE batching! So one element each
    # Map each element with our transform
    transforms = TransformManager(model_config.training_transforms)
    train_ds = train_ds.map(lambda x, y: (transform_image(x, transforms), y))

    transforms = TransformManager(model_config.evaluation_transforms)
    val_ds = val_ds.map(lambda x, y: (transform_image(x, transforms), y))

    # Apply batching
    train_ds = train_ds.batch(model_config.batch_size)
    val_ds = val_ds.batch(model_config.batch_size)

    return train_ds, val_ds


def load_datasets(lab: Lab, ds_config: DatasetConfig, model_config: ModelConfig, model_manager:ModelManager):
    if ds_config.data_type == "image":
        train_list, val_list = jb_data.dataspec_to_manifests(
            lab,
            dataset_config=ds_config,
            splitting_config=model_config.get_validation_split_config(),
            preprocessors=TransformManager(model_config.preprocessors))

        # Save the manifests
        jb_data.save_path_label_manifest(train_list, model_manager.get_training_data_manifest_path(), lab.data_root())
        jb_data.save_path_label_manifest(val_list, model_manager.get_validation_data_manifest_path(), lab.data_root())

        # Now make the loaders
        return construct_image_datasets(model_config, train_list, val_list)
    elif ds_config.data_type == "tabular":
        logger.error("TensorFlow is currently not ready to support tabular data sets. EXITING")
        sys.exit(-1)
    elif ds_config.data_type == "tensorflow":
        return load_tf_dataset(ds_config, model_config)
    elif ds_config.data_type == "torchvision":
        logger.error("Torchvision datasets cannot be used with tensorflow. EXITING")
        sys.exit(-1)





def save_sample_images(tf_ds, image_dir, label_map:dict=None, max_images:int=5):
    # Reset the random seed so we get different images each dry run
    random.seed()

    # Walk through the loader batches, sampling one from each
    idx = 0
    for images, labels in iter(tf_ds):
        # Only put out the number they asked for
        if idx >= max_images:
            return

        if isinstance(images, tf.Tensor):
            images = images.numpy()
        if isinstance(labels, tf.Tensor):
            labels = labels.numpy()

        # Get a random item from the batch and save it. Note, we need to make sure it is uint8
        rand_idx = random.randrange(0, len(images))
        np_image = images[rand_idx]
        np_image = np_image.astype(np.uint8)

        # The labels come back as an array of arrays but it is one element
        # label_num = labels[rand_idx][0]
        label_num = labels[rand_idx]

        # Convert and save
        img = Image.fromarray(np_image)
        label = label_num
        if label_map is not None:
            label = label_map[label_num]
        path = Path(image_dir) / f"{idx:03}_{label}.png"
        img.save(str(path))

        # Next image
        idx += 1