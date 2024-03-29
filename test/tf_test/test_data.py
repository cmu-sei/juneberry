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

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from juneberry.config.dataset import TensorFlowData
from juneberry.config.model import SplittingAlgo, Validation, ValidationArgs
import juneberry.tensorflow.data as tf_data


def generate_filled_box(filename: str, width, height, bands: int, color_rate=None) -> None:
    """
    Generates an image in the file path that has a box with a series of concentric bands
    of stepped color (0 to max) from black (outside) to max insie.
    :param filename: The file to save to
    :param width: The width of the image
    :param height: The height of the image
    :param bands: How many bands to draw
    :param color_rate: The rate of color change of each change.  Range 0-1.
    :return: Nothing
    """
    color_step = 255 // bands
    x_step = width // (bands * 2)
    y_step = height // (bands * 2)
    if not color_rate:
        color_rate = [1, 1, 1]

    img = Image.new("RGB", (width, height), color=(0, 0, 0))

    channel_color = 0
    x = 0
    y = 0

    draw = ImageDraw.Draw(img)
    for i in range(bands):
        if i == bands - 1:
            # The end is always max so we can test the center
            color = (255 * color_rate[0], 255 * color_rate[1], 255 * color_rate[2])
        else:
            color = (channel_color * color_rate[0], channel_color * color_rate[1], channel_color * color_rate[2])
        draw.rectangle(((x, y), (width - x, height - y)), fill=color)
        channel_color += color_step
        x += x_step
        y += y_step

    img.save(filename)


def setup_sample_dir(tmp_path):
    paths = []
    # Just make a simple structure
    p = Path(tmp_path)
    print(str(p))
    dir1 = p / "dir1"
    dir1.mkdir()
    generate_filled_box(str(dir1 / "file_01.png"), 48, 48, 5, (1, 0, 0))
    generate_filled_box(str(dir1 / "file_02.png"), 48, 48, 5, (0, 1, 0))
    paths.append(str(dir1 / "file_01.png"))
    paths.append(str(dir1 / "file_02.png"))

    dir2 = p / "dir2"
    dir2.mkdir()
    generate_filled_box(str(dir2 / "file_03.png"), 48, 48, 5, (0, 0, 1))
    generate_filled_box(str(dir2 / "file_04.png"), 48, 48, 5, (1, 1, 1))
    paths.append(str(dir2 / "file_03.png"))
    paths.append(str(dir2 / "file_04.png"))

    return paths


def test_dataset(tmp_path):
    paths = setup_sample_dir(tmp_path)
    labels = [0, 1, 2, 3]
    data_list = list(zip(paths, labels))

    from juneberry.config.model import ShapeHWC
    hwc = ShapeHWC(32, 32, 3)
    image_ds = tf_data.TFImageDataSequence(data_list, 2, (lambda x, **args: (x.resize((32, 32)), args["label"])), hwc)

    for tmp_data, tmp_labels in image_ds:
        # print(f"data:{type(tmp_data)} shape:{tmp_data.shape} "
        #       f"labels:{type(tmp_labels)} shape:{tmp_labels.shape}")
        # print(f"{tmp_data} {tmp_labels}")
        assert len(tmp_data) == 2
        assert len(tmp_labels) == 2

    # This article lists a bunch of "gotchas"
    # https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    data_0, labels_0 = image_ds[0]
    print(f"data:{type(data_0)} shape:{data_0.shape} "
          f"labels:{type(labels_0)} shape:{labels_0.shape}")
    row_0, label_0 = data_0[0], labels_0[0]
    print(f"data:{type(row_0)} shape:{row_0.shape} "
          f"labels:{type(label_0)} shape:{label_0.shape}")

    assert isinstance(data_0, np.ndarray)
    assert isinstance(labels_0, np.ndarray)

    # Check the labels
    assert image_ds[0][1][0] == 0
    assert image_ds[0][1][1] == 1
    assert image_ds[1][1][0] == 2
    assert image_ds[1][1][1] == 3

    # Check the center color
    assert np.array_equal(image_ds[0][0][0][16][16], [255, 0, 0])
    assert np.array_equal(image_ds[0][0][1][16][16], [0, 255, 0])
    assert np.array_equal(image_ds[1][0][0][16][16], [0, 0, 255])
    assert np.array_equal(image_ds[1][0][1][16][16], [255, 255, 255])


def test_prep_tfds_load_args():
    tf_stanza = TensorFlowData(name="frodo", load_kwargs={"batch_size": 2, "shuffle": True, "foo": "bar"})

    load_args = tf_data._prep_tfds_load_args(tf_stanza)

    assert load_args.get('as_supervised', None) == True
    assert 'batch_size' not in load_args
    assert 'shuffle' not in load_args
    assert load_args['foo'] == 'bar'


def test_make_tfds_split_args():
    # Tensorflow and split args
    val_stanza = Validation()
    val_stanza.algorithm = SplittingAlgo.TENSORFLOW.value
    val_stanza.arguments = {}
    load_args = {'split': ["foo", "bar"]}

    tf_data._make_tfds_split_args(val_stanza, load_args)

    # Nothing should change
    assert load_args['split'] == ["foo", "bar"]

    # ===========================
    # Random fraction and NO split (so use the default train split)
    val_stanza = Validation()
    val_stanza.algorithm = SplittingAlgo.RANDOM_FRACTION.value
    val_stanza.arguments = ValidationArgs()
    val_stanza.arguments.fraction = 0.3
    load_args = {}

    tf_data._make_tfds_split_args(val_stanza, load_args)

    # Now, we should have two string with the lower and higher percent with the default split
    assert load_args['split'] == ["train[:70%]", "train[70%:]"]

    # ===========================
    # Random fraction and split with a custom name
    load_args = {'split': 'banana'}

    tf_data._make_tfds_split_args(val_stanza, load_args)

    # Now, we should have two string with the lower and higher percent with the default split
    assert load_args['split'] == ["banana[:70%]", "banana[70%:]"]


def test_make_tfds_eval_args():
    # No split
    load_args = {}
    tf_data._make_tfds_eval_args(load_args)
    assert load_args['split'] == 'test'

    # If they give us a split, we use it.
    load_args = {'split': "foo"}
    tf_data._make_tfds_eval_args(load_args)
    assert load_args['split'] == 'foo'
