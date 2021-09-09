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

"""
A set of utilities for image manipulation.
"""

import logging
import numpy as np
import random
import warnings

from PIL import Image
from PIL import ImageOps
from typing import List, Tuple

logger = logging.getLogger(__name__)


def translate_image(image, up, left):
    """
    Performs an Affine Transformation on the image
    :param image: Image to be transformed
    :param up: Number of pixels to move the image upwards
    :param left: Number of pixels to move the image leftwards
    """
    return image.transform(image.size, Image.AFFINE, (1, 0, left, 0, 1, up))


def compute_elementwise_mean(np_arrays):
    """
    Computes a element-wise mean from numpy arrays of dtype uint8 and returns the mean.
    Each array must have the same shape.
    :param np_arrays: The stack of arrays from which to compute the mean.
    :return: The element-wise mean
    """

    # DESIGN NOTE: We can support other than 8bit entries but we need to change the
    # accumulator size or type appropriately, write tests, etc. but we don't currently
    # have a need.

    if np_arrays[0].dtype != 'uint8':
        logger.error("elementwise_mean requires input type of uint8! Returning None.")
        return None

    # First we need a deep image with "larger" pixels (byte-wise) to allow the adds without byte overrun
    accumulator = np.zeros(np_arrays[0].shape, dtype='uint32')
    for img in np_arrays:
        # Just keep accumulating to the accumulator so we don't make copies
        accumulator = np.add(accumulator, img, accumulator)

    # Make a uniform divisor of the number of images
    divisor = np.full(np_arrays[0].shape, len(np_arrays), dtype='uint32')
    mean_array = np.divide(accumulator, divisor).astype('uint8')

    # Return the array as a numpy array of the right type
    return mean_array.astype('uint8')


def random_shift_image(pillow_image, max_width, max_height):
    """
    Randomly shifts the image width randomly with in [-max_width, max_width] and the
    height within the range [-max_height, max_height] adding black space at the edges.
    :param pillow_image: The image to shift
    :param max_width: The maximum amount to shift left or right.
    :param max_height: The maximum height to shift up or down.
    :return: Shifted image.
    """
    dx = random.randint(max_width * -1, max_width)
    dy = random.randint(max_height * -1, max_height)
    return pillow_image.crop((dx, dy, pillow_image.size[0] + dx, pillow_image.size[1] + dy))


def random_mirror_flip(pillow_image, mirror_chance=None, flip_chance=None):
    """
    Randomly mirrors (horizontal) or flips (vertical) the provided image. Uses 'random' for generation.
    :param pillow_image: Image to mirror or flip.
    :param mirror_chance: Chance [0,1] to mirror.
    :param flip_chance: Chance [0,1] to flip.
    :return: Modified image.
    """
    if mirror_chance is not None and random.random() < mirror_chance:
        pillow_image = ImageOps.mirror(pillow_image)
    if flip_chance is not None and random.random() < flip_chance:
        pillow_image = ImageOps.flip(pillow_image)

    return pillow_image


def random_crop_mirror_image(pillow_image, mirror: bool, width_pixels: int, height_pixels: int) -> List:
    """
    Randomly shifts and mirrors a single image
    :param pillow_image: The image to shift/mirror
    :param mirror: Set to true to randomly mirror
    :param width_pixels: Max shift width pixels
    :param height_pixels: Max shift width pixels
    :return: The shifted and maybe mirrored image
    """

    new_img = random_shift_image(pillow_image, width_pixels, height_pixels)

    if mirror and bool(random.getrandbits(1)):
        new_img = ImageOps.mirror(new_img)

    return new_img


def random_crop_mirror_images(pillow_images, mirror: bool, width_pixels: int, height_pixels: int) -> List:
    """
    Randomly shifts and mirrors a list of images
    :param pillow_images: The images to shift/mirror
    :param mirror: Set to true to randomly mirror
    :param width_pixels: Max shift width pixels
    :param height_pixels: Max shift width pixels
    :return: List of shifted and maybe mirrored images
    """
    new_images = []
    for img in pillow_images:
        # Crop image
        new_img = random_shift_image(img, width_pixels, height_pixels)
        if mirror and bool(random.getrandbits(1)):
            new_img = ImageOps.mirror(new_img)
        new_images.append(new_img)

    return new_images


def compute_channel_means(images):
    """
    Computes the mean for each channel across the entire set of images.
    :param images: A list of the ndarray images to process
    :return: The mean of each channel as a list.
    """
    if len(images[0].shape) == 3 and images[0].shape[2] > images[0].shape[0]:
        logger.error("Image should have a shape of W X H X C")
    if len(images[0].shape) == 2:
        channels = None
    else:
        channels = tuple(i for i in range(images[0].shape[2]))
    return np.array(images).mean(axis=channels) / 255


def compute_channel_stds(images):
    """
    Computes the mean for each channel across the entire set of images.
    :param images: A list of the ndarray images to process
    :return: The std of each channel as a list.
    """
    if len(images[0].shape) == 3 and images[0].shape[2] > images[0].shape[0]:
        logger.error("Image should have a shape of W X H X C")
    if len(images[0].shape) == 2:
        channels = None
    else:
        channels = tuple(i for i in range(images[0].shape[2]))
    return np.array(images).std(axis=channels) / 255


def find_rectangular_bbox(mask_image, mask_id):
    """
    Finds one rectangular aligned bounding box around all the pixels with the mask_id.
    :param mask_image: The pillow image of the mask
    :param mask_id: The id (cell value) to look for.
    :return: top, left, bottom, right of bounding box.
    """
    # We use numpy to find all the elements with the mask_id in the image. Numpy.where returns
    # parallel arrays of rows and columns.  We get the bbox top, left, bottom, right
    # by taking minimums and maximums.
    np_mask_img = np.array(mask_image)
    fr, fc = np.where(np_mask_img == mask_id)
    return [fr.min(), fc.min(), fr.max(), fc.max()]


def find_rectangular_bbox_omit(mask_image, omit_id):
    """
    Finds the bounding box of all that is not the omit_id.
    :param mask_image: The pillow image of the mask
    :param omit_id: The id (cell value) of things to ignore.
    :return: top, left, bottom, right of bounding box.
    """
    np_mask_img = np.array(mask_image)
    fr, fc = np.where(np_mask_img != omit_id)
    return [fr.min(), fc.min(), fr.max(), fc.max()]


def resize_image(src_img, dst_width, dst_height, color=(0, 0, 0)):
    """
    Resizes/reshapes the image to the destination size. This will
    pad the sides or top as needed.
    :param src_img: The source image
    :param dst_width: The destination image width.
    :param dst_height: The destination image height.
    :param color: Color for the destination image.
    :return: The resized (reshaped) image.
    """

    # Basic algorithm
    # Resize the original image to fit INSIDE the box
    # Create a new box
    # Past the image in

    src_width, src_height = src_img.size

    # First let's determine the aspect ratio if width to height of src and dst
    src_aspect = src_width / src_height
    dst_aspect = dst_width / dst_height

    if src_aspect > dst_aspect:
        # Src is aspect is large than dest, so black bars need to go on top and bottom.
        # Think wide screen to normal with black bars on top/bottom
        # So we fill dest width and scale height
        tmp_width = dst_width
        tmp_height = int(dst_width / src_aspect)
    else:
        # Need black bars sides
        tmp_height = dst_height
        tmp_width = int(dst_height * src_aspect)

    # Resize the source image
    resized = src_img.resize((tmp_width, tmp_height), Image.ANTIALIAS)

    # Make a new empty image of the destination size.
    new_img = Image.new("RGB", (dst_width, dst_height), color)

    # Now paste in the resized part with a top left (x,y)
    new_img.paste(resized, (int((dst_width - tmp_width) / 2), int((dst_height - tmp_height) / 2)))

    return new_img


def convert_image(img, colorspace, width, height):
    """
    Process the image to appropriate colorspace and size
    :param img: The pillow img to process
    :param colorspace: Defines if we're using gray images or RGB.
    :param width: The OUTPUT width of the image in pixels.
    :param height: The OUTPUT height of the image in pixels.
    :return: The processed image.
    """
    src_width, src_height = img.size

    # If the image is NOT in the right dimensions, resize
    if width != src_width or height != src_height:
        img = resize_image(img, width, height)

    if img.mode != 'RGB' and img.mode != 'L':
        logger.error(f"We expect input images to be either RGB or L (8 bit gray). Image is: {img.mode}")

    # Now change the colorspace(mode) if needed
    if img.mode == 'L' and colorspace == "rgb":
        img = img.convert('RGB')
    elif img.mode == 'RGB' and colorspace == 'gray':
        img = img.convert('L')

    return img


def process_images(image_list, colorspace, width, height):
    """
    Loads and process the images in the list, performing resizing and color conversion if necessary.
    :param image_list: The full file paths of the images to load
    :param colorspace: Defines if we're using gray images or RGB.
    :param width: The OUTPUT width of the image in pixels.
    :param height: The OUTPUT height of the image in pixels.
    :return: The loaded and modified images AS NUMPY ARRAYS
    """
    images = []
    for filename in image_list:
        # Filters out a warning associated with a known Pillow issue.
        warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

        img = Image.open(filename)
        img = convert_image(img, colorspace, width, height)
        images.append(np.array(img))

    return images


def load_prepare_images(data: List[Tuple[str, int]], size_wh: Tuple[int, int], colorspace: str):
    """
    Prepares the images and labels specified from the data_list make them the specified
    size and colorspace.
    :param data: A of the data as pairs of filename, int
    :param size_wh: Size of images (width, height) as a tuple.
    :param colorspace: A string describing the colorspace such as RGB.
    :return: list of PIL images, a list of integer labels
    """
    random.shuffle(data)
    files, labels = zip(*data)
    images = process_images(files, colorspace, size_wh[0], size_wh[1])

    return images, labels


def make_np_arrays(images: List, labels: List[int], size_wh: Tuple[int, int], colorspace: str):
    """
    Converts the images and labels to properly sizes numpy.ndarrays
    :param images: The images
    :param labels: The labels
    :param size_wh: The size in width and height
    :param colorspace: String representing the colorspace
    :return: numpy.ndarray of PIL images, numpy.ndarray of integer labels
    """
    channels = 1 if colorspace == 'gray' else 3

    np_images = np.array(images).reshape(-1, size_wh[1], size_wh[0], channels)
    np_labels = np.array(labels)

    return np_images, np_labels


def load_prepare_images_np(data: List[Tuple[str, int]], size_wh: Tuple[int, int], colorspace: str):
    """
    Prepares the images and labels specified from the data_list make them the specified
    size and colorspace and returns them as arrays of NP values.
    :param data: A of the data as pairs of filename, int
    :param size_wh: Size of images (width, height) as a tuple.
    :param colorspace: A string describing the colorspace such as RGB.
    :return: numpy.ndarray of PIL images, numpy.ndarray of integer labels
    """
    images, labels = load_prepare_images(data, size_wh, colorspace)
    return make_np_arrays(images, labels, size_wh, colorspace)
