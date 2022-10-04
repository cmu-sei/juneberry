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

"""
A set of utilities for image manipulation.
"""

import logging
import random
from typing import List, Tuple
import warnings

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import albumentations as A

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


def rotate_image(src_image: Image, rotation_degrees: int) -> Image:
    """
    :param src_image: The image receiving the rotation.
    :param rotation_degrees: The maximum amount of rotation in degrees (negative and positive) to apply to the image.
    :return: The rotated Image.
    """
    rotation_amount = random.randint(-rotation_degrees, rotation_degrees)
    logging.debug(f"Rotating the image by {rotation_amount} degrees.")
    return src_image.rotate(rotation_amount, expand=True)


def blur_image(src_image: Image, blur_max_radius: int) -> Image:
    """
    This function is responsible for applying a Gaussian blur to an image.
    :param src_image: The image receiving the Gaussian blur.
    :param blur_max_radius: The maximum radius to use when applying the Gaussian blur.
    :return: The Image with the Gaussian blur applied.
    """
    # The blur_max_radius is treated as the ceiling for the amount to use for the radius in
    # a Gaussian blur.
    blur_radius = random.randint(0, blur_max_radius)
    logging.debug(f"Applying Gaussian blur to the image using radius {blur_radius}.")
    return src_image.filter(ImageFilter.GaussianBlur(blur_radius))


def scale_image(src_image: Image, scale_range: tuple) -> Image:
    """
    This function is responsible for scaling the size of an image.
    :param src_image: The image whose size is being adjusted.
    :param scale_range: A tuple indicating the range of values to consider when scaling the image. scale_range[0]
    represents the lower boundary for the scale value, while scale_range[1] represents the upper boundary.
    :return: A scaled version of the original source image.
    """
    # A random scale value between scale_range[0] and scale_range[1] is chosen for the amount by which to scale
    # the Image's area. Scale values less than 1.0 will shrink the Image, while scale values greater than 1.0
    # will increase the size of the Image.

    logging.debug(f"Scaling the image size...")
    watermark_width, watermark_height = src_image.size
    logging.debug(f"  original image dimensions: width = {watermark_width}, height = {watermark_height}, "
                  f"area = {watermark_width * watermark_height}")

    scale = round(random.uniform(scale_range[0], scale_range[1]), 1)
    logging.debug(f"  applying a scale factor of {scale} to the area of the watermark.")
    src_image = src_image.resize((int(watermark_width * np.sqrt(scale)), int(watermark_height * np.sqrt(scale))),
                                 Image.ANTIALIAS)

    watermark_width, watermark_height = src_image.size
    logging.debug(f"  new image dimensions: width = {watermark_width}, height = {watermark_height}, "
                  f"area = {watermark_width * watermark_height}")

    return src_image


def transform_image(src_image: Image, scale_range: tuple = (None, None), rotation: int = 0, blur: int = 0):
    """
    This function is responsible for applying transformations to the provided image.
    :param src_image: The image receiving the transformations.
    :param scale_range: Tuple indicating the min/max values to use for scaling the area of the watermark,
    with 1.0 being no scaling.
    :param rotation: The maximum amount of rotation in degrees (negative and positive) to apply to the image.
    :param blur: The maximum radius to use when applying Gaussian blur to the image.
    :return: The transformed image.
    """

    # If a blur radius was provided, attempt to apply a Gaussian blur to the image.
    if blur:
        src_image = blur_image(src_image=src_image, blur_max_radius=blur)

    # Scale the image if a scale_range was provided that would actually adjust the image size.
    if scale_range != (None, None) and scale_range != (1.0, 1.0):
        src_image = scale_image(src_image=src_image, scale_range=scale_range)

    # When a rotation value is provided, this transform will randomly select a value between
    # -rotationAmount and +rotationAmount and apply that rotation to the watermark image. The
    # "expand" argument ensures that the watermark will not be trimmed if the rotation causes pixels
    # to fall outside of the original dimensions of the image.
    if rotation:
        src_image = rotate_image(src_image, rotation)

    return src_image


def make_random_insert_position(src_size_wh, target_box_wh, randomizer=None) -> (int, int):
    """
    Randomly computes a position inside the target box which will fit the entirety
    of the source box without clipping.
    :param src_size_wh: The size of the source box to insert
    :param target_box_wh: The size of the destination region
    :param randomizer: The randomizer to use.
    :return:
    """

    # Make a "reduced" area so we can compute left top. (xy)
    max_x = target_box_wh[0] - src_size_wh[0]
    max_y = target_box_wh[1] - src_size_wh[1]

    # Find a value in the range
    if randomizer is None:
        randomizer = random

    if max_x < 0 or max_y < 0:
        print(f"Asked to compute random position for watermark:{src_size_wh} "
              f"target:{target_box_wh} and max_x or max_y is less than zero. "
              f"max_x={max_x}, max_y={max_y}")
        # raise RuntimeError("Watermark larger than target image.")
        return 0, 0
    x = randomizer.randint(0, max_x)
    y = randomizer.randint(0, max_y)

    return x, y


def insert_watermark_at_position(image: Image, watermark: Image, position_xy):
    """
    Inserts the watermark into the image at the specified position. If the watermark
    is an RGBA image, then it is also passed in as the mask.
    :param image: The image in which to place to watermark.
    :param watermark: The watermark.
    :param position_xy: The position at which to insert the watermark.
    :return The modified image.
    """

    # The third argument here is a mask parameter.
    # It is an RGBA image so the mask will be the alpha channel of the watermark.
    if watermark.mode == "RGBA":
        image.paste(watermark, position_xy, watermark)
    else:
        image.paste(watermark, position_xy)

    return image


def random_sharpen(image: Image):
    """
    Randomly blur or sharpen an image.
    """
    scale = random.uniform(0.8,1.1)
    transform = None
    if scale < 1:
        scale = (1.0-scale) * 50.0
        transform = A.augmentations.transforms.Blur(blur_limit=scale, p=1)
    elif scale > 1:
        scale = scale - 1.0
        transform = A.augmentations.transforms.Sharpen(alpha= (scale,scale), lightness=(1.0,1.0), p=1)
    else:
        return image
    return transform(image=image)['image']