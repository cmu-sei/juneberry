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

from collections import defaultdict, namedtuple
import copy
from datetime import datetime as dt
import logging
from pathlib import Path
from random import shuffle as rand_shuffle
import sys
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from juneberry.config.dataset import DatasetConfig
from juneberry.config.coco_anno import CocoAnnotations
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)

ImageAnnotations = namedtuple('ImageAnnotations', ['image', 'annotations'])


class COCOImageHelper:
    """
    A helper object for accessing the COCODataset in an image centric way. The
    raw data is directly accessible via the 'data' property with convenience
    dictionaries of image_id -> image in the 'images' property and
    image_id -> list of annotations in the 'annotations' property.

    The helper works as a convenience wrapper that behaves like a dictionary with
    the key being the image_id (not index) and the value being a pair of
    `image, tuple(annotations)`. Accordingly, all the proper dictionary methods
    are implemented.

    This leads to the basic pattern such as:

    for image_id, (image, annotations) in helper.items():
        ...

    Or without the keys:

    for image, annotations in helper.values():
        ...

    Also, the values are returned as `namedtuple`s so one can do:

    for item in helper.values():
        item.image[...]
        item.annotations[...]
    """

    def __init__(self, data: CocoAnnotations, file_path=None):
        """
        Creates a helper object to manage the coco annotations structure.
        :param data: The data such as one would get from a file
        :param file_path: Optional file path if read from a file.
        """
        self.data = data

        # Image lookup
        self.images = {i.id: i for i in self.data.images}

        # Annotations lookup
        self.annotations = defaultdict(list)
        for anno in self.data.annotations:
            self.annotations[anno.image_id].append(anno)

        # Optional file path for debugging
        self.file_path = file_path

        # Determine the maximum annotation ID so that new annotations receive the correct ID.
        self._next_annotation_id = -1
        for anno in self.data.annotations:
            if anno.id > self._next_annotation_id:
                self._next_annotation_id = anno.id
        self._next_annotation_id += 1

    def __contains__(self, item: int):
        return item in self.images

    def __getitem__(self, item: int):
        """
        Returns the image stanza and a tuple of annotations as a tuple
        :param item: The id of the image to return
        :return: image, tuple(annotations)
        """
        return ImageAnnotations(self.images[item], tuple(self.annotations[item]))

    def __iter__(self):
        """ :return: Iterator of image keys. """
        return iter(self.images.keys())

    def __len__(self):
        """ :return: The number of images. """
        return len(self.images)

    def keys(self):
        """ :return: The image_id keys. """
        return self.images.keys()

    def items(self):
        """
        A convenience function to return a view of
        key, (image, tuple(annotation))
        :return: iterator
        """
        for k in self.images.keys():
            yield k, (self.images[k], tuple(self.annotations[k]))

    def values(self):
        """
        A convenience function to return a view of
        image, tuple(annotation)
        :return: iterator
        """
        for k in self.images.keys():
            yield self.images[k], tuple(self.annotations[k])

    def remove_image(self, image_id: int) -> None:
        """
        Removes the image and any annotations associated with this image.
        :param image_id: The image_id
        :return: None
        """
        self.data.images = [i for i in self.data.images if i.id != image_id]
        self.data.annotations = [i for i in self.data.annotations if i.image_id != image_id]

        # Remove from the quick lookups
        del self.images[image_id]
        del self.annotations[image_id]

    def add_annotation(self, annotation) -> None:
        """
        Add the provided annotation to the image. Assigns the appropriate next 'id' to the annotation.
        :param annotation: The annotation to add.
        :return: None
        """
        if annotation['image_id'] not in self.images:
            logger.error(f"No image exists for image_id={annotation['image_id']}. EXITING")
            sys.exit(-1)

        # Set the annotation id
        annotation['id'] = self._next_annotation_id
        self._next_annotation_id += 1

        # Add it to the underlying store and the quick lookup
        self.data.annotations.append(annotation)
        self.annotations[annotation['image_id']].append(annotation)

    def to_image_list(self):
        """
        Creates a new image list of the values with the following structure:
        {
           <copy of image properties>
           "annotations": <list of annotations for this image>
        }
        :return: List of image structures
        """
        flattened = []
        for image, annotations in self.values():
            entry = copy.copy(image)
            entry['annotations'] = annotations
            flattened.append(entry)

        return flattened

    def get_category_map(self):
        """
        Provides a dictionary that maps the integer class labels to the human-readable
        string name for the label. In the dictionary, the keys are the integer label values and the values
        are the human-readable string.
        :return:
        """
        return {int(x.id): x.name for x in self.data.categories}


def load_from_json_file(file_path) -> COCOImageHelper:
    """
    Constructs a COCOImageHelper from the specified file.
    :param file_path:
    :return: Constructed COCOImageHelper
    """
    return COCOImageHelper(CocoAnnotations.load(file_path), file_path)


def convert_predictions_to_annotations(predictions: list) -> list:
    """
    The input is an ARRAY of:
    {
        "bbox": [
            1.7060012817382812,
            301.7135314941406,
            62.832000732421875,
            80.3057861328125
        ],
        "category_id": 3,
        "image_id": 1,
        "score": 0.9976264834403992
    },

    :param predictions:
    :return:
    """
    annos = []
    obj_id = 0

    for pred in predictions:
        anno = copy.copy(pred)
        anno['area'] = pred['bbox'][2] * pred['bbox'][3]
        anno['id'] = obj_id
        anno['iscrowd'] = 0
        obj_id += 1
        annos.append(anno)

    return annos


def convert_predictions_to_coco(coco_data: CocoAnnotations, predictions: list,
                                category_list: List = None) -> CocoAnnotations:
    """
    Converts a predictions (detections) list to a coco formatted annotation file with images.
    :param coco_data: A base coco file with images.
    :param predictions: The list of predictions (detections).
    :param category_list: Optional list of categories to use instead of the ones int the coco_data.
    :return: coco formatted data
    """
    if not category_list:
        category_list = coco_data.categories

    coco_data = {
        "info": {
            "date_created": str(dt.now().replace(microsecond=0).isoformat())
        },
        'images': coco_data.images,
        'annotations': convert_predictions_to_annotations(predictions),
        'categories': category_list
    }
    return CocoAnnotations.construct(coco_data)


def convert_jbmeta_to_coco(metadata_list, categories: Dict[int, str], *, renumber=True, add_info=True) -> dict:
    """
    Converts the provided metadata list to coco-style annotations with images, annotations,
    and categories. The category of each annotation is validated against the categories.
    If renumber is specified, all the ids are renumbered from zero. IDs are always added when missing.
    NOTE: No license fields are added.
    :param metadata_list: The metadata to convert.
    :param categories: A dict of categories in the format of id:int -> name:str
    :param renumber: True to ALWAYS renumber the object and annotation ids. (Regardless we
    add ids if needed and make sure image_ids match annotations.)
    :param add_info: True to add a simple info section.
    :return: The coco formatted structure.
    """
    images = []
    annotations = []

    crowd_added = 0

    anno_id = 0
    for image_id, meta in enumerate(metadata_list):
        # We want the image without the annotations entry.
        new_img = copy.copy(meta)
        del new_img['annotations']
        if 'id' not in new_img or renumber:
            new_img['id'] = image_id
        images.append(new_img)

        # Copy over each annotation, cleaning up and checking the fields.
        for anno in meta['annotations']:
            new_anno = copy.deepcopy(anno)
            if 'iscrowd' not in new_anno:
                new_anno['iscrowd'] = 0
                crowd_added += 1
            # We always set the image_id no matter what was here before.
            # We may have changed the previous one or it may just be stale.
            # Regardless, it must match the image to be valid.
            new_anno['image_id'] = new_img['id']
            if 'id' not in new_anno or renumber:
                new_anno['id'] = anno_id
            anno_id += 1
            category_id = new_anno['category_id']
            if category_id not in categories:
                logger.error(f"Category_id {category_id} missing from categories in annotation: {new_anno}")
                sys.exit(-1)
            annotations.append(new_anno)

    if crowd_added > 0:
        logger.warning(f"Added 'iscrowd' to {crowd_added} annotations.")

    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': [{"id": k, "name": v} for k, v in categories.items()]
    }

    if add_info:
        coco_data['info'] = {
            "date_created": str(dt.now().replace(microsecond=0).isoformat())
        }

    return CocoAnnotations.construct(coco_data)


def save_predictions_as_anno(data_root: Path, dataset_config: str, predict_file: str, category_list: List = False,
                             output_file: Path = None, eval_manifest_path: Path = None):
    """
    This function is responsible for converting coco-style object detection predictions into
    a coco-style annotations file.
    :param data_root: A Path to a data root directory where the source images can be found.
    :param dataset_config: A string indicating the path to the dataset config that describes the source images.
    :param predict_file: A string indicating the path to the file containing the bounding boxes that were predicted.
    :param category_list: A list containing the coco formatted categories.
    :param output_file: An optional Path indicating where to save the resulting annotations file. When this is not
    provided, the default will save the annotations file to the current working directory using a variation of
    the name of the predictions file.
    :param eval_manifest_path: An optional Path to an eval manifest. When provided, the "images" and "categories"
    data listed in the manifest will be given priority over the version of those fields retrieved from the dataset
    config.
    :return: Nothing.
    """

    if output_file is None:
        output_file = Path.cwd() / (Path(predict_file).stem + "_anno.json")

    logger.info(f"Saving predictions in annotation-style format in {output_file}")

    # Obtain the coco metadata; the eval_manifest is higher priority.
    if eval_manifest_path:
        coco_data = CocoAnnotations.load(str(eval_manifest_path))

    else:
        # Alternatively, the dataset config should have a version of the metadata.
        dataset = DatasetConfig.load(dataset_config)
        coco_path = dataset.image_data.sources[0]['directory']
        coco_data = CocoAnnotations.load(data_root / coco_path)

    predictions = jbfs.load_file(predict_file)

    coco_out = convert_predictions_to_coco(coco_data, predictions, category_list)
    coco_out.save(str(output_file))


def generate_bbox_images(coco_json: Path, lab, dest_dir: str = None, sample_limit: int = None, shuffle: bool = False):
    """
    This function is responsible for drawing bounding boxes on images based on the information
    contained inside a COCO annotations file.
    :param coco_json: The path to the COCO annotations JSON file describing the images and the bounding
    boxes to be drawn on the images.
    :param lab: A Juneberry lab object which assists with locating the source images.
    :param dest_dir: A destination directory where the boxed images will be placed. When not provided, this
    will default to a directory named "boxed_imgs" in the current working directory.
    :param sample_limit: An integer value used to limit the number of boxed images that will be saved. When
    not provided, the function will draw boxes on every image in the annotations file.
    :param shuffle: Boolean that controls whether or not to shuffle the order of the input images before
    they receive boxes.
    :return: Nothing.
    """

    # Determine the output directory.
    output_dir = Path.cwd() / "boxed_imgs" if dest_dir is None else Path(dest_dir)
    logger.info(f"Images will be saved to the following directory: {output_dir}")

    # Create the output directory if it doesn't exist.
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.info(f"The output directory was not found, so it was created.")

    # Load the COCO annotations.
    coco = CocoAnnotations.load(str(coco_json))

    # Use a COCOImageHelper to obtain the legend.
    helper = COCOImageHelper(coco)
    legend = helper.get_category_map()

    img_count = 0
    box_total = 0

    # Establish the list of images.
    image_list = coco.images

    # If requested, shuffle the list of images.
    if shuffle:
        rand_shuffle(image_list)

    # Iterate through the list of images and draw the bounding boxes.
    for image in image_list:

        # If the number of sampled images has been reached, there's no need to continue the loop.
        if sample_limit is not None and img_count >= sample_limit:
            break

        img_file = lab.data_root() / Path(image.file_name)
        image_id = image.id

        logger.info(f"Attempting to draw boxes on {img_file}")

        # Grab the associated bounding boxes and labels.
        bbox_labels = [{"bbox": x.bbox, "category_id": x.category_id, "category": legend[x.category_id],
                        "score": x.score, } for x in coco.annotations if x.image_id == image_id]

        box_str = "bounding box" if len(bbox_labels) else "bounding boxes"
        logger.info(f"    Adding {len(bbox_labels)} {box_str} to the above image.")

        # Draw the boxes on the image.
        with Image.open(img_file) as file:
            # Conditional to check what type of data we are working with.
            if not (file.mode == "I;16" or file.mode == "I"):
                img = file.convert('RGB')
            else:
                # We need to take the PIL image and normalize it again.
                img = np.array(file)
                norm = (img.astype(np.float32)) * 255.0 / (img.max())
                img = Image.fromarray(norm).convert('RGB')
        draw = ImageDraw.Draw(img)

        # colorblind palette: https://davidmathlogic.com/colorblind  IBM version
        palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
        box_count = 0
        for obj in bbox_labels:
            bbox = obj['bbox']
            # We want the rectangle to act as border around bounding box,
            # so shift start and end points one pixel away from bounding box.
            line_width = 2
            start_point = (bbox[0] - line_width, bbox[1] - line_width)
            end_point = (bbox[0] + bbox[2] - 1 + line_width, bbox[1] + bbox[3] - 1 + line_width)
            outline = palette[obj['category_id'] % len(palette)]
            draw.rectangle(xy=[start_point, end_point], outline=outline, width=line_width)
            draw.text(start_point, f"{obj['category_id']} - {obj['category']}: {obj['score'] * 100:.2f}%")
            box_count += 1

        if box_count > 0:
            img.save(output_dir / f"{img_file.stem}.png")
            box_total += box_count
            img_count += 1

    img_str = "image" if img_count == 1 else "images"
    box_str = "bounding box" if box_total == 1 else "bounding boxes"

    logger.info(f"Added bounding boxes to {img_count} {img_str}.")
    logger.info(f"Drew {box_total} {box_str} across all images.")


def get_class_label_map(anno: Dict) -> List[str]:
    """
    This function is responsible for retrieving the class label map from the annotations file.
    The class label map is used to convert the values in the class_label column of the
    detections Dataframe from integers into strings.
    :param anno: The annotations Dict in COCO format containing the class label information.
    :return: A List of str containing the classes for each integer label.
    """

    categories = anno["categories"]

    # Create an ID list, which contains every integer value that appears
    # as a category in the annotations file.
    id_list = []
    for category in categories:
        id_list.append(category["id"])

    # Set up the class label map such that there is one entry for every
    # possible integer, even if the integer does not appear as a category
    # in the annotations file.
    class_label_map = [None] * (max(id_list) + 1)

    # For the categories that appear in the annotations file, fill in the
    # appropriate entry of the class label map using the string for that
    # integer.
    for category in categories:
        class_label_map[category["id"]] = category["name"]

    # Brambox expects the first item in the class label map to be for
    # label 1, so take the first item (label 0) and move it to the end of
    # the class label map.
    class_label_map.append(class_label_map.pop(0))

    return class_label_map


"""
{
    "info" : info, 
    "images" : [image], 
    "annotations" : [annotation], 
    "licenses" : [license],
}

info{
    "year" : int, 
    "version" : str, 
    "description" : str, 
    "contributor" : str, 
    "url" : str, 
    "date_created" : datetime,
}

image{
    "id" : int, 
    "width" : int, 
    "height" : int, 
    "file_name" : str, 
    "license" : int, 
    "flickr_url" : str, 
    "coco_url" : str, 
    "date_captured" : datetime,
}

annotation{
    "id" : int, 
    "image_id" : int, 
    "category_id" : int, 
    "segmentation" : RLE or [polygon], 
    "area" : float, 
    "bbox" : [x,y,width,height], 
    "iscrowd" : 0 or 1,
}

categories[{
    "id" : int, 
    "name" : str, 
    "supercategory" : str,
}]

"""
