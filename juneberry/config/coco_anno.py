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

import logging
import sys

from prodict import List, Prodict

import juneberry.config.util as jb_conf_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


# For more information about the COCO annotations data format, see https://cocodataset.org/#format-data


class License(Prodict):
    id: int
    name: str
    url: str


class Category(Prodict):
    id: int
    name: str


class Image(Prodict):
    id: int
    width: int
    height: int
    file_name: str


class Annotation(Prodict):
    id: int
    image_id: int
    category_id: int


class CocoAnnotations(Prodict):
    """
    A class to validate and manage the coco annotations files.
    """
    FORMAT_VERSION = '1.2'
    SCHEMA_NAME = 'coco_anno_schema.json'

    info: dict
    licenses: List[License]
    categories: List[Category]
    images: List[Image]
    annotations: List[Annotation]

    def _finish_init(self) -> None:
        """
        Validate coco fields
        :return: None
        """
        error_count = 0

        # Check for duplicate images
        id_set = set()
        for image in self.images:
            if image.id in id_set:
                logger.error(f"Found duplicate image id: id= '{image.id}'.")
                error_count += 1
            else:
                id_set.add(image.id)

        # If errors found, report and exit
        if error_count > 0:
            logger.error(f"Found {error_count} errors in experiment config. EXITING.")
            sys.exit(-1)

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a coco annotations object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: The constructed object.
        """
        # Convert category ids to integers
        for index in range(0, len(data["categories"])):
            data["categories"][index]["id"] = int(data["categories"][index]["id"])

        # Set info and licenses to their default data structures
        if "info" not in data:
            data["info"] = {}
        if "licenses" not in data:
            data["licenses"] = []

        # Validate with our schema
        if not jb_conf_utils.validate_schema(data, CocoAnnotations.SCHEMA_NAME):
            logger.error(f"Validation errors in {file_path}. See log. EXITING.")
            sys.exit(-1)

        # Construct the CocoAnnotations object
        config = CocoAnnotations.from_dict(data)
        config._finish_init()
        return config

    @staticmethod
    def load(data_path: str):
        """
        Loads the coco annotations file from the provided path, validates, and constructs the CocoAnnotations object.
        :param data_path: Path to the coco annotations file.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading COCO ANNOTATIONS from {data_path}")
        data = jb_fs.load_file(data_path)

        # Construct the config.
        return CocoAnnotations.construct(data, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the coco annotations file to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        jb_conf_utils.validate_and_save_json(self.to_json(), data_path, CocoAnnotations.SCHEMA_NAME)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to json"""
        return jb_conf_utils.prodict_to_dict(self)
