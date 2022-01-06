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

from collections import defaultdict
import logging

from juneberry.config.coco_anno import CocoAnnotations
from juneberry.config.coco_utils import COCOImageHelper

logger = logging.getLogger(__name__)


class ObjectRelabel:
    """
    This preprocessor relabels the object detections based on a property name from the specified key
    in the extendedLabels structure, if one exists.
    """

    def __init__(self, key, labels):
        """
        :param key: The key of the extended_categories property that we should use to get the new label
        :param labels: A map of the new categories to set as the categories. e.g {'0': 'dog, '2': 'cat'}
        """
        self.key = key
        self.categories = labels

    def __call__(self, data):
        # Relabel each object detection
        for anno in data['annotations']:
            anno['category_id'] = anno.get('extended_categories', {}).get(self.key, anno['category_id'])

        # Now reset the categories
        data['categories'] = [{'id': int(k), 'name': self.categories[k]} for k in sorted(list(self.categories.keys()))]

        return data


class FilterDetections:
    """
    Retains objects that are IN the labels (by default) or are NOT in the labels if 'contains' is set to False.
    Returns None if no detections remain.
    """

    def __init__(self, labels, contains=True):
        self.labels = labels
        self.contains = contains

    def __call__(self, data):
        # First off, go through and rebuild all the annotations
        if self.contains:
            data['annotations'] = [i for i in data['annotations'] if i['category_id'] in self.labels]
        else:
            data['annotations'] = [i for i in data['annotations'] if i['category_id'] not in self.labels]

        # Now, build a count of annotations
        counts = defaultdict(int)
        for anno in data['annotations']:
            counts[anno['image_id']] += 1

        # Now, rebuild the images off the ones that have nonzero counts
        data['images'] = [i for i in data['images'] if counts[i['id']] > 0]

        return data


class LabelMinusOne:
    """
    This preprocessor subtracts one from the label, to allow for one indexed datasets to be used without an 
    empty, zero class.
    """

    def __call__(self, data):
        for anno in data['annotations']:
            anno['category_id'] -= 1

        for cat in data['categories']:
            cat['id'] -= 1

        return data


class LogLabel:
    """
    Logs the value of the labels
    """

    def __call__(self, data):
        helper = COCOImageHelper(CocoAnnotations.construct(data))
        for image, annotations in helper.values():
            logger.info(f"{image['file_name']} category_ids={[x['category_id'] for x in annotations]}")


class LogExtendedLabels:
    """
    Logs the value of the extended labels
    """

    def __call__(self, data):
        helper = COCOImageHelper(CocoAnnotations.construct(data))
        for image, annotations in helper.values():
            logger.info(f"{image['file_name']} "
                        f"extended_categories={[x.get('extended_categories', '') for x in annotations]}")
