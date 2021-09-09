#! /usr/bin/env python

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

from collections import defaultdict
import logging
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
        helper = COCOImageHelper(data)
        for image, annotations in helper.values():
            logger.info(f"{image['file_name']} category_ids={[x['category_id'] for x in annotations]}")


class LogExtendedLabels:
    """
    Logs the value of the extended labels
    """

    def __call__(self, data):
        helper = COCOImageHelper(data)
        for image, annotations in helper.values():
            logger.info(f"{image['file_name']} "
                        f"extended_categories={[x.get('extended_categories', '') for x in annotations]}")
