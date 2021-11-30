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

import unittest

from juneberry.config.coco_anno import CocoAnnotations


def make_basic_config():
    # Based on https://blog.superannotate.com/coco-dataset-introduction/
    return {
        "info": {
            "year": 2021,
            "version": 1.2
        },
        "licenses": {},
        "categories": [
            {
                "id": 1,
                "name": "poodle",
                "supercategory": "dog",
                "isthing": 1,
                "color": [1, 0, 0]
            },
            {
                "id": 2,
                "name": "ragdoll",
                "supercategory": "cat",
                "isthing": 1,
                "color": [2, 0, 0]
            }
        ],
        "image": [
            {
                "id": 122214,
                "width": 640,
                "height": 640,
                "file_name": "84.jpg",
                "license": 1,
                "date_captured": "2021-07-19  17:49"
            }
        ],
        "annotations": [
            {
                "segmentation": [[34, 55, 10, 71, 76, 23, 98, 43, 11, 8]],
                "area": 600.4,
                "iscrowd": 1,
                "image_id": 122214,
                "bbox": [473.05, 395.45, 38.65, 28.92],
                "category_id": 1,
                "id": 934
            }
        ],
    }


def replace_with_rle(coco_dict: dict):
    """
    Replaces segmentation portion of annotation with run-length encoding (RLE).
    :param coco_dict:
    :return Coco dictionary containing an annotation with an RLE-style segmentation property.
    """
    rle_segmentation = {
        "segmentation": {
            "counts": [34, 55, 10, 71],
            "size": [240, 480]
        }
    }
    coco_dict["annotations"]["segmentation"] = rle_segmentation
    return coco_dict


def test_config_basics():
    config = make_basic_config()
    coco_anno = CocoAnnotations.construct(config)
    assert len(config['images']) == len(coco_anno['images'])
    assert len(config['annotations']) == len(coco_anno['annotations'])


class TestFormatErrors(unittest.TestCase):
    def test_version(self):
        config = make_basic_config()
        config['formatVersion'] = "0.0.0"

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            CocoAnnotations.construct(config)
        message = "Coco annotations file at"
        self.assertIn(message, log.output[0])

    def test_duplicate_images(self):
        config = make_basic_config()
        config['images'].append(config['images'][0])

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            CocoAnnotations.construct(config)
        message = "Found duplicate image id: id= '0'."
        self.assertIn(message, log.output[0])

    def test_rle_segmentation(self):
        config = make_basic_config()
        config = replace_with_rle(config)
        CocoAnnotations.construct(config)
