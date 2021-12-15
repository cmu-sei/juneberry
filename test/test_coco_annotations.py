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


import unittest
from juneberry.config.coco_anno import CocoAnnotations


def make_basic_config():
    # Based on https://blog.superannotate.com/coco-dataset-introduction/
    return {
        "info": {
            "year": 2021,
            "version": "1.2"
        },
        "licenses": [],
        "categories": [
            {
                "id": 1,
                "name": "poodle",
                "supercategory": "dog"
            },
            {
                "id": 2,
                "name": "ragdoll",
                "supercategory": "cat"
            }
        ],
        "images": [
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
                "area": 600.4,
                "iscrowd": 1,
                "image_id": 122214,
                "bbox": [473.05, 395.45, 38.65, 28.92],
                "category_id": 1,
                "id": 934
            }
        ]
    }


class TestCocoAnno(unittest.TestCase):
    def test_config_basics(self):
        config = make_basic_config()
        coco_anno = CocoAnnotations.construct(config)
        assert len(config['images']) == len(coco_anno['images'])
        assert len(config['annotations']) == len(coco_anno['annotations'])

    def test_duplicate_images(self):
        config = make_basic_config()
        config['images'].append(config['images'][0])

        with self.assertRaises(SystemExit), self.assertLogs(level='ERROR') as log:
            CocoAnnotations.construct(config)
        message = "Found duplicate image id: id= '122214'."
        self.assertIn(message, log.output[0])
