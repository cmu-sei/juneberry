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

from juneberry.transforms.metadata_preprocessors import ObjectRelabel
from juneberry.transforms.metadata_preprocessors import FilterDetections
from juneberry.transforms.metadata_preprocessors import LabelMinusOne


def test_object_relabel():
    new_cats = {
        '0': "dog",
        '1': "cat",
        '2': "bird",
        '3': "fish"
    }

    relabel = ObjectRelabel("foo", new_cats)

    orig = {
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, 'extended_categories': {'foo': 0}},
            {"id": 2, "image_id": 1, "category_id": 4, 'extended_categories': {'foo': 1}},
            {"id": 3, "image_id": 1, "category_id": 5, 'extended_categories': {'foo': 2}},
            {"id": 4, "image_id": 2, "category_id": 6, 'extended_categories': {'foo': 3}},
        ],
        "categories": [
            {"id": 0, "name": "zero"},
            {"id": 1, "name": "one"},
            {"id": 2, "name": "two"},
            {"id": 3, "name": "three"},
            {"id": 4, "name": "four"},
            {"id": 5, "name": "five"},
            {"id": 6, "name": "six"}
        ]
    }

    result = relabel(orig)

    assert result['annotations'][0]['category_id'] == 0
    assert result['annotations'][1]['category_id'] == 1
    assert result['annotations'][2]['category_id'] == 2
    assert result['annotations'][3]['category_id'] == 3
    for cat in result['categories']:
        assert cat['name'] == new_cats[str(cat['id'])]


def make_sample():
    images = [
        {"id": 1},
        {"id": 2}
    ]
    annos = []

    # Now make two set of annotations
    # 0,1,2,3,4
    for i in range(5):
        annos.append({"id": i, "image_id": 1, "category_id": i})
    # 5,6
    for i in range(5, 7):
        annos.append({"id": i, "image_id": 2, "category_id": i})
    return {"images": images, "annotations": annos}


def test_filter_detections():
    my_filter = FilterDetections(labels=[1, 2, 3])

    orig = make_sample()
    result = my_filter(orig)

    assert len(result['annotations']) == 3
    assert result['annotations'][0]['category_id'] == 1
    assert result['annotations'][1]['category_id'] == 2
    assert result['annotations'][2]['category_id'] == 3
    assert len(result['images']) == 1
    assert result['images'][0]['id'] == 1


def test_negated_filter_detections():
    my_filter = FilterDetections(labels=[1, 2, 3], contains=False)

    orig = make_sample()
    result = my_filter(orig)

    assert len(result['annotations']) == 4
    assert result['annotations'][0]['category_id'] == 0
    assert result['annotations'][1]['category_id'] == 4
    assert result['annotations'][2]['category_id'] == 5
    assert result['annotations'][3]['category_id'] == 6
    assert len(result['images']) == 2
    assert result['images'][0]['id'] == 1
    assert result['images'][1]['id'] == 2


def test_label_minus_one():

    relabel = LabelMinusOne()

    orig = {
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 3, "image_id": 1, "category_id": 2},
            {"id": 4, "image_id": 2, "category_id": 1},
        ],
        "categories": [
            {"id": 2, "name": "one"},
            {"id": 1, "name": "zero"},
        ]
    }

    result = relabel(orig)

    assert result['annotations'][0]['category_id'] == 0
    assert result['annotations'][1]['category_id'] == 1
    assert result['annotations'][2]['category_id'] == 0
    assert result['categories'][0]['id'] == 1
    assert result['categories'][1]['id'] == 0
