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

from juneberry.transforms.metadata_preprocessors import FilterDetections, LabelMinusOne, ObjectRelabel


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
