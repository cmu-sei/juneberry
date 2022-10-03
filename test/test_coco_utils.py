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

from juneberry.config.coco_anno import CocoAnnotations
import juneberry.config.coco_utils as coco_utils

IMAGE_IDS = [2, 4, 8, 20]
OBJECT_IDS = [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10]]
LABEL_MAP = {0: 'zero', 1: 'one', 2: 'two', 3: 'three'}


# Deterministic routine for making an image.
def create_image(image_idx, image_ids):
    return {
        "id": image_ids[image_idx],
        "width": (image_idx + 1) * 10,
        "height": (image_idx + 1) * 15,
        "file_name": f"image_{image_ids[image_idx]}.png"
    }


def create_annotation(image_idx, object_idx, image_ids, object_ids):
    offset = (image_idx + 1) * 10 + object_idx
    return {
        "id": object_ids[image_idx][object_idx],
        "image_id": image_ids[image_idx],
        "category_id": object_idx,
        "area": (30 + offset) * (40 + offset),
        "bbox": [10 + offset, 20 + offset, 30 + offset, 40 + offset],
        "iscrowd": 0,
    }


def make_sample_coco(image_ids, object_ids) -> CocoAnnotations:
    images = [create_image(i, image_ids) for i in range(len(object_ids))]
    annos = [create_annotation(img_idx, obj_idx, image_ids, object_ids)
             for img_idx, anno_ids in enumerate(object_ids) for obj_idx in range(len(anno_ids))]
    data = {
        "images": images,
        "annotations": annos,
        "categories": [{"id": k, "name": v} for k, v in LABEL_MAP.items()]
    }
    return CocoAnnotations.construct(data)


def test_coco_helper_reading():
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)
    assert len(helper) == 4
    for image_id, count in zip(IMAGE_IDS, [len(x) for x in OBJECT_IDS]):
        assert len(helper[image_id][1]) == count
        assert helper[image_id].image['id'] == image_id


def test_coco_helper_writing():
    # The goal of the helper is to provide access to the underlying data in a different way.
    # So, if we change something using the get method, it should reflect in the underlying structure.
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)

    # Remember the access of the helper is by image_id not index.
    helper[4].annotations[2]['category_id'] = 1234
    assert data['annotations'][4]['category_id'] == 1234


def test_coco_iteration():
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)

    # Pairs should give us everything.
    idx = 0
    for i, a in helper.values():
        assert i['id'] == IMAGE_IDS[idx]
        assert len(a) == len(OBJECT_IDS[idx])
        idx += 1


def test_coco_key_iter():
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)

    idx = 0
    for k in helper:
        assert k == IMAGE_IDS[idx]
        idx += 1


def test_remove_image():
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)

    helper.remove_image(4)
    assert len(helper.keys()) == 3
    assert len(list(helper.values())) == 3
    assert len(helper) == 3
    assert len(helper.data['images']) == 3
    assert len(helper.data['annotations']) == 8
    # Make sure it _really_ modified the underlying store.
    for anno in data['annotations']:
        assert anno['image_id'] != 4


def test_add_annotation():
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)

    new_anno = {
        "image_id": 8,
        "category_id": 0,
        "area": 100,
        "bbox": [10, 10, 10, 10],
        "iscrowd": 0,
        "foo": "bar"
    }

    helper.add_annotation(new_anno)
    assert len(helper.keys()) == 4
    assert len(list(helper.values())) == 4
    assert len(helper) == 4
    assert len(helper.data['images']) == 4
    assert len(helper.data['annotations']) == 12
    assert helper.data['annotations'][-1]['foo'] == 'bar'


def test_to_image_list():
    data = make_sample_coco(IMAGE_IDS, OBJECT_IDS)
    helper = coco_utils.COCOImageHelper(data)

    image_list = helper.to_image_list()
    assert len(image_list) == 4
    assert [x['id'] for x in image_list] == IMAGE_IDS
    assert [len(x['annotations']) for x in image_list] == [2, 3, 4, 2]
    for image, ids in zip(image_list, OBJECT_IDS):
        assert [x['id'] for x in image['annotations']] == ids


def test_convert_jbmeta_to_coco():
    jb_meta = [
        create_image(0, IMAGE_IDS),
        create_image(3, IMAGE_IDS)
    ]
    # Now make any annotations.
    jb_meta[0]['annotations'] = [
        create_annotation(0, 0, IMAGE_IDS, OBJECT_IDS),
        create_annotation(0, 1, IMAGE_IDS, OBJECT_IDS)
    ]
    jb_meta[1]['annotations'] = [
        create_annotation(3, 0, IMAGE_IDS, OBJECT_IDS),
        create_annotation(3, 1, IMAGE_IDS, OBJECT_IDS)
    ]

    # ======== RENUMBERING

    # NOTE: We created the second image/annotation with a discontinuity.
    # When we get the file back, all the numbers should be good
    # and everything should match.
    results = coco_utils.convert_jbmeta_to_coco(jb_meta, LABEL_MAP)
    assert results['images'][0]['id'] == 0
    assert results['images'][1]['id'] == 1

    # The remaining fields should be the same.
    for idx in [0, 1]:
        assert results['images'][idx]['file_name'] == jb_meta[idx]['file_name']
        assert results['images'][idx]['width'] == jb_meta[idx]['width']
        assert results['images'][idx]['height'] == jb_meta[idx]['height']

    # Now, the annotations should map properly, but be consecutive.
    # The originals were 0,1,9,10.
    for idx, val in enumerate([0, 1, 2, 3]):
        assert results['annotations'][idx]['id'] == val

    # The originals were 0,3.
    for idx, val in enumerate([0, 0, 1, 1]):
        assert results['annotations'][idx]['image_id'] == val

    # ======== RENUMBERING OFF
    results = coco_utils.convert_jbmeta_to_coco(jb_meta, LABEL_MAP, renumber=False)
    assert results['images'][0]['id'] == 2
    assert results['images'][1]['id'] == 20

    # The remaining fields should be the same.
    for idx in [0, 1]:
        assert results['images'][idx]['file_name'] == jb_meta[idx]['file_name']
        assert results['images'][idx]['width'] == jb_meta[idx]['width']
        assert results['images'][idx]['height'] == jb_meta[idx]['height']

    # Now, the annotations should map properly, but be consecutive.
    # The originals were 0,1,9,10.
    for idx, val in enumerate([0, 1, 9, 10]):
        assert results['annotations'][idx]['id'] == val

    # The originals were 0,3.
    for idx, val in enumerate([2, 2, 20, 20]):
        assert results['annotations'][idx]['image_id'] == val

    # ======== RENUMBERING OFF but NO numbers, so it must add them.

    jb_meta[0]['id'] = 100
    jb_meta[1]['id'] = 108
    for x, y in [[x, y] for x in range(2) for y in range(2)]:
        del jb_meta[x]['annotations'][y]['id']

    results = coco_utils.convert_jbmeta_to_coco(jb_meta, LABEL_MAP, renumber=False)
    assert results['images'][0]['id'] == 100
    assert results['images'][1]['id'] == 108

    # The remaining fields should be the same.
    for idx in [0, 1]:
        assert results['images'][idx]['file_name'] == jb_meta[idx]['file_name']
        assert results['images'][idx]['width'] == jb_meta[idx]['width']
        assert results['images'][idx]['height'] == jb_meta[idx]['height']

    # In this case with renumber off we expect it to create numbers as appropriate.
    # The image_ids should have been added based on the original ids
    # and the annotations ids are sequential.

    for idx, val in enumerate([0, 1, 2, 3]):
        assert results['annotations'][idx]['id'] == val

    for idx, val in enumerate([100, 100, 108, 108]):
        assert results['annotations'][idx]['image_id'] == val
