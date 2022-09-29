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
import gzip
import json

import juneberry.filesystem as jb_fs
import juneberry.metrics.objectdetection.detection_metrics


def ltwh2ltrb(ltwh):
    return [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]]


def load_coco_truth(gt_path, wanted_images):
    """
    Loaded the truth boxes for a certain set of images in the results set.
    :param gt_path:
    :param wanted_images:
    :return:
    """
    #         {
    #             "area": 2765.1486500000005,
    #             "bbox": [ 199.84, 200.46, 77.71, 70.88 ],
    #             "category_id": 58,
    #             "id": 156,
    #             "image_id": 558840,
    #             "iscrowd": 0,
    #             "segmentation": [ [ <floats> ] ]
    #         },

    if gt_path.endswith(".gz"):
        # TODO: Switch over to using jb_fs.load_file once gzip support has been implemented.
        with gzip.open(gt_path) as json_file:
            data = json.load(json_file, 'json')
    else:
        data = jb_fs.load_file(gt_path)

    truths = []
    for item in data['annotations']:
        if item['image_id'] in wanted_images:
            ltwh = item['bbox']
            truths.append({
                "id": item['id'],
                "image_id": item['image_id'],
                "label": item['category_id'],
                "ltrb": [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]],
                "wh": [ltwh[2], ltwh[3]],
                "mask": None,
                "keypoints": None,
                "ignore": False
            })

            # label = item['category_id']
            # if label == 28 or label == 58:
            #     print(f"{truths[-1]}'")

            # print(f"ltwh={ltwh}, ltrb={results[-1]['ltrb']}, wh={results[-1]['wh']}'")

    return truths


def load_coco_results(results_path):
    results = []

    if results_path.endswith(".gz"):
        # TODO: Switch over to using jb_fs.load_file once gzip support has been implemented.
        with gzip.open(results_path) as json_file:
            data = json.load(json_file)
    else:
        data = jb_fs.load_file(results_path)

    #     {
    #         "image_id": 42,
    #         "category_id": 18,
    #         "bbox": [ 258.15, 41.29, 348.26, 243.78 ],
    #         "score": 0.236
    #     },

    image_ids = defaultdict(int)
    idx = 0
    for item in data:
        ltwh = item['bbox']
        results.append({
            "id": idx,
            "image_id": item['image_id'],
            "ltrb": [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]],
            "wh": [ltwh[2], ltwh[3]],
            "mask": None,
            "keypoints": None,
            "label": item['category_id'],
            "confidence": item['score'],
            "ignore": 0
        })

        # label = item['category_id']
        # if label == 28 or label == 58:
        #     print(f"{results[-1]}'")

        # print(f"ltwh={ltwh}, ltrb={results[-1]['ltrb']}, wh={results[-1]['wh']}'")
        image_ids[item['image_id']] += 1
        idx += 1

    return results, image_ids


def make_truth(truth_id, image_id, label, ltwh):
    return {
        "id": truth_id,
        "image_id": image_id,
        "label": label,
        "ltrb": ltwh2ltrb(ltwh),
        "wh": [ltwh[2], ltwh[3]],
        "mask": None,
        "keypoints": None,
        "ignore": False
    }


def make_detect(detect_id, image_id, label, confidence, ltwh):
    return {
        "id": detect_id,
        "image_id": image_id,
        "label": label,
        "confidence": confidence,
        "ltrb": ltwh2ltrb(ltwh),
        "wh": [ltwh[2], ltwh[3]],
        "mask": None,
        "keypoints": None,
        "ignore": 0
    }


# =================


def test_iou():
    # Two simple bounding boxes slightly offset
    bb1 = [0, 0, 10, 10]  # area is 100
    bb2 = [1, 1, 11, 11]  # area is 100
    intersect = 9 * 9
    union = 100 + 100 - 81
    iou = round(intersect / float(union), 6)

    iou2 = juneberry.metrics.objectdetection.detection_metrics.iou_bbox(bb1, bb2)
    assert iou == iou2

    # Not at origin
    bb1 = [100, 100, 110, 110]  # area is 100
    bb2 = [101, 101, 111, 111]  # area is 100
    intersect = 9 * 9
    union = 100 + 100 - 81
    iou = round(intersect / float(union), 6)

    iou2 = juneberry.metrics.objectdetection.detection_metrics.iou_bbox(bb1, bb2)
    assert iou == iou2

    # Different sizes
    bb1 = [100, 100, 110, 110]  # area is 100
    bb2 = [101, 101, 112, 112]  # area is 121
    intersect = 9 * 9
    union = 100 + 121 - 81
    iou = round(intersect / float(union), 6)

    iou2 = juneberry.metrics.objectdetection.detection_metrics.iou_bbox(bb1, bb2)
    assert iou == iou2

    # Pure overlay
    bb1 = [100, 100, 110, 110]  # area is 100
    bb2 = [100, 100, 110, 110]  # area is 100
    intersect = 10 * 10
    union = 100 + 100 - 100
    iou = round(intersect / float(union), 6)

    iou2 = juneberry.metrics.objectdetection.detection_metrics.iou_bbox(bb1, bb2)
    assert iou == iou2


def get_bboxes():
    return [[0, 0, 10, 10]]


def make_test_truths():
    # Make a set of truths one at each bbox, with increasing labels
    bboxes = get_bboxes()
    image_id = 42
    truth_id = 0
    label = 0
    truths = []

    for bbox in bboxes:
        truths.append({
            "id": truth_id,
            "image_id": image_id,
            "label": label,
            "ltrb": bbox,
            "mask": None,
            "keypoints": None,
            "ignore": 0
        })
        truth_id += 1
        label += 1

    return truths


def make_test_detect():
    # Make a set of detects. One that is offset of one, same size, one with offset 5 larger, and one offset 20
    bboxes = get_bboxes()
    image_id = 42
    detect_id = 0
    label = 0
    detects = []

    for bbox in bboxes:
        detect_bboxes = [[bbox[0] + 1, bbox[1] + 1, bbox[2] + 1, bbox[3] + 1],
                         [bbox[0] + 2, bbox[1] + 2, bbox[2] + 7, bbox[3] + 7],
                         [bbox[0] + 100, bbox[1] + 100, bbox[2] + 100, bbox[3] + 100]]

        confidences = [0.8, 0.7, 0.45]
        for bbox2, confidence in zip(detect_bboxes, confidences):
            detects.append({
                "id": detect_id,
                "image_id": image_id,
                "ltrb": bbox2,
                "mask": None,
                "keypoints": None,
                "label": label,
                "confidence": confidence,
                "ignore": 0
            })
            detect_id += 1

        label += 1

    return detects


def test_find_detect():
    truths = make_test_truths()
    detects = make_test_detect()

    evals = juneberry.metrics.objectdetection.detection_metrics.find_best_detects(truths, detects)
    # DEBUGGING
    # print(f"truths: {json.dumps(truths, indent=4)}")
    # print(f"detects: {json.dumps(detects, indent=4)}")
    # print(f"evals: {json.dumps(evals, indent=4)}")

    # Check results
    for i in range(3):
        iou = juneberry.metrics.objectdetection.detection_metrics.iou_bbox(truths[0]['ltrb'], detects[i]['ltrb'])
        assert evals[i]['detect_id'] == i
        assert evals[i]['score'] == detects[i]['confidence']
        assert evals[i]['iou'] == iou

    assert evals[0]['object_id'] == 0
    assert evals[1]['object_id'] == 0
    assert evals[2]['object_id'] == -1

    assert evals[0]['true_positive'] == 1
    assert evals[1]['true_positive'] == 0
    assert evals[2]['true_positive'] == 0


def test_evaluator_single():
    evaluator = juneberry.metrics.objectdetection.detection_metrics.MAPEvaluator()
    evaluator.add_ground_truth([make_truth(1817255, 42, 18, [214.15, 41.29, 348.26, 243.78])])
    evaluator.add_detections([make_detect(0, 42, 18, 0.236, [258.15, 41.29, 348.26, 243.78])])
    assert round(evaluator.mean_average_precision(), 5) == 1.0


def test_evaluator_two_truth_one_detect():
    print("TWO TRUTH, one DETECT")
    evaluator = juneberry.metrics.objectdetection.detection_metrics.MAPEvaluator()

    # Two motorcycle (cat4) truths
    evaluator.add_ground_truth([
        make_truth(100, 13, 4, [13.0, 22.75, 535.98, 609.67]),
        make_truth(101, 13, 4, [1.66, 3.32, 268.6, 271.91])
    ])
    evaluator.add_detections([make_detect(0, 13, 4, 0.236, [12.66, 3.32, 268.6, 271.91])])
    # print(evaluator.mean_average_precision())
    assert round(evaluator.mean_average_precision(), 6) == round(0.5454545454545454, 6)


def test_evaluator_two_truth_one_detect_one_bogus():
    evaluator = juneberry.metrics.objectdetection.detection_metrics.MAPEvaluator()

    # Two motorcycle (cat4) truths
    evaluator.add_ground_truth([
        make_truth(100, 13, 4, [13.0, 22.75, 535.98, 609.67]),
        make_truth(101, 13, 4, [1.66, 3.32, 268.6, 271.91])
    ])
    evaluator.add_detections([make_detect(0, 13, 4, 0.236, [12.66, 3.32, 268.6, 271.91])])
    evaluator.add_detections([make_detect(0, 13, 11, 0.236, [61, 22.75, 504, 609.67])])
    print(evaluator.mean_average_precision())
    assert round(evaluator.mean_average_precision(), 6) == round(0.2727272727272727, 6)


def evalutor_test_from_files(gt_path, dt_path, map50):
    # Load the results and ground truth
    results, wanted = load_coco_results(dt_path)
    truth = load_coco_truth(gt_path, wanted)

    # Build evaluator
    evaluator = juneberry.metrics.objectdetection.detection_metrics.MAPEvaluator()

    # Add in ground truths
    evaluator.add_ground_truth(truth)

    # Add detections
    evaluator.add_detections(results)

    # See how we are doing
    # print(evaluator.mean_average_precision())
    assert round(map50, 6) == round(evaluator.mean_average_precision(), 6)


def NOT_test_evaluator_2014_reduced():
    # TODO: If we want to use this, we need to check in these files
    # We are going to use the same test values as the coco eval
    # data_dir = Path.cwd() / '..' / 'datasets' / 'coco2014'
    # truth_full = data_dir / 'annotations' / 'instances_val2014.json'
    # detects_full = data_dir / 'results' / 'instances_val2014_fakebbox100_results.json'
    truth_full = 'test/data/instances_val2014_reduced.json.gz'
    detects_full = 'test/data/instances_val2014_fakebbox100_results.json.gz'

    evalutor_test_from_files(truth_full, detects_full, 0.5789761089220521)


def test_compute_precision_and_recall():
    import math
    evals = [
        {'true_positive': 1},
        {'true_positive': 0},
        {'true_positive': 1},
        {'true_positive': 0},
    ]
    juneberry.metrics.objectdetection.detection_metrics.compute_precision_and_recall(evals, 5)
    assert evals[0]['precision'] == 1.0
    assert evals[0]['recall'] == 0.2
    assert evals[1]['precision'] == 0.5
    assert evals[1]['recall'] == 0.2
    assert math.isclose(evals[2]['precision'], 0.666666666667, abs_tol=0.00001)
    assert evals[2]['recall'] == 0.4
    assert evals[3]['precision'] == 0.5
    assert evals[3]['recall'] == 0.4


def test_convert_to_thresholds():
    # data = [
    #     {
    #         "true_positive": 1,
    #         "precision": 1.0,
    #         "recall": 0.2
    #     },
    #     {
    #         "true_positive": 0,
    #         "precision": 0.5,
    #         "recall": 0.2
    #     },
    #     {
    #         "true_positive": 1,
    #         "precision": 0.6666666666666666,
    #         "recall": 0.4
    #     },
    #     {
    #         "true_positive": 0,
    #         "precision": 0.5,
    #         "recall": 0.4
    #     }
    # ]

    data = [
        {
            "true_positive": 1,
            "precision": 1.0,
            "recall": 0.5
        },
    ]
    # This is (0.0 * 5 + 1.0 * 6)/11
    assert round(juneberry.metrics.objectdetection.detection_metrics.interpolated_stepped_average(data, 11), 6) == round(0.5454545454545454, 6)


def main():
    image_ids = [42, 73, 74, 133, 136, 139, 143, 164, 192, 196, 208, 241, 257, 283, 285, 294, 328, 338, 357, 359, 360,
                 387, 395, 397, 400, 415, 428, 459, 472, 474, 486, 488, 502, 520, 536, 544, 564, 569, 589, 590, 599,
                 623, 626, 632, 636, 641, 661, 675, 692, 693, 699, 711, 715, 724, 730, 757, 761, 764, 772, 775, 776,
                 785, 802, 810, 827, 831, 836, 872, 873, 885, 923, 939, 962, 969, 974, 985, 987, 999, 1000, 1029, 1064,
                 1083, 1089, 1103, 1138, 1146, 1149, 1153, 1164, 1171, 1176, 1180, 1205, 1228, 1244, 1268, 1270, 1290,
                 1292]


if __name__ == "__main__":
    main()
