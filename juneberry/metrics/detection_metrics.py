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

"""
Object detection metrics.
"""

from collections import defaultdict
import json

import numpy as np

# This module expects the following values to be available for each object

truth_struct = {
    "id": 0,
    "image_id": 0,
    "label": 0,
    "ltrb": None,
    "mask": None,
    "keypoints": None,
    "ignore": 0
}

detect_struct = {
    "id": 0,
    "image_id": 0,
    "ltrb": None,
    "mask": None,
    "keypoints": None,
    "label": 0,  # Proposed
    "confidence": 0,  # SCORE  0.0 <= confidence <= 1.0
    "ignore": 0
}

# evaluations
eval_struct = {
    "object_id": 0,
    "detect_id": 0,
    "iou": None,  # IoU of the detection
    # If true then this is the one
    "true_positive": 0,
    "score": 0,  # SCORE/confidence - for convenience from detects

    "precision": 0,
    "recall": 0
}


# =============================================================================
# Utility Functions

def iou_bbox(ltrb1, ltrb2, wh1=None, wh2=None) -> float:
    # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    assert ltrb1[0] < ltrb1[2]
    assert ltrb1[1] < ltrb1[3]
    assert ltrb2[0] < ltrb2[2]
    assert ltrb2[1] < ltrb2[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(ltrb1[0], ltrb2[0])
    y_top = max(ltrb1[1], ltrb2[1])
    x_right = min(ltrb1[2], ltrb2[2])
    y_bottom = min(ltrb1[3], ltrb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    if wh1 is not None:
        bb1_area = wh1[0] * wh1[1]
    else:
        bb1_area = (ltrb1[2] - ltrb1[0]) * (ltrb1[3] - ltrb1[1])

    if wh2 is not None:
        bb2_area = wh2[0] * wh2[1]
    else:
        bb2_area = (ltrb2[2] - ltrb2[0]) * (ltrb2[3] - ltrb2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    # print(f"intersection_area={intersection_area}, bb1_area={bb1_area}, bb2_area={bb2_area}, "
    #       f"denom=-{float(bb1_area + bb2_area - intersection_area)}, iou={iou}, "
    #       f"ltrb2={ltrb2}, ltrb2{ltrb2}, wh1={wh1}, wh2={wh2}")

    # Round because due to rounding error we can get values slightly over 1 at extreme decimals
    iou = round(iou, 6)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def iou_mask(mask1, mask2) -> float:
    # Use sklearn or scipy or the vectorized numpy stuff
    return 0.0


def compute_ious(truths, detects):
    """
    Compute all the ious for this level.
    :param truths: Array of truths in truth struct format
    :param detects: Array of detections in detect struct format
    :return: A list of tuples of (iou, object, detect)
    """

    # TODO: Vectorize!!!

    # Vectorized version from
    # https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    # def np_vec_no_jit_iou(boxes1, boxes2):
    #     def run(bboxes1, bboxes2):
    #         x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    #         x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    #         xA = np.maximum(x11, np.transpose(x21))
    #         yA = np.maximum(y11, np.transpose(y21))
    #         xB = np.minimum(x12, np.transpose(x22))
    #         yB = np.minimum(y12, np.transpose(y22))
    #         interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    #         boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    #         boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    #         iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    #         return iou
    #
    #     tic = time()
    #     run(boxes1, boxes2)
    #     toc = time()
    #     return toc - tic

    ious = []
    for obj in truths:
        for det in detects:
            iou = iou_bbox(obj['ltrb'], det['ltrb'], obj.get('wh', None), det.get('wh', None))
            if iou > 0.0:
                ious.append((iou, obj, det))

    # Now, sort them based on iou.  We'll do this to filter later
    ious.sort(key=lambda x: x[0], reverse=True)

    return ious


def find_best_detects(truths, detects, iou_threshold=0.5):
    """
    This function finds the "best" (based on IoU) pairings between a set of target truths
    and a set of potential detections.  There may by 0 or more detections per truth. We generate
    at most one true positive for each truth and any remaining detections are considered false
    positives.
    NOTE: We assume all the boxes are in the same image for the same label.
    :param truths:  Truths - In SAME image, SAME label
    :param detects:  Potential detects - In SAME image, SAME label
    :param iou_threshold: Threshold for completely disregarding an image base on iou.
    :return: A list of of evaluations of the detections.
    """

    # Compute the iou for every truth/detect pair and get the ordered (descending) list.
    ious = compute_ious(truths, detects)

    # Match the detects against the truths. We have two sets: one to track truths for which we
    # haven't found a detect (remain_truths) and one set for the detects we haven't matched
    # against against a truth (remain_detects). The algorithm is to walk ALL the iou
    # calculations, match them up, and whittle down the truths and detects until we run out.
    remain_truths = {t['id']: t for t in truths}
    remain_detects = {d['id']: d for d in detects}

    evaluations = []
    for iou, truth, det in ious:
        if det['id'] in remain_detects:
            true_positive = 0
            if truth['id'] in remain_truths:
                true_positive = 1
                del remain_truths[truth['id']]

            # If the IoU is less than the threshold it is a false positive
            if iou < iou_threshold:
                true_positive = 0

            # Record what we found
            evaluations.append({
                "object_id": truth['id'],
                "detect_id": det['id'],
                "iou": iou,
                "true_positive": true_positive,
                "score": det['confidence']
            })
            del remain_detects[det['id']]

    # We may have some detects that didn't even get an IoU. They are false positives.
    for det in remain_detects.values():
        evaluations.append({
            "object_id": -1,
            "detect_id": det['id'],
            "iou": 0.0,
            "true_positive": False,
            "score": det['confidence']
        })

    # Double check that we don't have duplicate detects. This is for debugging.
    detect_counts = defaultdict(list)
    for e in evaluations:
        obj_id = e['object_id']
        if e['true_positive']:
            detect_counts[obj_id].append(e['detect_id'])
        if obj_id != -1 and len(detect_counts[obj_id]) > 1:
            print(f"ERROR object={obj_id} has {detect_counts[obj_id]} counts")

    return evaluations


def add_bad_detects(detects):
    """
    Generates a series of evaluations from the detections as not true positive detections.
    :param detects: The detections to turn into not true_positive evaluations.
    :return: The evaluations.
    """
    evaluations = []
    for det in detects:
        evaluations.append({
            "object_id": -1,
            "detect_id": det['id'],
            "iou": 0.0,
            "true_positive": False,
            "score": det['confidence']
        })
    return evaluations


def compute_precision_and_recall(evaluations, num_truths) -> None:
    """
    Computes the precision and recall for each entry down the list following the methods described online.
    The precision and recall is added to each evaluation entry.
    :param evaluations: The evaluations in the desired order.
    :param num_truths: The number of truths for computation of recall.
    """
    # We assume they are in the right order
    tp = 0
    fp = 0
    for e in evaluations:
        if e['true_positive'] > 0:
            tp += 1
        else:
            fp += 1
        e['precision'] = tp / float(tp + fp)
        if num_truths > 0:
            e['recall'] = tp / num_truths
        else:
            e['recall'] = 0.0


def interpolated_stepped_average(evals, count=11) -> float:
    """
    This generates an average from a series of points by finding a series of points at
    evenly spaced intervals and then averaging.  The "stepped" refers to the fact that
    we "smooth" the curve so it is never increasing. Thus any point to the left is greater than
    or equal to any point on the right.  (Notice right to left direction.)

    This is based on the Mean Average Precision method.
    See: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    :param evals: The evaluations to use for the interpolated average.
    :param count: The number of intervals from [0.0, 1.0].  Normally 11.
    :return: The interpolated average.
    """
    if len(evals) == 0:
        return 0.0

    # Walk the list backwards creating an entry at every threshold.
    precisions = [1.0] * count
    idx = count - 1
    recall = 1.0
    recall_delta = 1.0 / (count - 1)

    # Find our starting index.
    # NOTE: We use < because Pascal VOC says "exceeds" in:
    # "The precision at each recall level r is interpolated by taking
    # the maximum precision measured for a method for which
    # the corresponding recall exceeds r."

    while evals[-1]['recall'] < round(recall, 4):
        precisions[idx] = 0.0
        idx -= 1
        recall -= recall_delta

    # Get down to precisions
    for e in reversed(evals):
        while recall > e['recall']:
            precisions[idx] = e['precision']
            idx -= 1
            recall -= recall_delta

    # TODO: Add some logging for unit testing?
    print(f"Precisions: {precisions}")
    return float(np.mean(precisions))


def compute_all_average_precisions(evals_by_label, truth_counts):
    """
    Computes the average precision for each class and return a dict of precision values.
    :param evals_by_label:
    :param truth_counts:
    :return:
    """
    precisions = {}

    for label, evals in evals_by_label.items():
        compute_precision_and_recall(evals, truth_counts[label])
        # TODO: Should we support 101 points?
        precisions[label] = interpolated_stepped_average(evals, 11)

    return precisions


def compute_mean_average_precision(evals_by_label, truth_counts):
    """
    Computes the mean average precision for all labels.
    :param evals_by_label:
    :param truth_counts:
    :return: The mean average precision.
    """
    return np.mean(list(compute_all_average_precisions(evals_by_label, truth_counts).values()))


class MAPEvaluator:
    def __init__(self):
        """
        Initializes an empty evaluator
        """
        # TODO: For now we only do bbox
        self.type = 'bbox'
        # dicts of: image_id -> label -> list(object)
        self.truths = defaultdict(lambda: defaultdict(list))
        self.detects = defaultdict(lambda: defaultdict(list))
        self.truth_counts = defaultdict(int)
        self.detect_counts = defaultdict(int)
        self.eval_by_label = {}

    def add_ground_truth(self, truths):
        """
        Add in ground truth data in the format:
        {
            "id": 0,
            "image_id": 0,
            "label": 0,
            "ltrb": None,
            "mask": None,
            "keypoints": None,
            "ignore": 0
        }
        :param truths: The truths.
        :return:
        """
        for o in truths:
            self.truths[o['image_id']][o['label']].append(o)
            self.truth_counts[o['label']] += 1

    def add_detections(self, detects):
        """
        Add in detections in the format:
        {
            "id": 0,
            "image_id": 0,
            "ltrb": None,
            "mask": None,
            "keypoints": None,
            "label": 0,  # Proposed
            "confidence": 0,  # SCORE  0.0 <= confidence <= 1.0
            "ignore": 0
        }
        :param detects:
        :return:
        """
        for d in detects:
            self.detects[d['image_id']][d['label']].append(d)
            self.detect_counts[d['label']] += 1

    def clear_detections(self) -> None:
        """
        Clears out detections.
        """
        self.detects = defaultdict(lambda: defaultdict(list))

    def mean_average_precision(self):
        """
        Compute the mean average precision based on the currently detections and ground truth data.
        :return:
        """
        eval_by_label = defaultdict(list)

        # Build a unified list of classes
        truth_labels = set([j for i in self.truths.values() for j in i.keys()])

        # Step 1: Get the detections matched up to the truths for each image
        for image_id, classes in self.truths.items():
            for label, truths in classes.items():
                evaluations = find_best_detects(truths, self.detects[image_id][label])

                # Add these to the evaluations for this label
                eval_by_label[label].extend(evaluations)

            # Now, within this image we have detects that are totally bogus.  They are false positives.
            bad_labels = set(self.detects[image_id].keys()) - set(classes.keys())
            for label in bad_labels:
                eval_by_label[label].extend(add_bad_detects(self.detects[image_id][label]))

        # Step 2: Find the average precision of each and get the mean across all classes
        self.eval_by_label = eval_by_label
        return compute_mean_average_precision(eval_by_label, self.truth_counts)

    def dump_debug(self):
        print("Truths:")
        print(json.dumps(self.truths, indent=4))
        print("Truth Counts:")
        print(json.dumps(self.truth_counts, indent=4))
        print("Detects:")
        print(json.dumps(self.detects, indent=4))
        print("Detect Count:")
        print(json.dumps(self.detect_counts, indent=4))
