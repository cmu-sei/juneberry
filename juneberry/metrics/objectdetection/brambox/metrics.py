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
This module provides object detection plugin classes to be used with Juneberry.
Add one or more of these plugins to the "evaluation_metrics" section of your
model config. When the MetricsManager is called with annotations and detections,
the metrics will be computed.
"""
import logging
from typing import Dict

import brambox as bb
from pandas.core.frame import DataFrame

from juneberry.metrics.objectdetection.brambox.utils import get_df

logger = logging.getLogger(__name__)


class Coco:

    def __init__(self,
                 iou_threshold: float,
                 max_det: int,
                 tqdm: bool) -> None:
        """
        Initialize a Coco metrics object
        :param iou_threshold: The iou threshold
        :param max_det: The maximum detections
        :param tqdm: Display progress bar
        :return: None
        """
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.tqdm = tqdm

    def __call__(self, anno: Dict, det: Dict):
        anno_df, det_df = get_df(anno, det)
        self.coco = bb.eval.COCO(det_df, anno_df, max_det=self.max_det, tqdm=self.tqdm)
        return self.get_metrics()

    def get_metrics(self) -> dict:
        """
        Get the metrics.
        :return: the metrics
        """
        mAP = self.coco.mAP.to_dict()
        ap_50 = self.coco.AP_50.to_dict()
        ap_75 = self.coco.AP_75.to_dict()
        # Use '|' operator when we upgrade to Python 3.9
        return {**mAP, **ap_50, **ap_75}


class Tide:

    def __init__(self,
                 pos_thresh: float,
                 bg_thresh: float,
                 max_det: int,
                 area_range_min: int,
                 area_range_max: int,
                 tqdm: bool) -> None:
        """
        Initialize a Tide metrics object using annotations and detections files in
        COCO JSON format.
        :param pos_thresh: The iou threshold
        :param area_range_min: The min of COCO area range
        :param area_range_max: The max of COCO area range
        :param bg_thresh: The background iou threshold
        :param tqdm: display progress bar
        :return: None
        """
        self.max_det = max_det
        self.iou_threshold = pos_thresh
        self.area_range_min = area_range_min
        self.area_range_max = area_range_max
        self.bg_thresh = bg_thresh
        self.tqdm = tqdm

    def __call__(self, anno: Dict, det: Dict) -> dict:
        anno_df, det_df = get_df(anno, det)

        self.tide = bb.eval.TIDE(det_df,
                                 anno_df,
                                 area_range=(self.area_range_min, self.area_range_max),
                                 max_det=self.max_det,
                                 pos_thresh=self.iou_threshold,
                                 bg_thresh=self.bg_thresh,
                                 tqdm=self.tqdm)
        return self.get_metrics()

    def get_metrics(self) -> dict:
        return self.tide.mdAP.to_dict()


class Summary:

    def __init__(self,
                 iou_threshold,
                 tp_threshold) -> None:
        """
        Initialize a Summary metrics object using annotations and detections files in
        COCO JSON format.
        :param iou_threshold: The iou threshold
        :param tp_threshold: The tp threshold
        :return: None
        """
        self.iou_threshold = iou_threshold
        self.tp_threshold = tp_threshold

    def __call__(self, anno: Dict, det: Dict) -> dict:
        self.anno_df, self.det_df = get_df(anno, det)
        return self.get_metrics(self.tp_threshold)

    def get_metrics(self, tp_threshold) -> dict:
        result = {
            "prc": self.prc(),
            "prc_df": self.prc_df(),
            "ap": self.ap(),
            "max_r": self.max_r(),
            "fscore": self.fscore(),
            "pr_auc": self.pr_auc(),
            "pc_auc": self.pc_auc(),
            "rc_auc": self.rc_auc(),
            "prediction_types": self.prediction_types(tp_threshold),
        }
        return result

    def prc_df(self) -> DataFrame:
        return bb.stat.pr(self.det_df,
                          self.anno_df,
                          self.iou_threshold)

    def prc(self):
        """
        Get the precision / recall / confidence values for this
        Metrics object.
        :returns: an ndarray of ndarrays. For each ndarray,
        ndarray[0] => precision, ndarray[1] => recall,
        ndarray[2] => confidence
        """
        return DataFrame.to_numpy(self.prc_df())

    def ap(self) -> float:
        return bb.stat.ap(self.prc_df())

    def max_r(self) -> float:
        return bb.stat.peak(self.prc_df(), y="recall")["recall"]

    def _fscore_df(self, beta: int = 1) -> DataFrame:
        return bb.stat.fscore(self.prc_df(), beta)

    def fscore(self, beta: int = 1):
        """
        Get the f1 / recall / confidence values for this
        Metrics object.
        :returns: an ndarray of ndarrays. For each ndarray,
        ndarray[0] => f1, ndarray[1] => recall,
        ndarray[2] => confidence
        """
        return DataFrame.to_numpy(self._fscore_df(beta))

    def pr_auc(self) -> float:
        """
        Get the precision-recall area under curve.
        :return: PR AUC float
        """
        return bb.stat.auc(self.prc_df(),
                           x="recall",
                           y="precision")

    def pc_auc(self) -> float:
        """
        Get the precision-confidence area under curve.
        :return: PC AUC float
        """
        return bb.stat.auc(self.prc_df(),
                           x="confidence",
                           y="precision")

    def rc_auc(self) -> float:
        """
        Get the recall-confidence area under curve.
        :return: RC AUC float
        """
        return bb.stat.auc(self.prc_df(),
                           x="confidence",
                           y="recall")

    def prediction_types(self, tp_threshold: float) -> Dict[str, int]:
        """
        Find the number of TP, FP, and FN given this Metrics object's
        annotations and detections.
        :param tp_threshold: The TP threshold
        :return: dict containing TP, FP, and FN values
        """
        result = {}
        match_det, match_anno = bb.stat.match_box(
            self.det_df, self.anno_df, tp_threshold)
        result["tp"] = match_det["tp"].values.sum()
        result["fp"] = match_det["fp"].values.sum()  # or could just be ~tp?
        # False negatives are annotations with no detections;
        # i.e. rows in the annotations DataFrame with NaN in the
        # detections column.
        result["fn"] = match_anno["detection"].isna().sum()
        return result
