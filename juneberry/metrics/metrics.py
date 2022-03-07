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

import logging
from typing import Dict

import brambox as bb
from pandas.core.frame import DataFrame

from juneberry.metrics.metrics_utils import MetricsUtils


logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self,
                 iou_threshold,
                 tp_threshold) -> None:
        """
        Initialize a Metrics object using annotations and detections files in
        COCO JSON format.
        :param iou_threshold: The iou threshold
        :param tp_threshold: The tp threshold
        :return: None
        """
        self.iou_threshold = iou_threshold
        self.tp_threshold = tp_threshold

    def __call__(self, anno: Dict, det: Dict) -> Dict:
        self.det, self.anno = MetricsUtils.load_det_and_anno(det, anno)
        return self.as_dict(self.tp_threshold)

    def as_dict(self, tp_threshold):
        result = {
            "prc": self.prc(),
            "ap": self.ap(),
            "max_r": self.max_r(),
            "fscore": self.fscore(),
            "pr_auc": self.pr_auc(),
            "pc_auc": self.pc_auc(),
            "rc_auc": self.rc_auc(),
            "prediction_types": self.prediction_types(tp_threshold),
        }
        return result

    def _prc_df(self) -> DataFrame:
        return bb.stat.pr(self.det,
                          self.anno,
                          self.iou_threshold)

    def prc(self):
        """
        Get the precision / recall / confidence values for this
        Metrics object.
        :returns: an ndarray of ndarrays. For each ndarray,
        ndarray[0] => precision, ndarray[1] => recall,
        ndarray[2] => confidence
        """
        return Metrics.df_to_ndarray(self._prc_df())

    def ap(self) -> float:
        return bb.stat.ap(self._prc_df())

    def max_r(self) -> float:
        return bb.stat.peak(self._prc_df(), y="recall")["recall"]

    def _fscore_df(self, beta: int = 1) -> DataFrame:
        return bb.stat.fscore(self._prc_df(), beta)

    def fscore(self, beta: int = 1):
        """
        Get the f1 / recall / confidence values for this
        Metrics object.
        :returns: an ndarray of ndarrays. For each ndarray,
        ndarray[0] => f1, ndarray[1] => recall,
        ndarray[2] => confidence
        """
        return Metrics.df_to_ndarray(self._fscore_df())

    def pr_auc(self) -> float:
        """
        Get the precision-recall area under curve.
        :return: PR AUC float
        """
        return bb.stat.auc(self._prc_df(),
                           x="recall",
                           y="precision")

    def pc_auc(self) -> float:
        """
        Get the precision-confidence area under curve.
        :return: PC AUC float
        """
        return bb.stat.auc(self._prc_df(),
                           x="confidence",
                           y="precision")

    def rc_auc(self) -> float:
        """
        Get the recall-confidence area under curve.
        :return: RC AUC float
        """
        return bb.stat.auc(self._prc_df(),
                           x="confidence",
                           y="recall")

    def prediction_types(self, tp_threshold: float) -> Dict[str, int]:
        """
        Find the number of TP, FP, and FN given this Metrics object's
        annotations and detections.
        :param tp_threshold: The TP threshold
        :return: Dict containing TP, FP, and FN values
        """
        result = {}
        match_det, match_anno = bb.stat.match_box(
            self.det, self.anno, tp_threshold)
        result["tp"] = match_det["tp"].values.sum()
        result["fp"] = match_det["fp"].values.sum()  # or could just be ~tp?
        # False negatives are annotations with no detections;
        # i.e. rows in the annotations DataFrame with NaN in the
        # detections column.
        result["fn"] = match_anno["detection"].isna().sum()
        return result

    @staticmethod
    def df_to_ndarray(df: DataFrame):
        return DataFrame.to_numpy(df)
