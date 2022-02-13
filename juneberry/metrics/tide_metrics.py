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

# needed to reference Metrics class inside Metrics for typing
# will no longer be necessary starting in Python 3.10
from __future__ import annotations

import csv
from enum import Enum
import juneberry.metrics.metrics as metrics
import logging
from pathlib import Path
from typing import Dict, List

import brambox as bb
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from juneberry.filesystem import EvalDirMgr, ModelManager


logger = logging.getLogger("juneberry.metrics.tide_metrics")


class CocoSizeRanges(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TideMetrics(metrics.Metrics):

    def __init__(self,
                 anno_file: Path,
                 det_file: Path,
                 model_name: str,
                 dataset_name: str,
                 iou_threshold: float = 0.5) -> None:
        """
        Initialize a TideMetrics object using annotations and detections files in
        COCO JSON format.
        :param anno_file: The annotations file in COCO JSON format.
        :param det_file: The detections file in COCO JSON format.
        :param model_name: The model name
        :param dataset_name: The dataset name
        :param iou_threshold: The iou threshold
        :return: None
        """
        super().__init__(anno_file, det_file, model_name, dataset_name, iou_threshold)
        self.area_range = None
        self.background_iou_threshold = 0.1
        self.show_tqdm = True
        self.reset_eval()

    def reset_eval(self) -> None:
        self._eval = bb.eval.TIDE(self.det,
                                  self.anno,
                                  area_range=self.area_range,
                                  pos_thresh=self.iou_threshold,
                                  bg_thresh=self.background_iou_threshold,
                                  tqdm=self.show_tqdm)

    def set_area_range(self, min: int, max: int) -> None:
        self.area_range = (min, max)
        self.reset_eval()

    def _set_area_coco_range(self, range: CocoSizeRanges) -> None:
        min = bb.eval.TIDE.coco_areas[range.value][0]
        max = bb.eval.TIDE.coco_areas[range.value][1]
        self.set_area_range(min, max)

    def set_area_coco_large(self) -> None:
        self._set_area_coco_range(CocoSizeRanges.LARGE)

    def set_area_coco_medium(self) -> None:
        self._set_area_coco_range(CocoSizeRanges.MEDIUM)

    def set_area_coco_small(self) -> None:
        self._set_area_coco_range(CocoSizeRanges.SMALL)

    def set_background_threshold(self, bg_thresh) -> None:
        self.bg_thresh = bg_thresh
        self.reset_eval()

    def set_show_tqdm(self, show_tqdm: bool) -> None:
        self.show_tqdm = show_tqdm
        self.reset_eval()

    @property
    def AP(self):
        return self._eval.AP

    @property
    def mAP(self) -> float:
        return self._eval.mAP

    @property
    def dAP_localisation(self):
        return self._eval.dAP_localisation

    @property
    def mdAP_localisation(self) -> float:
        return self._eval.mdAP_localisation

    @property
    def dAP_classification(self):
        return self._eval.dAP_classification

    @property
    def mdAP_classification(self) -> float:
        return self._eval.mdAP_classification

    @property
    def dAP_both(self):
        return self._eval.dAP_both

    @property
    def mdAP_both(self) -> float:
        return self._eval.mdAP_both

    @property
    def dAP_duplicate(self):
        return self._eval.dAP_duplicate

    @property
    def mdAP_duplicate(self) -> float:
        return self._eval.mdAP_duplicate

    @property
    def dAP_background(self):
        return self._eval.dAP_background

    @property
    def mdAP_background(self):
        return self._eval.mdAP_background

    @property
    def dAP_missed(self):
        return self._eval.dAP_missed

    @property
    def mdAP_missed(self):
        return self._eval.mdAP_missed

    @property
    def dAP_fp(self):
        return self._eval.dAP_fp

    @property
    def mdAP_fp(self):
        return self._eval.mdAP_fp

    @property
    def dAP_fn(self):
        return self._eval.dAP_fn

    @property
    def mdAP_fn(self):
        return self._eval.mdAP_fn
