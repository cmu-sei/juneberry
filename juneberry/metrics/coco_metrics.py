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
import pandas as pd

from juneberry.metrics.metrics_utils import MetricsUtils


logger = logging.getLogger("juneberry.metrics.coco_metrics")


class CocoMetrics:

    def __init__(self,
                 iou_threshold: float,
                 max_det: int,
                 tqdm: bool) -> None:
        """
        Initialize a CocoMetrics object using annotations and detections files in
        COCO JSON format.
        :param iou_threshold: The iou threshold
        :param max_det: The maximum detections
        :param tqdm: Display progress bar
        :return: None
        """
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.tqdm = tqdm

    def __call__(self, anno: Dict, det: Dict):
        self.det, self.anno = MetricsUtils.load_det_and_anno(det, anno)
        self.coco = bb.eval.COCO(self.det,
                                 self.anno,
                                 max_det=self.max_det,
                                 tqdm=self.tqdm)

        return self.as_dict()

    def as_dict(self) -> Dict:
        """
        Convenience method to return interesting metrics
        in a DataFrame.
        :return: Dict
        """
        result = self.coco.mAP.to_dict()

        AP_50 = self.coco.AP_50.to_dict()
        for k, v in AP_50.items():
            new_k = "AP_50_" + k
            result[new_k] = v

        AP_75 = self.coco.AP_75.to_dict()
        for k, v in AP_75.items():
            new_k = "AP_75_" + k
            result[new_k] = v

        return result

