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

from juneberry.metrics.metrics_utils import MetricsUtils

logger = logging.getLogger(__name__)


class TideMetrics:

    def __init__(self,
                 pos_thresh: float,
                 bg_thresh: float,
                 max_det: int,
                 area_range_min: int,
                 area_range_max: int,
                 tqdm: bool) -> None:

        """
        Initialize a TideMetrics object using annotations and detections files in
        COCO JSON format.
        :param pos_thresh: The iou threshold
        :param area_range: The COCO area range ("small", "medium", "large", "all")
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

    def __call__(self, anno: Dict, det: Dict) -> Dict:
        self.det, self.anno = MetricsUtils.load_det_and_anno(det, anno)
        self.tide = bb.eval.TIDE(self.det,
                                 self.anno,
                                 area_range=(self.area_range_min, self.area_range_max),
                                 max_det=self.max_det,
                                 pos_thresh=self.iou_threshold,
                                 bg_thresh=self.bg_thresh,
                                 tqdm=self.tqdm)
        return self.tide.mdAP
