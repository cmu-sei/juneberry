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

DECIMAL_PLACES = 3

logger = logging.getLogger(__name__)


class DefaultFormatter:
    def __init__(self):
        pass

    def __call__(self, metrics: Dict) -> Dict:
        coco_metrics = metrics["juneberry.metrics.metrics.Coco"]
        tide_metrics = metrics["juneberry.metrics.metrics.Tide"]
        summary_metrics = metrics["juneberry.metrics.metrics.Summary"]

        result = {
            "bbox": {},
            "bbox_per_class": {},
            "summary": {}
        }

        result["bbox"]["mAP"] = coco_metrics["mAP_coco"]
        result["bbox"]["mAP_50"] = coco_metrics["mAP_50"]
        result["bbox"]["mAP_75"] = coco_metrics["mAP_75"]
        result["bbox"]["mAP_s"] = coco_metrics["mAP_small"]
        result["bbox"]["mAP_m"] = coco_metrics["mAP_medium"]
        result["bbox"]["mAP_l"] = coco_metrics["mAP_large"]

        result["bbox"]["mdAP_localisation"] = tide_metrics["mdAP_localisation"]
        result["bbox"]["mdAP_classification"] = tide_metrics["mdAP_classification"]
        result["bbox"]["mdAP_both"] = tide_metrics["mdAP_both"]
        result["bbox"]["mdAP_duplicate"] = tide_metrics["mdAP_duplicate"]
        result["bbox"]["mdAP_background"] = tide_metrics["mdAP_background"]
        result["bbox"]["mdAP_missed"] = tide_metrics["mdAP_missed"]
        result["bbox"]["mdAP_fp"] = tide_metrics["mdAP_fp"]
        result["bbox"]["mdAP_fn"] = tide_metrics["mdAP_fn"]

        for key, value in coco_metrics.items():
            if not key.startswith("mAP"):
                result["bbox_per_class"]["mAP_" + key] = value

        result["summary"]["pr_auc"] = summary_metrics["pr_auc"]
        result["summary"]["pc_auc"] = summary_metrics["pc_auc"]
        result["summary"]["rc_auc"] = summary_metrics["rc_auc"]
        result["summary"]["max_r"] = summary_metrics["max_r"]
        result["summary"]["ap"] = summary_metrics["ap"]
        result["summary"]["tp"] = summary_metrics["prediction_types"]["tp"]
        result["summary"]["fp"] = summary_metrics["prediction_types"]["fp"]
        result["summary"]["fn"] = summary_metrics["prediction_types"]["fn"]

        return result
