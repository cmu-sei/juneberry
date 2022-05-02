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

        result["bbox"]["mAP"] = round(coco_metrics["mAP_coco"], DECIMAL_PLACES)
        result["bbox"]["mAP_50"] = round(coco_metrics["mAP_50"], DECIMAL_PLACES)
        result["bbox"]["mAP_75"] = round(coco_metrics["mAP_75"], DECIMAL_PLACES)
        result["bbox"]["mAP_s"] = round(coco_metrics["mAP_small"], DECIMAL_PLACES)
        result["bbox"]["mAP_m"] = round(coco_metrics["mAP_medium"], DECIMAL_PLACES)
        result["bbox"]["mAP_l"] = round(coco_metrics["mAP_large"], DECIMAL_PLACES)

        result["bbox"]["mdAP_localisation"] = round(tide_metrics["mdAP_localisation"], DECIMAL_PLACES)
        result["bbox"]["mdAP_classification"] = round(tide_metrics["mdAP_classification"], DECIMAL_PLACES)
        result["bbox"]["mdAP_both"] = round(tide_metrics["mdAP_both"], DECIMAL_PLACES)
        result["bbox"]["mdAP_duplicate"] = round(tide_metrics["mdAP_duplicate"], DECIMAL_PLACES)
        result["bbox"]["mdAP_background"] = round(tide_metrics["mdAP_background"], DECIMAL_PLACES)
        result["bbox"]["mdAP_missed"] = round(tide_metrics["mdAP_missed"], DECIMAL_PLACES)
        result["bbox"]["mdAP_fp"] = round(tide_metrics["mdAP_fp"], DECIMAL_PLACES)
        result["bbox"]["mdAP_fn"] = round(tide_metrics["mdAP_fn"], DECIMAL_PLACES)

        for key, value in coco_metrics.items():
            if not key.startswith("mAP"):
                result["bbox_per_class"]["mAP_" + key] = round(value, DECIMAL_PLACES)

        result["summary"]["pr_auc"] = round(summary_metrics["pr_auc"], DECIMAL_PLACES)
        result["summary"]["pc_auc"] = round(summary_metrics["pc_auc"], DECIMAL_PLACES)
        result["summary"]["rc_auc"] = round(summary_metrics["rc_auc"], DECIMAL_PLACES)
        result["summary"]["max_r"] = round(summary_metrics["max_r"], DECIMAL_PLACES)
        result["summary"]["ap"] = round(summary_metrics["ap"], DECIMAL_PLACES)
        result["summary"]["tp"] = round(summary_metrics["prediction_types"]["tp"], DECIMAL_PLACES)
        result["summary"]["fp"] = round(summary_metrics["prediction_types"]["fp"], DECIMAL_PLACES)
        result["summary"]["fn"] = round(summary_metrics["prediction_types"]["fn"], DECIMAL_PLACES)

        return result
