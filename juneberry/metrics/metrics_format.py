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

logger = logging.getLogger(__name__)


class Coco:
    def __init__(self):
        pass

    def __call__(self, metrics: Dict) -> Dict:
        result = {
            "bbox": {},
            "bbox_per_class": {}
        }

        result["bbox"]["mAP"] = metrics["mAP_coco"]
        result["bbox"]["mAP_50"] = metrics["mAP_50"]
        result["bbox"]["mAP_75"] = metrics["mAP_75"]
        result["bbox"]["mAP_s"] = metrics["mAP_small"]
        result["bbox"]["mAP_m"] = metrics["mAP_medium"]
        result["bbox"]["mAP_l"] = metrics["mAP_large"]

        for key, value in metrics.items():
            if not key.startswith("mAP"):
                result["bbox_per_class"]["mAP_" + key] = value

        return result
