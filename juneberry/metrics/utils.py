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

import json
from typing import Dict, Tuple

import brambox as bb
from pandas import DataFrame

from juneberry.config import coco_utils


def get_df(anno: Dict, det: Dict) -> Tuple[DataFrame, DataFrame]:
    """
    Create brambox-compatible DataFrames to be used in Metrics calls.
    :param anno: a dict of annotations in COCO format
    :param det: a dict of detections in COCO format
    :return: the annotations and detections DataFrames
    """
    # TODO don't need to make an anno_parser every time, it doesn't depend on anno or det
    anno_parser = bb.io.parser.annotation.CocoParser(parse_image_names=False)
    anno_parser.deserialize(json.dumps(anno))
    anno_df = anno_parser.get_df()

    det_parser = bb.io.parser.detection.CocoParser(class_label_map=coco_utils.get_class_label_map(anno))
    det_parser.deserialize(json.dumps(det))
    det_df = det_parser.get_df()

    return anno_df, det_df
