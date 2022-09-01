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

import numpy as np

logger = logging.getLogger(__name__)


class DT2NoOp:
    """
    Example of a (no-operation) transformer which demonstrates ALL available extension points when
    building your own DT2 Transform class.
    """
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        return box

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_polygons(self, polygons: list) -> list:
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class DT2Logger:
    def __init__(self, msg="None"):
        self.msg = msg

    def apply_coords(self, coords: np.ndarray):
        logger.info(f"apply_coords: msg={self.msg}, coords-type={type(coords)}")
        return coords

    def apply_polygons(self, polygons: list) -> list:
        logger.info(f"apply_polygons: msg={self.msg}, polygons-type={type(polygons)}")
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        logger.info(f"apply_segmentation: msg={self.msg}, segmentation-type={type(segmentation)}")
        return segmentation
