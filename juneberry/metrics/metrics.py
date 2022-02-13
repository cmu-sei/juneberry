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
import json
import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple

import brambox as bb
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas.core.frame import DataFrame

from juneberry.filesystem import EvalDirMgr, ModelManager


logger = logging.getLogger("juneberry.metrics.metrics")


class MetricsToolkit(Enum):
    COCO = 1
    TIDE = 2


class Metrics:
    def __init__(self,
                 anno_file: Path,
                 det_file: Path,
                 model_name: str,
                 dataset_name: str,
                 iou_threshold: float = 0.5) -> None:
        """
        Initialize a Metrics object using annotations and detections files in
        COCO JSON format.
        :param anno_file: The annotations file in COCO JSON format.
        :param det_file: The detections file in COCO JSON format.
        :param model_name: The model name
        :param dataset_name: The dataset name
        :param iou_threshold: The iou threshold
        :return: None
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.iou_threshold = iou_threshold
        self.class_label_map = Metrics._get_class_label_map(anno_file)
        self.det = bb.io.load("det_coco",
                              str(det_file),
                              class_label_map=self.class_label_map)
        self.anno = bb.io.load("anno_coco",
                               str(anno_file),
                               parse_image_names=False)
        if self.anno.empty:
            raise ValueError("Cannot initialize Metrics with no annotations.")

    @staticmethod
    def _create(anno_file: Path,
                det_file: Path,
                model_name: str,
                dataset_name: str,
                iou_threshold: float,
                toolkit: MetricsToolkit) -> Metrics:
        if (toolkit is MetricsToolkit.COCO):
            from juneberry.metrics.coco_metrics import CocoMetrics
            return CocoMetrics(anno_file, det_file, model_name, dataset_name, iou_threshold)
        elif (toolkit is MetricsToolkit.TIDE):
            from juneberry.metrics.tide_metrics import TideMetrics
            return TideMetrics(anno_file, det_file, model_name, dataset_name, iou_threshold)
        else:
            raise ValueError(f"Invalid MetricsToolkit {toolkit}")

    @staticmethod
    def create_with_filesystem_managers(model_mgr: ModelManager,
                                        eval_dir_mgr: EvalDirMgr,
                                        iou_threshold: float = 0.5,
                                        toolkit: MetricsToolkit = MetricsToolkit.COCO) -> Metrics:
        """
        Create a Metrics object using a model manager and eval dir manager.
        :param model_mgr: The model manager
        :param eval_dir_mgr: The eval dir mgr
        :param iou_threshold: iou_threshold
        :param metrics_toolkit: Which toolkit to use for evaluation
        :return: a Metrics object
        """
        model_name = model_mgr.model_name
        dataset_name = eval_dir_mgr.get_dir().stem

        anno_file = Path(eval_dir_mgr.get_manifest_path())
        det_file = Path(eval_dir_mgr.get_detections_path())

        return Metrics._create(anno_file, det_file, model_name, dataset_name, iou_threshold, toolkit)

    @staticmethod
    def create_with_data(anno: Dict,
                         det: Dict,
                         model_name: str = "unknown model",
                         dataset_name: str = "unknown dataset",
                         iou_threshold: float = 0.5,
                         toolkit: MetricsToolkit = MetricsToolkit.COCO) -> Metrics:
        """
        Create a Metrics object using dictionaries containing
        annotations and detections.
        :param anno: annotations
        :param det: detections
        :param model_name: model name
        :param dataset_name: dataset name
        :param iou_threshold: iou_threshold
        :return: a Metrics object
        """
        anno_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(anno, anno_file)
        anno_file.flush()

        det_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(det, det_file)
        det_file.flush()

        return Metrics._create(anno_file.name, det_file.name, model_name, dataset_name, iou_threshold, toolkit)

    @property
    def _prc_df(self) -> DataFrame:
        return bb.stat.pr(self.det,
                          self.anno,
                          self.iou_threshold)

    @property
    def prc(self) -> ndarray[ndarray]:
        """
        Get the precision / recall / confidence values for this
        Metrics object.
        :returns: an ndarray of ndarrays. For each ndarray,
        ndarray[0] => precision, ndarray[1] => recall,
        ndarray[2] => confidence
        """
        return Metrics.df_to_ndarray(self._prc_df)

    @property
    def ap(self) -> float:
        return bb.stat.ap(self._prc_df)

    @property
    def max_r(self) -> float:
        return bb.stat.peak(self._prc_df, y="recall")["recall"]

    @property
    def _fscore_df(self, beta: int = 1) -> DataFrame:
        return bb.stat.fscore(self._prc_df, beta)

    @property
    def fscore(self, beta: int = 1) -> ndarray[ndarray]:
        """
        Get the f1 / recall / confidence values for this
        Metrics object.
        :returns: an ndarray of ndarrays. For each ndarray,
        ndarray[0] => f1, ndarray[1] => recall,
        ndarray[2] => confidence
        """
        return Metrics.df_to_ndarray(self._fscore_df)

    @property
    def pr_auc(self) -> float:
        """
        Get the precision-recall area under curve.
        :return: PR AUC float
        """
        return bb.stat.auc(self._prc_df,
                           x="recall",
                           y="precision")

    @property
    def pc_auc(self) -> float:
        """
        Get the precision-confidence area under curve.
        :return: PC AUC float
        """
        return bb.stat.auc(self._prc_df,
                           x="confidence",
                           y="precision")

    @property
    def rc_auc(self) -> float:
        """
        Get the recall-confidence area under curve.
        :return: RC AUC float
        """
        return bb.stat.auc(self._prc_df,
                           x="confidence",
                           y="recall")

    @staticmethod
    def df_to_ndarray(df: DataFrame) -> ndarray[ndarray]:
        return DataFrame.to_numpy(df)

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
    def _get_class_label_map(anno_file: Path) -> List[str]:
        """
        This function is responsible for retrieving the class label map from
        the annotations file. The class label map is used to convert the
        values in the class_label column of the detections Dataframe from
        integers into strings.
        :param anno_file: The annotations file containing the class label
        information.
        :return: A List of str containing the classes for each integer label.
        """

        # Open the annotation file and retrieve the information in the
        # categories field.
        with open(anno_file) as json_file:
            categories = json.load(json_file)["categories"]

        # Create an ID list, which contains every integer value that appears
        # as a category in the annotations file.
        id_list = []
        for category in categories:
            id_list.append(category["id"])

        # Set up the class label map such that there is one entry for every
        # possible integer, even if the integer does not appear as a category
        # in the annotations file.
        class_label_map = [None] * (max(id_list) + 1)

        # For the categories that appear in the annotations file, fill in the
        # appropriate entry of the class label map using the string for that
        # integer.
        for category in categories:
            class_label_map[category["id"]] = category["name"]

        # Brambox expects the first item in the class label map to be for
        # label 1, so take the first item (label 0) and move it to the end of
        # the class label map.
        class_label_map.append(class_label_map.pop(0))

        return class_label_map
