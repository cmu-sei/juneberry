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

import csv
import logging
from pathlib import Path
from typing import Dict, List

import brambox as bb

import juneberry.metrics.metrics as metrics


logger = logging.getLogger("juneberry.metrics.coco_metrics")


class CocoMetrics(metrics.Metrics):

    def __init__(self,
                 anno_file: Path,
                 det_file: Path,
                 model_name: str,
                 dataset_name: str,
                 iou_threshold: float = 0.5) -> None:
        """
        Initialize a CocoMetrics object using annotations and detections files in
        COCO JSON format.
        :param anno_file: The annotations file in COCO JSON format.
        :param det_file: The detections file in COCO JSON format.
        :param model_name: The model name
        :param dataset_name: The dataset name
        :param iou_threshold: The iou threshold
        :return: None
        """
        super().__init__(anno_file, det_file, model_name, dataset_name, iou_threshold)
        self._eval = bb.eval.COCO(self.det, self.anno, max_det=100, tqdm=False)

    @property
    def mAP(self) -> float:
        return self._eval.mAP_coco

    @property
    def mAP_50(self) -> float:
        return self._eval.mAP_50

    @property
    def mAP_75(self) -> float:
        return self._eval.mAP_75

    @property
    def mAP_small(self) -> float:
        return self._eval.mAP_small

    @property
    def mAP_medium(self) -> float:
        return self._eval.mAP_medium

    @property
    def mAP_large(self) -> float:
        return self._eval.mAP_large

    @property
    def mAP_per_class(self) -> Dict[str, float]:
        # The remaining data to be added to the CSV is on a per-class basis,
        # meaning it is a function of the number of classes in the data.
        # Loop through every class in the data. The class_label_map is used
        # to enforce ordering of the labels.

        # Brambox wanted the class_label_map to start with label 1, so restore
        # the item at the end to its rightful place at position 0.
        result = {}

        self.class_label_map.insert(0, self.class_label_map.pop())

        for label in self.class_label_map:
            # Add the mAP data for that label to the result dict.
            try:
                result[label] = self._eval.AP_coco[label]
                logger.info(f"        {label}: {self._eval.AP_coco[label]}")
            # Handle cases where there might not be mAP data for that label.
            except KeyError:
                logger.info(f"        {label}: N/A")

        return result


    def as_dict(self) -> Dict:
        """
        Convenience method to return interesting metrics
        in a dictionary.
        :return: Dict
        """
        return {
            "mAP": self.mAP,
            "mAP_50": self.mAP_50,
            "mAP_75": self.mAP_75,
            "mAP_s": self.mAP_small,
            "mAP_m": self.mAP_medium,
            "mAP_l": self.mAP_large,
        }

    @staticmethod
    def export(metrics: List[metrics.Metrics],
               output_file: Path = Path("eval_metrics.csv")) -> None:
        """
        This function is responsible for summarizing the metrics generated
        during the current execution of this script. There are two aspects to
        this summary: logging messages and writing the data to CSV.
        :param output_file: The name of the output CSV file
        :param metrics: The list of metrics that were plotted.
        :return: None
        """

        # Write to the CSV file.
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header row for the CSV file.
            writer.writerow(["model",
                             "dataset",
                             "pr_auc",
                             "pc_auc",
                             "rc_auc",
                             "max_r",
                             "average mAP",
                             "class",
                             "per class mAP",
                             ])

            # Loop through every model in the plotted model list.
            for m in metrics:

                # Start constructing the row of CSV data.
                row = [m.model_name,
                       m.dataset_name,
                       m.pr_auc,
                       m.pc_auc,
                       m.rc_auc,
                       m.max_r,
                       m.mAP,
                       ]

                # Log messages for all the data that was just added to the CSV.
                logger.info(
                    f"  Model: {m.model_name}    Dataset: {m.dataset_name}")
                logger.info(f"    PR_AUC: {round(m.pr_auc, 3)}, "
                            f"PC_AUC: {round(m.pc_auc, 3)}, "
                            f"RC_AUC: {round(m.rc_auc, 3)}")
                logger.info(f"      max recall: {m.max_r}")
                logger.info(f"      mAP: {m.mAP}")
                logger.info(f"      mAP (per class):")

                # The remaining data to be added to the CSV is on a per-class
                # basis, meaning it is a function of the number of classes in
                # the data. Loop through every class in the data. The
                # class_label_map is used to enforce ordering of the labels.

                # Work with a copy of the class label map so we don't
                # clobber the brambox-friendly version that lives in
                # the Metrics object.
                clm_copy = m.class_label_map.copy()

                # Brambox wanted the class_label_map to start with label 1,
                # so restore the item at the end to its rightful place
                # at position 0.
                clm_copy.insert(0, clm_copy.pop())

                for label in clm_copy:
                    row.append(label)

                    # Add the mAP data for that label to the CSV row.
                    try:
                        row.append(m._coco.AP_coco[label])
                        logger.info(
                            f"        {label}: {m._coco.AP_coco[label]}")

                    # Handle cases where there might not be mAP data
                    # for that label.
                    except KeyError:
                        row.append(None)
                        logger.info(f"        {label}: N/A")

                # Construction of the CSV row is complete, so write it
                # to the CSV file.
                writer.writerow(row)

        logger.info(f"Metrics have been saved to {output_file}")

    def prediction_types(self, tp_threshold: float) -> Dict[str, int]:
        """
        Find the number of TP, FP, and FN given this Metrics object's
        annotations and detections.
        :param tp_threshold: The TP threshold
        :return: Dict containing TP, FP, and FN values
        """
        result = {}
        match_det, match_anno = bb.stat.match_box(self.det, self.anno, tp_threshold)
        result["tp"] = match_det["tp"].values.sum()
        result["fp"] = match_det["fp"].values.sum()  # or could just be ~tp?
        # False negatives are annotations with no detections;
        # i.e. rows in the annotations DataFrame with NaN in the
        # detections column.
        result["fn"] = match_anno["detection"].isna().sum()
        return result
