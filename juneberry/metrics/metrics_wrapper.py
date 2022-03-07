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
import sys

from pandas.core.frame import DataFrame

import juneberry.loader as loader


logger = logging.getLogger(__name__)


class MetricsToolkit(Enum):
    COCO = 1
    TIDE = 2


class MetricsWrapper:

    @staticmethod
    def _create(anno_file: Path,
                det_file: Path,
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
    def create(anno: Dict,
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