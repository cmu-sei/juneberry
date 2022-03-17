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
import logging
from pathlib import Path
import tempfile
from typing import Dict, List

import brambox as bb
from pandas.core.frame import DataFrame

#from juneberry.filesystem import EvalDirMgr, ModelManager
#from metrics_manager import MetricsManager

logger = logging.getLogger(__name__)


class MetricsUtils:

    @staticmethod
    def load_dets_and_annos_files(dets_file: Path, annos_file: Path):
        # Load the dets and annos files via brambox into DataFrames
        annos_df = bb.io.load("anno_coco", annos_file.name, parse_image_names=False)
        # NOTE: Loading the detections requires access to the annotations file. That's why we load
        # detections and annotations in the same method; so that the TemporaryFiles dets_file
        # and annos_file both exist for this call.
        dets_df = bb.io.load("det_coco", dets_file.name,
                             class_label_map=MetricsUtils.get_class_label_map(annos_file.name))
        return dets_df, annos_df

    @staticmethod
    def load_dets_and_annos(dets: Dict, annos: Dict):
        with tempfile.NamedTemporaryFile(mode="w+") as dets_file, tempfile.NamedTemporaryFile(mode="w+") as annos_file:
            # Write the dets and annos dicts to temporary files
            json.dump(dets, dets_file)
            json.dump(annos, annos_file)
            dets_file.flush()
            annos_file.flush()
            return MetricsUtils.load_dets_and_annos_files(dets_file, annos_file)

    @staticmethod
    def get_class_label_map(annotations_file: str) -> List[str]:
        """
        This function is responsible for retrieving the class label map from the annotations file.
        The class label map is used to convert the values in the class_label column of the
        detections Dataframe from integers into strings.
        :param annotations_file: The annotations file containing the class label information.
        :return: A List of str containing the classes for each integer label.
        """

        # Open the annotation file and retrieve the information in the
        # categories field.
        with open(annotations_file) as json_file:
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

    @staticmethod
    def df_to_ndarray(df: DataFrame):
        return DataFrame.to_numpy(df)

    # @staticmethod
    # def export(metrics: List[metrics.Metrics],
    #            output_file: Path = Path("eval_metrics.csv")) -> None:
    #     """
    #     This function is responsible for summarizing the metrics generated
    #     during the current execution of this script. There are two aspects to
    #     this summary: logging messages and writing the data to CSV.
    #     :param output_file: The name of the output CSV file
    #     :param metrics: The list of metrics that were plotted.
    #     :return: None
    #     """

    #     # Write to the CSV file.
    #     with open(output_file, 'w', newline='') as f:
    #         writer = csv.writer(f)

    #         # Header row for the CSV file.
    #         writer.writerow(["model",
    #                          "dataset",
    #                          "pr_auc",
    #                          "pc_auc",
    #                          "rc_auc",
    #                          "max_r",
    #                          "average mAP",
    #                          "class",
    #                          "per class mAP",
    #                          ])

    #         # Loop through every model in the plotted model list.
    #         for m in metrics:

    #             # Start constructing the row of CSV data.
    #             row = [m.model_name,
    #                    m.dataset_name,
    #                    m.pr_auc,
    #                    m.pc_auc,
    #                    m.rc_auc,
    #                    m.max_r,
    #                    m.mAP,
    #                    ]

    #             # Log messages for all the data that was just added to the CSV.
    #             logger.info(
    #                 f"  Model: {m.model_name}    Dataset: {m.dataset_name}")
    #             logger.info(f"    PR_AUC: {round(m.pr_auc, 3)}, "
    #                         f"PC_AUC: {round(m.pc_auc, 3)}, "
    #                         f"RC_AUC: {round(m.rc_auc, 3)}")
    #             logger.info(f"      max recall: {m.max_r}")
    #             logger.info(f"      mAP: {m.mAP}")
    #             logger.info(f"      mAP (per class):")

    #             # The remaining data to be added to the CSV is on a per-class
    #             # basis, meaning it is a function of the number of classes in
    #             # the data. Loop through every class in the data. The
    #             # class_label_map is used to enforce ordering of the labels.

    #             # Work with a copy of the class label map so we don't
    #             # clobber the brambox-friendly version that lives in
    #             # the Metrics object.
    #             clm_copy = m.class_label_map.copy()

    #             # Brambox wanted the class_label_map to start with label 1,
    #             # so restore the item at the end to its rightful place
    #             # at position 0.
    #             clm_copy.insert(0, clm_copy.pop())

    #             for label in clm_copy:
    #                 row.append(label)

    #                 # Add the mAP data for that label to the CSV row.
    #                 try:
    #                     row.append(m._coco.AP_coco[label])
    #                     logger.info(
    #                         f"        {label}: {m._coco.AP_coco[label]}")

    #                 # Handle cases where there might not be mAP data
    #                 # for that label.
    #                 except KeyError:
    #                     row.append(None)
    #                     logger.info(f"        {label}: N/A")

    #             # Construction of the CSV row is complete, so write it
    #             # to the CSV file.
    #             writer.writerow(row)

    #     logger.info(f"Metrics have been saved to {output_file}")
