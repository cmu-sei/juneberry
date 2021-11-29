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
import json
import logging
from pathlib import Path
import tempfile
from typing import Dict, List

import brambox as bb
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas.core.frame import DataFrame

from juneberry.filesystem import EvalDirMgr, ModelManager


logger = logging.getLogger("juneberry.metrics")


class Metrics:

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
    def create_with_filesystem_managers(model_mgr: ModelManager,
                                        eval_dir_mgr: EvalDirMgr,
                                        iou_threshold: float = 0.5) -> Metrics:
        """
        Create a Metrics object using a model manager and eval dir manager.
        :param model_mgr: The model manager
        :param eval_dir_mgr: The eval dir mgr
        :param iou_threshold: iou_threshold
        :return: a Metrics object
        """
        model_name = model_mgr.model_name
        dataset_name = eval_dir_mgr.get_dir().stem

        anno_file = Path(eval_dir_mgr.get_manifest_path())
        det_file = Path(eval_dir_mgr.get_detections_path())

        return Metrics(anno_file,
                       det_file,
                       model_name,
                       dataset_name,
                       iou_threshold)

    @staticmethod
    def create_with_data(anno: Dict,
                         det: Dict,
                         model_name: str = "unknown model",
                         dataset_name: str = "unknown dataset",
                         iou_threshold: float = 0.5) -> Metrics:
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

        return Metrics(Path(anno_file.name),
                       Path(det_file.name),
                       model_name,
                       dataset_name,
                       iou_threshold)

    @property
    def _coco(self):
        return bb.eval.COCO(self.det,
                            self.anno,
                            max_det=100,
                            tqdm=False)

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
    def mAP(self) -> float:
        return self._coco.mAP_coco

    @property
    def mAP_50(self) -> float:
        return self._coco.mAP_50

    @property
    def mAP_75(self) -> float:
        return self._coco.mAP_75

    @property
    def mAP_small(self) -> float:
        return self._coco.mAP_small

    @property
    def mAP_medium(self) -> float:
        return self._coco.mAP_medium

    @property
    def mAP_large(self) -> float:
        return self._coco.mAP_large

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
                result[label] = self._coco.AP_coco[label]
                logger.info(f"        {label}: {self._coco.AP_coco[label]}")
            # Handle cases where there might not be mAP data for that label.
            except KeyError:
                logger.info(f"        {label}: N/A")

        return result

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
    def df_to_ndarray(df: DataFrame) -> ndarray[ndarray]:
        return DataFrame.to_numpy(df)

    @staticmethod
    def export(metrics: List[Metrics],
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


class MetricsPlot:

    def __init__(self,
                 xlabel: str = "x",
                 ylabel: str = "y") -> None:
        """
        Initialize a MetricsPlot object.
        :param xlabel: The x-axis label for this MetricsPlot.
        :param ylabel: The y-axis label for this MetricsPlot.
        :return: None
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        # TODO why am I saving metrics? (not currently using them)
        self.metrics = []

    def _format(self) -> None:
        """
        Format this MetricsPlot's figure and axes to Juneberry specifications.
        This can be overridden for custom formatting.
        :return: None
        """
        _, labels = self.ax.get_legend_handles_labels()
        num_legend_labels = len(labels)

        # The dimensions of the figure will adjust slightly depending on how
        # many curves are in the plot. This makes it easier accommodate
        # larger legends.
        dimension = 7 + .1 * num_legend_labels

        # Establish a fixed size for the figure.
        self.fig.set_size_inches(w=dimension, h=dimension)

        # Set the range for the X and Y axes.
        self.ax.set_xlim(0, 1.05)
        self.ax.set_ylim(0, 1.05)

        # Move the axes up slightly to make room for a legend below the plot.
        box = self.ax.get_position()

        # This factor controls the placement of the plot and legend. It
        # scales dynamically based on the number of curves that have been
        # added to the figure.
        factor = .025 * num_legend_labels

        # Use the factor to adjust the position of the plot axes.
        bottom = box.y0 + box.height * factor
        top = box.height * (1 - factor)
        self.ax.set_position([box.x0, bottom, box.width, top])

        # The midpoint of the plot will be used to center the legend.
        x_midpoint = (box.x0 + box.x1) / 2

        # Place the legend.
        self.ax.legend(loc="upper center",
                       bbox_to_anchor=(x_midpoint, -.08 * (1 + factor)),
                       fancybox=True,
                       ncol=1,
                       fontsize="x-small",
                       shadow=True)

    def _plot(self,
              data: DataFrame,
              model_name: str,
              dataset_name: str,
              auc: float,
              iou_threshold: float) -> None:
        """
        Plot a DataFrame according to the specifications for
        MetricPlot objects.
        :param data: The DataFrame to plot
        :param model_name: the model name
        :param dataset_name: the dataset name
        :param auc: the area under curve for this data
        :param iou_threshold: the iou threshold
        :return: None
        """
        xlabel: str = self.ax.get_xlabel()
        ylabel: str = self.ax.get_ylabel()
        self.ax.set_title(self._get_title({
            "ylabel": ylabel,
            "xlabel": xlabel,
            "iou_threshold": iou_threshold,
        }))
        data.plot(xlabel,
                  ylabel,
                  drawstyle="steps",
                  label=self._get_plot_label({
                      "model_name": model_name,
                      "dataset_name": dataset_name,
                      "auc": auc,
                  }),
                  ax=self.ax)
        self._format()

    # override in subclass
    def _get_auc(self, m: Metrics) -> float:
        """
        Select the proper AUC value from the Metrics object
        for this MetricsPlot.
        :param m: the Metrics object
        :return: the AUC float value
        """
        logger.warning("Calling do-nothing superclass implementation "
                       "of _get_auc. Implement this in your subclass.")
        return m.pr_auc

    # override this for a custom title
    @staticmethod
    def _get_title(title_data: Dict) -> str:
        return f"{title_data['ylabel'].capitalize()}-" \
            f"{title_data['xlabel'].capitalize()} " \
            f"Curve (IoU = {title_data['iou_threshold']})"

    # override this for a custom plot label
    @staticmethod
    def _get_plot_label(plot_label_data: Dict) -> str:
        return f"m({plot_label_data['model_name']}) " \
            f"d({plot_label_data['dataset_name']}) " \
            f"(AUC {round(plot_label_data['auc'], 3)})"

    def add_metrics(self, m: Metrics) -> None:
        """
        Add a Metrics object to this MetricsPlot.
        :param m: a Metrics object to add to this MetricsPlot
        :return: None
        """
        self.metrics.append(m)
        self._plot(m._prc_df,
                   m.model_name,
                   m.dataset_name,
                   self._get_auc(m),
                   m.iou_threshold)

    def add_metrics_list(self,
                         ms: List[Metrics]) -> None:
        """
        Add multiple Metrics objects to this MetricsPlot.
        :param ms: a List of Metrics objects to be added to this MetricsPlot
        :return: None
        """
        for m in ms:
            self.add_metrics(m)

    def save(self, output_file: Path = Path("metrics.png")) -> None:
        """
        Save this metrics plot to a file.
        :param output_file: the file to save this MetricPlot figure to
        :return: None
        """
        self.fig.savefig(output_file)


class PrecisionRecallPlot(MetricsPlot):

    def __init__(self) -> None:
        """
        Initialize a new PrecisionRecallPlot.
        :return: None
        """
        super().__init__(xlabel="recall",
                         ylabel="precision")

    def _get_auc(self, m: Metrics) -> float:
        return m.pr_auc


class PrecisionConfidencePlot(MetricsPlot):

    def __init__(self) -> None:
        """
        Initialize a new PrecisionConfidencePlot.
        :return: None
        """
        super().__init__(xlabel="confidence",
                         ylabel="precision")

    def _get_auc(self, m: Metrics) -> float:
        return m.pc_auc


class RecallConfidencePlot(MetricsPlot):

    def __init__(self) -> None:
        """
        Initialize a new RecallConfidencePlot.
        :return: None
        """
        super().__init__(xlabel="confidence",
                         ylabel="recall")

    def _get_auc(self, m: Metrics) -> float:
        return m.rc_auc
