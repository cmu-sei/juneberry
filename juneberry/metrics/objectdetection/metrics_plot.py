#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

logger = logging.getLogger(__name__)


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
    def _get_auc(self, stats: dict) -> float:
        """
        Select the proper AUC value from the Metrics object
        for this MetricsPlot.
        :param stats: a dict containing metrics stats
        :return: the AUC float value
        """
        logger.warning("Calling do-nothing superclass implementation "
                       "of _get_auc. Implement this in your subclass.")
        return stats["pr_auc"]

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

    def add_metrics(self, stats: dict, iou_threshold: float, model_name: str, dataset_name: str) -> None:
        """
        Add metrics to this MetricsPlot.
        :param stats: a dict containing metrics stats
        :param iou_threshold: iou_threshold used when generating common_metrics
        :param model_name: the model name
        :param dataset_name: the dataset name
        :return: None
        """
        self._plot(stats["prc_df"],
                   model_name,
                   dataset_name,
                   self._get_auc(stats),
                   iou_threshold)

    def save(self, output_file: Path = Path("metrics.png")) -> None:
        """
        Save this MetricsPlot to a file.
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

    def _get_auc(self, stats: dict) -> float:
        return stats["pr_auc"]


class PrecisionConfidencePlot(MetricsPlot):

    def __init__(self) -> None:
        """
        Initialize a new PrecisionConfidencePlot.
        :return: None
        """
        super().__init__(xlabel="confidence",
                         ylabel="precision")

    def _get_auc(self, stats: dict) -> float:
        return stats["pc_auc"]


class RecallConfidencePlot(MetricsPlot):

    def __init__(self) -> None:
        """
        Initialize a new RecallConfidencePlot.
        :return: None
        """
        super().__init__(xlabel="confidence",
                         ylabel="recall")

    def _get_auc(self, stats: dict) -> float:
        return stats["rc_auc"]
