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

from prodict import Prodict

from juneberry.config.plugin import Plugin
from juneberry.filesystem import ModelManager
import juneberry.metrics.objectdetection.metrics_manager as metrics_manager
from juneberry.metrics.objectdetection.metrics_plot import PrecisionRecallPlot, PrecisionConfidencePlot, RecallConfidencePlot
from juneberry.reporting.report import Report

logger = logging.getLogger(__name__)

DEFAULT_PC_FILENAME = "pc_curve.png"
DEFAULT_PR_FILENAME = "pr_curve.png"
DEFAULT_RC_FILENAME = "rc_curve.png"


class PRCurve(Report):
    """
    The purpose of this Report subclass is to reproduce the functionality tha was previously
    contained inside the 'jb_plot_pr' script. This report will create three figures:
      1) A Precision-Recall plot, containing one or more curves.
      2) A Precision-Confidence plot, containing one or more curves.
      3) A Recall-Confidence plot, containing one or more curves.
    """
    def __init__(self, output_dir: str = "", iou: float = 0.5, curve_sources: dict = None,
                 tp_threshold: float = 0.8, stats_fqcn: str = "juneberry.metrics.objectdetection.brambox.metrics.Summary"):
        super().__init__(output_str=output_dir)

        # Store some attributes for various parameters that affect the curves.
        self.iou = iou
        self.tp_threshold = tp_threshold
        self.stats_fqcn = stats_fqcn

        # Store the curve sources, which is a dictionary containing pairs of models (keys) and
        # eval datasets (values).
        self.curve_sources = curve_sources

    def create_report(self) -> None:
        """
        This method is responsible for creating the three figures and saving them to the desired
        output directory.
        """
        logger.info(f"Starting to generate PR, PC, and RC curves...")

        if self.curve_sources is None:
            logger.error(f"No curve sources were provided. Nothing to add to the PR plots!")
            return

        # Create empty MetricsPlots
        pr_plot = PrecisionRecallPlot()
        pc_plot = PrecisionConfidencePlot()
        rc_plot = RecallConfidencePlot()

        metrics_config = Plugin.from_dict({
            "fqcn": self.stats_fqcn,
            "kwargs": {
                "iou_threshold": self.iou,
                "tp_threshold": self.tp_threshold
            }
        })

        mm = metrics_manager.MetricsManager([metrics_config])

        # Add a curve to the figure for each model:eval_dataset pair in the curve sources.
        for idx, curve_source in enumerate(self.curve_sources):
            curve_source = Prodict.from_dict(curve_source)
            model_mgr = ModelManager(curve_source.model)
            eval_dir_mgr = model_mgr.get_eval_dir_mgr(curve_source.dataset)

            metrics = mm.call_with_eval_dir_manager(eval_dir_mgr)[self.stats_fqcn]

            model_name = model_mgr.model_name
            dataset_name = eval_dir_mgr.get_dir().stem

            pr_plot.add_metrics(metrics, self.iou, model_name, dataset_name)
            pc_plot.add_metrics(metrics, self.iou, model_name, dataset_name)
            rc_plot.add_metrics(metrics, self.iou, model_name, dataset_name)

        # Save the figures.
        logger.info(f"Saving plot files to {self.output_dir}")
        pr_plot.save(self.output_dir / DEFAULT_PR_FILENAME)
        pc_plot.save(self.output_dir / DEFAULT_PC_FILENAME)
        rc_plot.save(self.output_dir / DEFAULT_RC_FILENAME)
