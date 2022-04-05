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
from shutil import copy
import sys
from typing import List

from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.training_output import TrainingOutput
from juneberry.filesystem import ModelManager
from juneberry.reporting.report import Report

logger = logging.getLogger(__name__)


class Summary(Report):
    # TODO: The intent of this class would be to replace jb_summary_report
    def __init__(self, markdown_str: str = "", csv_str: str = "", prediction_files: List = None,
                 plot_files: List = None):
        super().__init__(output_str=markdown_str)

        # Handle the case when an output file is not provided.
        if markdown_str == "":
            self.report_path = self.output_dir / "summary.md"
        else:
            self.report_path = Path(markdown_str)
        logger.info(f"Saving the report to {self.report_path}")

        self.csv_path = None if csv_str == "" else Path(csv_str)
        self.prediction_files = prediction_files
        self.plot_files = plot_files

        self.metric = self._determine_shared_metric() if self.prediction_files is not None else None

        self.report_files_path = None
        markdown_parent = self.report_path.parent
        self.report_files_path = markdown_parent / "report_files"
        if not self.report_files_path.exists():
            self.report_files_path.mkdir(parents=True)

        # One or more prediction files - These go into the table
        # One or more plot files - These go into the Experiment Plots Section
        # Output filename - This is the filename for the markdown file
        # CSV filename - This is the filename for the optional CSV file

    def create_report(self):
        """
        This method is responsible for creating the Summary report and writing it to file.
        """

        csv_content = [["model", "duration", "eval_dataset", {self.metric}]]

        with open(self.report_path, "w") as report_file:

            report_file.write(f"# Experiment summary\n")

            if self.prediction_files is not None:
                if self.csv_path is not None:
                    with open(self.csv_path, "w") as csv_output_file:
                        self._build_summary_table(report_file, csv_output_file)
                else:
                    self._build_summary_table(report_file)
            else:
                logger.info(f"No predictions files were provided to the Summary Report. Nothing to add "
                            f"to the Summary Table!")

            if self.plot_files is not None:
                for plot_file in self.plot_files:
                    pass
            else:
                logger.info(f"No plot files were provided to the Summary Report. No plots to add "
                            f"to the report!")

    def _build_summary_table(self, output_file, csv_output_file=None):
        # Write the header for this table.
        output_file.write(f"Model | Duration (seconds) | Eval Dataset | {self.metric} | Train Chart\n"
                          f"--- | --- | --- | --- | ---\n")

        csv_writer = None
        if csv_output_file is not None:
            csv_writer = csv.writer(csv_output_file)
            csv_writer.writerow(["model", "duration", "eval_dataset", {self.metric}])

        for file in self.prediction_files:
            logger.info(f"Adding the data from {file} to the summary report.")

            prediction_data = EvaluationOutput.load(file)
            eval_data_name = prediction_data.options.dataset.config
            model_name = prediction_data.options.model.name
            model_mgr = ModelManager(model_name)

            train_data = TrainingOutput.load(model_mgr.get_training_out_file())
            duration = train_data.times.duration

            orig_training_plot = model_mgr.get_training_summary_plot()
            dst_training_plot = self.report_files_path / f"{model_name}_output.png"
            copy(orig_training_plot, dst_training_plot)

            metric_value = self._retrieve_metric_data(prediction_data.results.metrics)

            if self.metric == 'Accuracy' or self.metric == 'Balanced Accuracy':
                output_file.write(f"{model_name} | {duration} | {eval_data_name} | {metric_value:.2%} | "
                                  f"[Training Chart](./report_files/{model_name}_output.png)\n")
            elif self.metric == 'mAP':
                output_file.write(f"{model_name} | {duration} | {eval_data_name} | {metric_value} | "
                                  f"[Training Chart](./report_files/{model_name}_output.png)\n")

            if csv_writer is not None:
                csv_writer.writerow([model_name, duration, eval_data_name, metric_value])

    def _retrieve_metric_data(self, metric_data):
        # Fetch the correct evaluation metric value from the metrics.
        if self.metric == 'Accuracy':
            metric_value = metric_data.accuracy
        elif self.metric == 'Balanced Accuracy':
            metric_value = metric_data.balanced_accuracy
        elif self.metric == 'mAP':
            metric_value = metric_data.bbox['mAP']
        else:
            metric_value = "N/A"

        return metric_value

    def _add_summary_plots(self):
        for plot in self.plot_files:
            pass

    def _determine_shared_metric(self):
        """
        This function iterates through the list of specified predictions files and determines if
        the report is summarizing classification or object detection metrics. If a mismatch is
        detected, this function will terminate execution.
        :return: A string ("Balanced Accuracy" | "mAP") indicating which metric the prediction files
        all have in common.
        """

        # Start with an empty list of metrics.
        from collections import defaultdict
        counts = defaultdict(list)

        # Loop through the list of predictions files and load the evaluation data.
        for prediction_file in self.prediction_files:
            eval_data = EvaluationOutput.load(prediction_file)

            # Identify the metrics.  In the case of the classification metrics, both may be there
            # so we need to count them.
            found = False
            if eval_data.results.metrics.accuracy is not None:
                counts['Acc'].append(prediction_file)
                found = True
            if eval_data.results.metrics.balanced_accuracy is not None:
                counts['BalAcc'].append(prediction_file)
                found = True
            if eval_data.results.metrics.bbox is not None:
                counts['mAP'].append(prediction_file)
                found = True
            if not found:
                counts['None'].append(prediction_file)

        # If we have some that we didn't identify, then show the files and bail.
        if len(counts['None']) > 0:
            logger.error(f"Found prediction files without known metrics: {counts['None']}. Exiting.")
            sys.exit(-1)

        # If we have Classification AND OD metrics at the same time, they are not comparable. Bail.
        if (len(counts['Acc']) > 0 or len(counts['BalAcc']) > 0) and len(counts['mAP']) > 0:
            logger.error(f"Found a mixture of classification and object detection metrics in the prediction files. "
                         f"Classification files={counts['Acc']} or {counts['BalAcc']}, "
                         f"object detection files={counts['mAP']}. Exiting.")
            sys.exit(-1)

        # If we have mAP use that, else figure out which classification is larger and use it.
        if len(counts['mAP']):
            metric = "mAP"
        elif len(counts['BalAcc']) > len(counts['Acc']):
            assert len(counts['BalAcc']) == len(self.prediction_files)
            metric = "Balanced Accuracy"
        else:
            assert len(counts['Acc']) == len(self.prediction_files)
            metric = "Accuracy"

        # If each prediction file contains the same type of metric, that metric must be the shared metric.
        logger.info(f"Building a report to summarize {metric}.")

        return metric
