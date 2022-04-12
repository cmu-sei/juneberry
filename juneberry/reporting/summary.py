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
from prodict import List, Prodict
from shutil import copy
import sys

from juneberry.config.eval_output import EvaluationOutput, Metrics
from juneberry.config.training_output import TrainingOutput
from juneberry.filesystem import ModelManager
from juneberry.reporting.report import Report
from juneberry.reporting.utils import determine_report_path

logger = logging.getLogger(__name__)


class Summary(Report):
    """
    The purpose of this Report subclass is to reproduce the functionality that was previously
    contained inside the 'jb_summary_report' script. There are three possible components in this
    type of Summary Report:
      1) An 'Experiment Summary' table, which summarizes models, how long they took to train, how they
         performed on an evaluation, and a link to the model's training chart.
      2) An 'Experiment Plots' section, which displays any plots that should be included in
         the Summary Report.
      3) (Optional) A CSV file summarizing the content in the 'Experiment Summary' table.
    """
    def __init__(self, md_str: str = "", csv_str: str = "", metrics_files: List = None,
                 plot_files: List = None):
        super().__init__(output_str=md_str)

        # Determine where to save the output for this report.
        default_filename = "summary.md"
        self.report_path = determine_report_path(self.output_dir, md_str, default_filename)
        logger.info(f"Saving the report to {self.report_path}")

        # CSV mode is optional. If a CSV path was not provided, skip the CSV steps.
        self.csv_path = None if csv_str == "" else Path(csv_str)

        # Store the two types of files to be included in the summary.
        self.metrics_files = metrics_files
        self.plot_files = plot_files

        # This Prodict will contain the relevant Summary data for each metrics file. It will be
        # populated during the determination of the shared metric (below). Building this dict
        # eliminates the need to load the metrics file AGAIN during construction of the Summary table.
        self.table_data_dict = Prodict()

        # Determine which metric the desired metrics files have in common.
        self.metric = self._determine_shared_metric() if self.metrics_files is not None else None

        # Establish the 'report_files' directory, which is used to store copies of any images
        # that are found in the summary report.
        markdown_parent = self.report_path.parent
        self.report_files_path = markdown_parent / "report_files"

        # Create the `report_files` directory (and any parent directories) if it does not exist.
        if not self.report_files_path.exists():
            self.report_files_path.mkdir(parents=True)

    def create_report(self):
        """
        This method is responsible for creating the Summary report and writing it to file.
        """
        # Open up the desired report (markdown) file for writing.
        with open(self.report_path, "w") as report_file:
            # Write the header for the summary report.
            report_file.write(f"# Experiment summary\n")

            # Add the data from any requested metrics files to the summary table.
            if self.metrics_files is not None:
                logger.info(f"Adding {len(self.metrics_files)} rows to the summary table.")

                # If a CSV file was specified, the summary data must also be added there.
                if self.csv_path is not None:
                    logger.info(f"A CSV file was requested. Saving the CSV to {self.csv_path}")

                    # Open the desired CSV file for writing and add the summary data to both files.
                    with open(self.csv_path, "w") as csv_output_file:
                        self._build_summary_table(report_file, csv_output_file)

                # If there's no CSV file, the summary data is only added to the markdown file.
                else:
                    self._build_summary_table(report_file)
            else:
                logger.info(f"No metrics files were provided to the Summary Report. Nothing to add "
                            f"to the Summary Table!")

            # Add the desired plot files (if there are any) to the report (markdown) file.
            if self.plot_files is not None:
                self._add_summary_plots(report_file)
            else:
                logger.info(f"No plot files were provided to the Summary Report. No plots to add "
                            f"to the markdown file!")

    def _build_summary_table(self, output_file: '__file__', csv_output_file: '__file__' = None):
        """
        This method is responsible for filling out the Summary table in the summary report. It
        can also (optionally) write the same data to CSV format when requested.
        :param output_file: The markdown file where this table will be written.
        :param csv_output_file: The CSV file where this table will be written.
        :return: Nothing.
        """
        # Write the header for this table in the markdown file.
        output_file.write(f"Model | Duration (seconds) | Eval Dataset | {self.metric} | Train Chart\n"
                          f"--- | --- | --- | --- | ---\n")

        # If a CSV file was also requested, establish the CSV writer and add a header to the CSV.
        csv_writer = None
        if csv_output_file is not None:
            csv_writer = csv.writer(csv_output_file)
            csv_writer.writerow(["model", "duration", "eval_dataset", {self.metric}])

        # Loop through the desired metrics files, summarize the info and add it to the report.
        for file in self.metrics_files:
            logger.info(f"Adding the metrics data from {file} to the summary report.")

            # Fetch the appropriate table data from the table_data_dict, retrieve the model name and eval dataset
            # that was used in the evaluation, and build a model manager.
            table_data = self.table_data_dict[file]
            eval_data_name = table_data.eval_data_name
            model_name = table_data.model_name
            model_name_str = str(model_name).replace("/", "_") if "/" in model_name else model_name

            model_mgr = ModelManager(model_name)

            # Load the training output and retrieve the amount of time spent training the model.
            train_data = TrainingOutput.load(model_mgr.get_training_out_file())
            duration = train_data.times.duration

            # Retrieve the model's training plot and place a copy of it in the `report_files` directory.
            orig_training_plot = model_mgr.get_training_summary_plot()
            dst_training_plot = self.report_files_path / f"{model_name_str}_output.png"
            copy(orig_training_plot, dst_training_plot)

            # Retrieve the metric value from the metrics data.
            metric_value = self._retrieve_metric_data(table_data.metrics_stanza)

            # Assemble the table row and add it to the summary table.

            # Accuracy metrics are formatted one way.
            if self.metric == 'Accuracy' or self.metric == 'Balanced Accuracy':
                output_file.write(f"{model_name} | {duration} | {eval_data_name} | {metric_value:.2%} | "
                                  f"[Training Chart](./report_files/{model_name_str}_output.png)\n")

            # And mAP is formatted another way.
            elif self.metric == 'mAP':
                output_file.write(f"{model_name} | {duration} | {eval_data_name} | {metric_value} | "
                                  f"[Training Chart](./report_files/{model_name_str}_output.png)\n")

            # If a CSV was requested, format the row and write it to the CSV file.
            if csv_writer is not None:
                csv_writer.writerow([model_name, duration, eval_data_name, metric_value])

    def _retrieve_metric_data(self, metric_data: Metrics):
        """
        The purpose of this method is to obtain the value of a particular metric from the
        metrics portion of the 'results' stanza in a Juneberry metrics file.
        :param metric_data:
        :return:
        """
        # Fetch the desired evaluation metric value from the Metrics.
        if self.metric == 'Accuracy':
            return metric_data.accuracy
        elif self.metric == 'Balanced Accuracy':
            return metric_data.balanced_accuracy
        elif self.metric == 'mAP':
            return metric_data.bbox['mAP']

        # Should at least return something if the desired metric isn't recognized.
        else:
            return "N/A"

    def _add_summary_plots(self, output_file: '__file__'):
        """
        This method is responsible for adding the desired plot files to the Summary Report. Since
        plot files only appear in the markdown output file, there is no need to write anything to
        the CSV file.
        :param output_file: The markdown file for this Summary Report.
        :return: Nothing.
        """
        # Write a section header for all the plot files.
        output_file.write(f"# Experiment Plots\n")

        # Loop through all of the desired plot files.
        for plot in self.plot_files:
            # Convert the plot str to a Path, determine the correct Path for the copy of the
            # plot that will be placed in the 'report_files' directory and perform the copy.
            plot_path = Path(plot)
            dst_plot_path = self.report_files_path / plot_path.name
            copy(plot_path, dst_plot_path)

            # Write two lines to the markdown file for the current plot. Make sure to
            # use the Path of the plot file that's inside the 'report_files' directory.
            output_file.write(f"![Plot Image](./report_files/{plot_path.name})\n")
            output_file.write(f"---\n")

    def _determine_shared_metric(self):
        """
        This function iterates through the list of specified metrics files and determines if
        the report is summarizing classification or object detection metrics. If a mismatch is
        detected, this function will terminate execution.
        :return: A string ("Balanced Accuracy" | "mAP") indicating which metric the metrics files
        all have in common.
        """

        # Start with an empty list of metrics.
        from collections import defaultdict
        counts = defaultdict(list)

        # Loop through the list of metrics files and load the evaluation data.
        for metrics_file in self.metrics_files:
            eval_data = EvaluationOutput.load(metrics_file)

            # Identify the metrics.  In the case of the classification metrics, both may be there
            # so we need to count them.
            found = False
            if eval_data.results.metrics.accuracy is not None:
                counts['Acc'].append(metrics_file)
                found = True
            if eval_data.results.metrics.balanced_accuracy is not None:
                counts['BalAcc'].append(metrics_file)
                found = True
            if eval_data.results.metrics.bbox is not None:
                counts['mAP'].append(metrics_file)
                found = True
            if not found:
                counts['None'].append(metrics_file)

            # Add the data needed to build the Summary table to the table_data_dict.
            self.table_data_dict[metrics_file] = Prodict(eval_data_name=eval_data.options.dataset.config,
                                                         model_name=eval_data.options.model.name,
                                                         metrics_stanza=eval_data.results.metrics)

        # If we have some that we didn't identify, then show the files and bail.
        if len(counts['None']) > 0:
            logger.error(f"Found metrics files without known metrics: {counts['None']}. Exiting.")
            sys.exit(-1)

        # If we have Classification AND OD metrics at the same time, they are not comparable. Bail.
        if (len(counts['Acc']) > 0 or len(counts['BalAcc']) > 0) and len(counts['mAP']) > 0:
            logger.error(f"Found a mixture of classification and object detection metrics in the metrics files. "
                         f"Classification files={counts['Acc']} or {counts['BalAcc']}, "
                         f"object detection files={counts['mAP']}. Exiting.")
            sys.exit(-1)

        # If we have mAP use that, else figure out which classification is larger and use it.
        if len(counts['mAP']):
            metric = "mAP"
        elif len(counts['BalAcc']) > len(counts['Acc']):
            assert len(counts['BalAcc']) == len(self.metrics_files)
            metric = "Balanced Accuracy"
        else:
            assert len(counts['Acc']) == len(self.metrics_files)
            metric = "Accuracy"

        # If each metrics file contains the same type of metric, that metric must be the shared metric.
        logger.info(f"Building a report to summarize {metric}.")

        return metric
