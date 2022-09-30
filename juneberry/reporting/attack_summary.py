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
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import List

from juneberry.config.attack import PropertyInferenceAttackConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.training_output import TrainingOutput
import juneberry.filesystem as jb_fs
from juneberry.reporting.report import Report
from juneberry.reporting.utils import determine_report_path

logger = logging.getLogger(__name__)


class AttackSummary(Report):

    def __init__(self, experiment_name: str = "", output_str: str = ""):
        super().__init__(output_str=output_str)

        # Determine where to save the output for this report.
        default_filename = "attack_summary.md"
        self.report_path = determine_report_path(self.output_dir, output_str, default_filename)
        logger.info(f"Saving the report to {self.report_path}")

        # Handle the case where an experiment name is not provided.
        if experiment_name == "":
            logger.error(f"Failed to build report. The experiment_name was not provided.")
            sys.exit(-1)

        # Establish the AttackManager for the Attack directory.
        self.attack_mgr = jb_fs.AttackManager(experiment_name)

    def create_report(self):
        """
        This method is responsible for creating the Attack Summary report and writing it to file.
        """
        # Open up the desired output file for writing, and add each of the tables.
        with open(self.report_path, "w") as output_file:
            self._build_table_1(output_file)
            self._build_table_2(output_file)
            self._build_table_3(output_file)

    def _build_table_1(self, output_file: '__file__') -> None:
        """
        This method is responsible for obtaining the data in the "Summary Statistics for Private/Meta Models"
        table and writing the table to the Attack Summary markdown file.
        :param output_file: The file where this table will be written.
        :return: Nothing.
        """
        # Create the header for the table and write it to the markdown file.
        table_header = f"## Table 1 - Summary Statistics for Private and Meta Models\n" \
                       f"Model | Training Accuracy | Validation Accuracy | Test Accuracy | Train Chart\n" \
                       f"--- | --- | --- | --- | ---\n"
        output_file.write(table_header)

        # Determine the model names for the two private models, then the two meta models.
        model_names = [self.attack_mgr.get_private_model_name(), self.attack_mgr.get_private_model_name(disjoint=True),
                       self.attack_mgr.get_meta_model_name(), self.attack_mgr.get_meta_model_name(disjoint=True)]

        # Add a row to the table for each model
        for model_name in model_names:
            self._write_table_1_row(model_name, output_file)

    @staticmethod
    def _write_table_1_row(model_name: Path, output_file: '__file__') -> List:
        """
        This method is responsible for writing a model's Summary Statistics row to a table. It's
        primarily used for Table 1, however the rows in Table 3 are nearly similar so this method
        is used again during the construction of Table 3.
        :param model_name: The Path to a model in the 'models' directory.
        :param output_file: The file where this row will be written.
        :return: A list of all the accuracy values used in the summary statistics row.
        """
        # Construct a model manager for the model and locate the model's training image.
        model_mgr = jb_fs.ModelManager(model_name)
        train_img_str = model_mgr.get_training_summary_plot()

        # Load the model's training output file and retrieve the accuracy values inside.
        training_output = TrainingOutput.load(model_mgr.get_training_out_file())
        # TODO: Right now this retrieves the last item in the list, but we may want the "best" epoch.
        train_acc = training_output.results.accuracy[-1]
        val_acc = training_output.results.val_accuracy[-1]

        # Construct an eval directory manager for the model, focusing on the evaluation of the
        # query dataset. Load the eval output file and retrieve the accuracy value inside.
        eval_dir_mgr = model_mgr.get_eval_dir_mgr(dataset_path="query_dataset_config")
        eval_output = EvaluationOutput.load(eval_dir_mgr.get_metrics_path())
        test_acc = eval_output.results.metrics.classification["accuracy"]

        # Write the Summary Statistics row to the output file.
        output_file.write(f"{model_name} | {train_acc} | {val_acc} | {test_acc} | [Training Chart]({train_img_str})\n")

        # Return the accuracy values.
        return [train_acc, val_acc, test_acc]

    def _build_table_2(self, output_file: '__file__') -> None:
        """
        This method is responsible for obtaining the data in the "Accuracy for Meta Models"
        table and writing the table to the Attack Summary markdown file.
        :param output_file: The file where this table will be written.
        :return: Nothing.
        """
        # Create the header for the table and write it to the markdown file.
        table_header = "\n\n\n## Table 2 - Accuracy Table for Meta Models\n" \
                       "-- | Superset | Disjoint\n" \
                       "--- | --- | ---\n"
        output_file.write(table_header)

        # Determine the model names for the two meta models.
        model_names = [self.attack_mgr.get_meta_model_name(), self.attack_mgr.get_meta_model_name(disjoint=True)]

        # Write a row to the table for each meta model.
        for model_name in model_names:
            self._write_table_2_row(model_name, output_file)

    @staticmethod
    def _write_table_2_row(model_name: Path, output_file: '__file__') -> None:
        """
        This method is responsible for writing rows to the "Accuracy for Meta Models" table.
        :param model_name: The Path to a model in the 'models' directory.
        :param output_file: The file where this row will be written.
        :return: Nothing.
        """
        # Write the model name to the row.
        output_file.write(f"{model_name}")

        # Create a model manager for the model.
        model_mgr = jb_fs.ModelManager(model_name)

        # Retrieve eval data from both the superset and disjoint evals.
        datasets = ["superset", "disjoint"]
        for dataset in datasets:
            # Construct the dataset name and use it to build the correct eval directory manager.
            dataset_name = f"in_out_{dataset}_private_test_dataset_config"
            eval_dir_mgr = model_mgr.get_eval_dir_mgr(dataset_path=dataset_name)

            # Load the eval output file, retrieve the accuracy value, and write the data to the row.
            eval_output = EvaluationOutput.load(eval_dir_mgr.get_metrics_path())
            acc = eval_output.results.metrics.classification["accuracy"]
            output_file.write(f" | {acc}")

        # The row must end with a new line character.
        output_file.write("\n")

    def _build_table_3(self, output_file: '__file__') -> None:
        """
        This method is responsible for obtaining the data in the "Summary Statistics for Shadow Models"
        table and writing the table to the Attack Summary markdown file.
        :param output_file: The file where this table will be written.
        :return: Nothing.
        """
        # Initialize lists for storing accuracy values. These will be used to calculate mean and stdev.
        train_acc = []
        val_acc = []
        test_acc = []
        acc_lists = [train_acc, val_acc, test_acc]

        # Create the header for the table and write it to the markdown file.
        table_header = "## Table 3 - Summary Statistics for Shadow Models\n" \
                       "-- | Model | Training Accuracy | Validation Accuracy | Test Accuracy | Train Chart\n" \
                       "--- | --- | --- | --- | --- | ---\n"
        output_file.write(table_header)

        # Load the attack file, in order to retrieve the number of shadow models.
        attack_file = self.attack_mgr.get_experiment_attack_file()
        attack_config = PropertyInferenceAttackConfig.load(attack_file)

        # Write a row to the table for each shadow superset model.
        for i in range(attack_config.models.shadow_superset_quantity):
            # Establish the correct model name.
            model_name = self.attack_mgr.get_shadow_model_name(i, disjoint=False)

            # Write the row for that model. The accuracy values are returned.
            acc_values = self._write_table_3_row(model_name, output_file)

            # Place the returned accuracy values in the appropriate accuracy list.
            for value, val_list in zip(acc_values, acc_lists):
                val_list.append(value)

        # Write a row to the table for each shadow disjoint model.
        for i in range(attack_config.models.shadow_disjoint_quantity):
            # Establish the correct model name.
            model_name = self.attack_mgr.get_shadow_model_name(i, disjoint=True)

            # Write the row for that model. The accuracy values are returned.
            acc_values = self._write_table_3_row(model_name, output_file)

            # Place the returned accuracy values in the appropriate accuracy list.
            for value, val_list in zip(acc_values, acc_lists):
                val_list.append(value)

        # Calculate the mean for each accuracy list, format the row, then write the row.
        mean_row = f"mean | - | {mean(train_acc)} | {mean(val_acc)} | {mean(test_acc)} | -\n"
        output_file.write(mean_row)

        # Calculate the stdev for each accuracy list, format the row, then write the row.
        stdev_row = f"stdev | - | {stdev(train_acc)} | {stdev(val_acc)} | {stdev(test_acc)} | -\n"
        output_file.write(stdev_row)

    def _write_table_3_row(self, model_name: Path, output_file: '__file__') -> List:
        """
        This method is responsible for writing a model's Summary Statistics row to Table 3. It
        takes advantage of the code written for Table 1, which produces a nearly identical row.
        :param model_name: The Path to a model in the 'models' directory.
        :param output_file: The file where this row will be written.
        :return: A list of all the accuracy values used in the summary statistics row.
        """
        # Model rows in Table 3 start with an extra blank column. Write that to the file.
        output_file.write(f" - | ")

        # Now write the remainder of the Summary Statistics row using the Table 1 format and
        # return the accuracy values.
        return self._write_table_1_row(model_name, output_file)
