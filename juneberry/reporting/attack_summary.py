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
from statistics import mean, stdev

from juneberry.config.attack import PropertyInferenceAttackConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.training_output import TrainingOutput
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class AttackSummary:

    def __init__(self, experiment_name: str = "", output_path: str = ""):
        self.table_1 = []
        self.table_2 = []
        self.table_3 = []
        self.output_path = output_path
        self.attack_mgr = jbfs.AttackManager(experiment_name)

    def build_table_1(self):
        table_rows = []
        table_header = f"## Table 1 - Summary Statistics for Private and Meta Models\n" \
                       f"Model | Training Accuracy | Validation Accuracy | Test Accuracy | Train Chart\n" \
                       f"--- | --- | --- | --- | ---\n"
        table_rows.append(table_header)

        model_names = [self.attack_mgr.get_private_model_name(), self.attack_mgr.get_private_model_name(disjoint=True),
                       self.attack_mgr.get_meta_model_name(), self.attack_mgr.get_meta_model_name(disjoint=True)]

        for model_name in model_names:
            row, train_acc, val_acc, test_acc = self.add_table_1_row(model_name)
            table_rows.append(row)

        self.table_1 = table_rows

    def build_table_2(self):
        table_rows = []
        table_header = "\n\n\n## Table 2 - Accuracy Table for Meta Models\n" \
                       "-- | Superset | Disjoint\n" \
                       "--- | --- | ---\n"
        table_rows.append(table_header)

        model_names = [self.attack_mgr.get_meta_model_name(), self.attack_mgr.get_meta_model_name(disjoint=True)]

        for model_name in model_names:
            table_rows.append(self.add_table_2_row(model_name))

        self.table_2 = table_rows

    def build_table_3(self):
        table_rows = []
        train_acc = []
        val_acc = []
        test_acc = []

        table_header = "## Table 3 - Summary Statistics for Shadow Models\n" \
                       "-- | Model | Training Accuracy | Validation Accuracy | Test Accuracy | Train Chart\n" \
                       "--- | --- | --- | --- | --- | ---\n"
        table_rows.append(table_header)

        attack_file = self.attack_mgr.get_experiment_attack_file()
        attack_config = PropertyInferenceAttackConfig.load(attack_file)

        num_superset = attack_config.models.shadow_superset_quantity
        num_disjoint = attack_config.models.shadow_disjoint_quantity

        lists = [table_rows, train_acc, val_acc, test_acc]

        for i in range(num_superset):
            model_name = self.attack_mgr.get_shadow_model_name(i, disjoint=False)
            row, train, val, test = self.add_table_3_row(model_name)
            table_rows.append(row)
            train_acc.append(train)
            val_acc.append(val)
            test_acc.append(test)

        for i in range(num_disjoint):
            model_name = self.attack_mgr.get_shadow_model_name(i, disjoint=True)
            row, train, val, test = self.add_table_3_row(model_name)
            table_rows.append(row)
            train_acc.append(train)
            val_acc.append(val)
            test_acc.append(test)

        mean_row = f"mean | - | {mean(train_acc)} | {mean(val_acc)} | {mean(test_acc)} | -\n"
        table_rows.append(mean_row)

        stdev_row = f"stdev | - | {stdev(train_acc)} | {stdev(val_acc)} | {stdev(test_acc)} | -\n"
        table_rows.append(stdev_row)

        self.table_3 = table_rows

    @staticmethod
    def add_table_1_row(model_name):
        model_mgr = jbfs.ModelManager(model_name)
        train_img_str = model_mgr.get_training_summary_plot()

        training_output = TrainingOutput.load(model_mgr.get_training_out_file())
        train_acc = training_output.results.accuracy[-1]
        val_acc = training_output.results.val_accuracy[-1]

        eval_dir_mgr = jbfs.EvalDirMgr(model_mgr.model_dir_path, platform="", dataset_name="query_dataset_config")
        eval_output = EvaluationOutput.load(eval_dir_mgr.get_metrics_path())
        test_acc = eval_output.results.metrics.accuracy

        row = f"{model_name} | {train_acc} | {val_acc} | {test_acc} | [Training Chart]({train_img_str})\n"

        return row, train_acc, val_acc, test_acc

    @staticmethod
    def add_table_2_row(model_name):
        row = f"{model_name}"

        model_mgr = jbfs.ModelManager(model_name)

        datasets = ["superset", "disjoint"]

        for dataset in datasets:
            dataset_name = f"in_out_{dataset}_private_test_dataset_config"
            eval_dir_mgr = jbfs.EvalDirMgr(model_mgr.model_dir_path, platform="", dataset_name=dataset_name)
            eval_output = EvaluationOutput.load(eval_dir_mgr.get_metrics_path())
            acc = eval_output.results.metrics.accuracy
            row += f" | {acc}"

        row += "\n"

        return row

    def add_table_3_row(self, model_name):
        row = f" - | "

        row_data, train_acc, val_acc, test_acc = self.add_table_1_row(model_name)
        row += row_data

        return row, train_acc, val_acc, test_acc

    def create_report(self):
        self.build_table_1()
        self.build_table_2()
        self.build_table_3()

        self.write_file()

    def write_file(self):
        content = self.table_1 + self.table_2 + self.table_3
        with open(self.output_path, "w") as output_file:
            for line in content:
                output_file.write(line)
