#! /usr/bin/env python

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

import copy
import argparse
import logging
from pathlib import Path
import sys

from juneberry.config.dataset import DatasetConfig
from juneberry.config.experiment import ExperimentConfig, Model, ModelTest, Report, ReportTest
import juneberry.filesystem as jbfs
from juneberry.filesystem import ExperimentManager, ModelManager

import juneberry.scripting as jbscripting

# DESIGN
# We asssume the following layout
# experiments/
#   <experiment-name>/
#     patches/
#     base_data_set.json
#
# We make
# experiments/
#   <experiment-name>/
#     data_sets/
# models/
#   <model-name>
#     evals
#       <patch_name>_<size>/
#         metrics.json
#
# DESIGN

logger = logging.getLogger("juneberry.jb_generate_patch_eval")


def make_variation_name(patch_name, size) -> str:
    str_size = f"{int(size)}d{int((size % 1) * 10000)}"
    return f"{patch_name}_{str_size}"


def make_dataset_path(experiment_name, patch_name, size) -> Path:
    exp_mgr = ExperimentManager(experiment_name)
    datasets_dir = exp_mgr.experiment_dir_path / "data_sets"

    return datasets_dir / f"{make_variation_name(patch_name, size)}.json"


def make_patch_dataset(base_config: DatasetConfig, dest_path: Path, patch_path: str, size):
    # Copy the config, adjust the patch_path, size and save
    new_config: DatasetConfig = copy.deepcopy(base_config)
    for transform in new_config.data_transforms.transforms:
        # TODO: Add warning if we don't find one
        if transform.fqcn == "juneberry.transforms.image.Watermark":
            transform.kwargs['watermark_path'] = patch_path
            transform.kwargs['min_scale'] = size
            transform.kwargs['max_scale'] = size
    logger.info(f"Saving dataset config to {dest_path}")
    new_config.save(dest_path)


def generate_experiment(experiment_name, model_name, sizes, target_class) -> list:
    # This function walks the patches directory abd create a dataset file for each patch and size pair.
    # So, when done, there should be patches X sizes files in the datasets directory.

    # Set up some paths
    exp_mgr = ExperimentManager(experiment_name)
    patches_dir = exp_mgr.experiment_dir_path / "patches"
    datasets_dir = exp_mgr.experiment_dir_path / "data_sets"
    datasets_dir.mkdir(exist_ok=True)

    # Load the base dataset config
    base_config = DatasetConfig.load(exp_mgr.experiment_dir_path / "base_data_set.json")

    # Make a new dataset config with the updated size for each patch/size combo
    test_list = []
    report_test_list = []
    curve_names = []
    series_list = []
    for patch_name in patches_dir.iterdir():
        if patch_name.name.startswith("."):
            continue

        curve_names.append(patch_name.stem)
        eval_series = []
        for size in sizes:
            # Make a patch dataset
            dataset_path = make_dataset_path(experiment_name, patch_name.stem, size)
            patch_path = patches_dir / patch_name.name
            make_patch_dataset(base_config, dataset_path, str(patch_path), size)

            # Add a test
            tag = make_variation_name(patch_name.stem, size)
            eval_series.append(tag)
            test_list.append(ModelTest(dataset_path=dataset_path, tag=tag, classify=0))
            report_test_list.append(ReportTest(tag=tag))
        series_list.append(eval_series)

    # Now, assempble an experiment config and save it
    exp_config = ExperimentConfig()
    exp_config.description = "Patch evaluation experiment"
    exp_config.format_version = "0.2.0"
    exp_config.models = [Model(name=model_name, tests=test_list)]

    # Make the reports stanza
    # TODO: The tool plots a bunch of eval metrics but just for that target
    report = Report(description="",
                    fqcn="juneberry.reporting.variation.VariationCurve",
                    kwargs={
                        'model_name': model_name,
                        'curve_names': curve_names,
                        'eval_names': series_list,
                        'target_class': target_class,
                        'x_title': "X Title",
                        'x_values': sizes,
                        'y_title': "Y Title",
                        'output_dir': str(exp_mgr.experiment_dir_path)
                    },
                    tests=report_test_list)
    exp_config.reports = [report]

    # model_name, eval_names, target_class, x_title, x_labels, y_title, output_dir

    # Save the config out.
    logger.info(f"Saving experiment config to: {exp_mgr.get_experiment_config()}")
    exp_config.save(exp_mgr.get_experiment_config())


def main():
    parser = argparse.ArgumentParser(description="Generates all the necessary data sets, models and experiment config "
                                                 "for evaluating the patches.")
    parser.add_argument('experimentName', help='Name of the experiment in the experiments directory that contains '
                                               'the patches directory, base_data_set.json and patch_eval_config.json.')
    parser.add_argument('modelName', help='Name of the base model to use for testing. The evaluations will be placed'
                                          'into this directory.')
    parser.add_argument('targetClass', type=int, help="The target class to evaluate.")
    args = parser.parse_args()

    # TODO: We don't need all the workspace/profile bits (do we?) because we just work within the experiment directory
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    sizes = [round(0.1 * x, 1) for x in range(1, 11)]
    generate_experiment(args.experimentName,
                        args.modelName,
                        sizes,
                        args.targetClass)


if __name__ == "__main__":
    jbscripting.run_main(main, logger)
