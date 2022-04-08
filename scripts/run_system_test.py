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

"""
This could be done via pytest, but we really don't have a need for that level of detail right now.
* Cleans the directory
* Builds the model
* Runs the test
* Generates all other output
* Compares against the reference output
"""

import argparse
import filecmp
import logging
import os
from pathlib import Path
import subprocess
import sys
import torch

from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.training_output import TrainingOutput
import juneberry.filesystem as jbfs

# model_name, eval_dataset, min_train_accuracy, min_eval_metric (balanced_accuracy | mAP)
# NOTE: The min_train_accuracy and min_eval_metric thresholds are rough guidelines of what
# values to expect during testing. They have been set to accommodate the LOWEST values that
# were observed during CPU training, SINGLE GPU training, TWO GPU training, and FOUR GPU training.
# Additional comments have been provided for certain thresholds marking the values that were
# observed during testing. The following models have been observed to train deterministically during
# the following conditions:
# imagenette - CPU training, Single GPU training
# tabular_binary_sample - CPU training, Single GPU training
# detectron2 - CPU training, Single GPU training
# mmdetection - None
CLSFY_TEST_SET = [
    [
        "imagenette_160x160_rgb_unit_test_pyt_resnet18",
        "data_sets/imagenette_unit_test.json",
        1.0,
        0.49
    ],
    [
        "imagenette_224x224_rgb_unit_test_tf_resnet50",
        "data_sets/imagenette_unit_test.json",
        1.0,
        0.49
    ],
    [
        "tabular_binary_sample",
        "models/tabular_binary_sample/test_data_config.json",
        0.95,
        0.8999
    ],
]

OD_GPU_TEST_SET = [
    [
        "text_detect/dt2/ut",
        "data_sets/text_detect_val.json",
        # Single GPU (0.34), 2 GPU (0.27), 4 GPU (0.31)
        # This number came from 2 A100 gpus
        0.18,
        #  Single GPU (0.004), 2 GPU (0.003), 4 GPU (0.004)
        0.00003
    ],
    [
        "text_detect/mmd/ut",
        "data_sets/text_detect_val.json",
        0.92,
        #  Single GPU (9.3), 2 GPU (2.5), 4 GPU (2.0) in testing.
        0.016
    ]
]

OD_CPU_TEST_SET = [
    [
        "text_detect/dt2/ut",
        "data_sets/text_detect_val.json",
        0.23,
        0.000038
    ]
]

if torch.cuda.is_available():
    OD_TEST_SET = OD_GPU_TEST_SET
    OD_EXPERIMENT = "smokeTests/od/gpu"
else:
    OD_TEST_SET = OD_CPU_TEST_SET
    OD_EXPERIMENT = "smokeTests/od/cpu"


def show_banner(message):
    logging.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"{message}...")


class CommandRunner:
    """
    Simple helper object used to encapsulate the command line arguments and manages calls to subprocess.
    """

    def __init__(self, bin_dir: Path, sub_env, workspace_root: Path, data_root: Path):
        """
        :param bin_dir: The directory in which the command is called from.
        :param workspace_root: Root directory of the current workspace.
        :param data_root: Root directory for the data files.
        """
        self.bin_dir = bin_dir
        self.sub_env = sub_env
        self.workspace_root = workspace_root
        self.data_root = data_root

    def run(self, args: list, *, add_roots: bool = False):
        if add_roots:
            args.append("-w")
            args.append(str(self.workspace_root))

            if self.data_root is not None:
                args.append("-d")
                args.append(str(self.data_root))

        args = [sys.executable] + args
        result = subprocess.run(args, env=self.sub_env)
        if result.returncode != 0:
            logging.error(f"Returned error code {result.returncode}. EXITING!!")
            exit(-1)


def check_for_init(init_if_needed, test_set) -> bool:
    """
    Checks to see if the model directory has been inited. If init_if_needed is False and the directory
    is not inited, the system will exit. If init_if_needed is set to True and the directory is not inited
    this function will return TRUE otherwise will return FALSE.
    :param init_if_needed: Set to true to not automatically exit and return true when there is any un-inited directory.
    :param test_set: The set of things to test.
    :return: True if we should init, False if we don't need to.
    """
    model_names = [x[0] for x in test_set]

    logging.info(f"Checking for known results in the model directories...")

    missing_count = 0
    for model_name in model_names:
        model_mgr = jbfs.ModelManager(model_name)
        known = model_mgr.get_known_results()

        if not known.exists():
            logging.info(f"No {known.name} in {model_name} to compare to! Consider using '--init' "
                         f"to initialize known.")
            missing_count += 1

    if missing_count > 0:
        if init_if_needed:
            logging.info("...missing inited test and '--initifneeded' set: Will init unit tests")
            return True
        else:
            logging.error("System test not inited, exiting. Consider using '--init' to initialize known "
                          "or '--initifneeded'")
            sys.exit(-1)

    # We are good!
    logging.info("...known results exist.")
    return False


def convert_output_to_latest(model_mgr, prediction_file_path: Path):
    """
    Convert the test output to the "latest" file. The conversion will remove the volatile parts such as timestamps,
    model hash, etc.
    :param model_mgr: The model manager for the model.
    :param prediction_file_path: The prediction file to convert.
    :return:
    """
    # Compare output. First, strip the timestamps and other variable stuff
    logging.info(f"Converting the output in {model_mgr.get_model_dir()}...")
    eval_output = EvaluationOutput.load(str(prediction_file_path))
    eval_output.times = None
    eval_output.options.model.hash = None
    eval_output.save(model_mgr.get_latest_results())


def init_test_results(model_mgr: jbfs.ModelManager, error_summary) -> None:
    """
    Renames "latest" results to "known" results.
    :param model_mgr: The model manager for a model.
    :param error_summary: A summary of generated errors.
    """
    latest = model_mgr.get_latest_results()
    known = model_mgr.get_known_results()

    logging.info(f">>> Initializing known results in {model_mgr.get_model_dir()} <<<")
    if known.exists():
        known.unlink()
    if not latest.exists():
        logging.error(f"Latest does NOT exist: {str(latest)}")
        error_summary.append(f"Latest does NOT exist: {str(latest)}")
    latest.rename(known)


def compare_test_results(model_mgr: jbfs.ModelManager, error_summary) -> int:
    """
    Compares the latest results to the known results in the model directory.
    :param model_mgr: The model dir
    :param error_summary: A summary of generated errors
    :return: 0 if no errors, 1 if an error
    """
    latest = model_mgr.get_latest_results()
    known = model_mgr.get_known_results()

    if known.exists():
        if not filecmp.cmp(known, latest):
            logging.error(f">>> Unit test FAILED. 'diff -y {str(known)} {str(latest)} | more' <<<")
            error_summary.append(f">>> Unit test FAILED. 'diff -y {str(known)} {str(latest)} | more' <<<")
            return 1
        else:
            logging.info(f">>> Successful results for for {model_mgr.get_model_dir()} <<<")
    else:
        err_str = (f"Unit test FAILED! No 'known_ut_results.json' to compare to! Consider using '--init' "
                   f"to initialize known.")
        logging.error(err_str)
        error_summary.append(err_str)
        return 1

    return 0


def check_training_metric(model_name, model_mgr, eval_dir_mgr, min_train_metric, min_eval_metric):
    """
    Checks the accuracy of the training run from the output against the minimum accuracy and returns True
    if the accuracy exceeds the minimum accuracy. Also logs results.
    :param model_name: The model name for logging.
    :param model_mgr: The model manager for the model.
    :param eval_dir_mgr: The eval dir manager for this model / test dataset combo.
    :param min_train_metric: The minimum value the model's training metric should achieve.
    :param min_eval_metric: The minimum value the model's evaluation metric should achieve.
    :return: True if metric thresholds are met; abandon the test when a threshold is missed.
    """

    train_outfile = model_mgr.get_training_out_file()
    training_data = TrainingOutput.load(train_outfile)
    if model_mgr.model_task == "classification":
        eval_data = EvaluationOutput.load(eval_dir_mgr.get_predictions_path())
    elif model_mgr.model_task == "objectDetection":
        eval_data = EvaluationOutput.load(eval_dir_mgr.get_metrics_path())
    else:
        logging.error(f"Unknown task type detected for model {model_name}. Unable to determine which training metric "
                      f"to check for this model. Exiting.")
        sys.exit(-1)

    platform = model_mgr.get_model_platform()

    train_metric_name = "accuracy"
    training_metric = training_data.results.accuracy[-1]

    if platform in ['pytorch', 'pytorch_privacy']:
        eval_metric_name = "balanced_accuracy"
        eval_metric = eval_data.results.metrics.balanced_accuracy

    elif platform in ['detectron2', 'mmdetection']:
        eval_metric_name = "mAP"
        eval_metric = eval_data.results.metrics.bbox['mAP']

    elif platform in ['tensorflow']:
        eval_metric_name = "accuracy"
        eval_metric = eval_data.results.metrics.accuracy

    else:
        logging.error(f"Unknown platform type detected for model {model_name}. Exiting.")
        sys.exit(-1)

    if training_metric >= min_train_metric:
        logging.info(f"Training {train_metric_name} for {model_name} of {training_metric} exceeded "
                     f"min {train_metric_name}: {min_train_metric}.")

        # Met the training threshold, so check the eval metric in the same fashion.
        if eval_metric >= min_eval_metric:
            logging.info(f"Evaluation {eval_metric_name} for {model_name} of {eval_metric} exceeded "
                         f"min {eval_metric_name}: {min_eval_metric}.")
            return True
        else:
            logging.error(f"Evaluation {eval_metric_name} for {model_name} of {eval_metric} did NOT exceed min "
                          f"{eval_metric_name}: {min_eval_metric}. See evaluation output file for details.")
            sys.exit(-1)

    else:
        logging.error(f"Training {train_metric_name} for {model_name} of {training_metric} did NOT exceed min "
                      f"{train_metric_name}: {min_train_metric}. See training output file for details.")
        sys.exit(-1)


def count_missing_patterns(dir_path: Path, file_patterns: list) -> int:
    """
    Used to count file patterns MISSING FROM a path.
    :param dir_path: The path to check.
    :param file_patterns: The file patterns to look for.
    :return: Count of missing patterns.
    """
    count = 0
    for pattern in file_patterns:
        if len(list(dir_path.glob(pattern))) == 0:
            logging.error(f"Directory {dir_path} is missing items from pattern {pattern}!")
            count += 1
    return count


def count_found_patterns(dir_path: Path, file_patterns: list) -> int:
    """
    Used to count file patterns IN a path.
    :param dir_path: The path to check.
    :param file_patterns: The file patterns to look for.
    :return: Count of found patterns.
    """
    count = 0
    for pattern in file_patterns:
        if len(list(dir_path.glob(pattern))) == 1:
            logging.error(f"Found items in pattern {pattern}!")
            count += 1

    return count


def get_experiment_file_list(experiment_name):
    """:return: List of expected files for the experiment directory."""

    if experiment_name == "smokeTests/classify":
        return [
            "Simple binary test ROC.png",
            "System Test 1 ROC class 0,217,482,491,497.png",
            "System Test 1 ROC class 566,569,571.574,701.png",
            "System Test Summary.csv",
            "System Test Summary.md",
            "rules.json",
            "main_dodo.py",
            "log_experiment.txt",
            "dryrun_dodo.py"
        ]

    if experiment_name == "smokeTests/od/cpu":
        return [
            "/ut_val_dt2/eval_metrics.csv",
            "/ut_val_dt2/log_plot_pr.txt",
            "/ut_val_dt2/pc_curve.png",
            "/ut_val_dt2/pr_curve.png",
            "/ut_val_dt2/rc_curve.png",
            "OD System Test Summary.csv",
            "System OD Test Summary.md",
            "rules.json",
            "main_dodo.py",
            "log_experiment.txt",
            "dryrun_dodo.py",
        ]

    if experiment_name == "smokeTests/od/gpu":
        return [
            "/ut_val_dt2/eval_metrics.csv",
            "/ut_val_dt2/log_plot_pr.txt",
            "/ut_val_dt2/pc_curve.png",
            "/ut_val_dt2/pr_curve.png",
            "/ut_val_dt2/rc_curve.png",
            "/ut_val_mmd/eval_metrics.csv",
            "/ut_val_mmd/log_plot_pr.txt",
            "/ut_val_mmd/pc_curve.png",
            "/ut_val_mmd/pr_curve.png",
            "/ut_val_mmd/rc_curve.png",
            "/ut_val_combined/eval_metrics.csv",
            "/ut_val_combined/log_plot_pr.txt",
            "/ut_val_combined/pc_curve.png",
            "/ut_val_combined/pr_curve.png",
            "/ut_val_combined/rc_curve.png",
            "OD System Test Summary.csv",
            "System OD Test Summary.md",
            "rules.json",
            "main_dodo.py",
            "log_experiment.txt",
            "dryrun_dodo.py",
        ]


def get_experiment_dry_run_list():
    """:return: Files expected in the experiment directory after the dry run."""
    return ["log_experiment_dryrun.txt"]


def get_predictions_file_patterns() -> list:
    """:return: List of patterns that we generate in the models directory during standard evaluation."""
    files = ["predictions_*.json", "eval_log_*.txt"]
    return files


def get_model_train_file_patterns(model_name: str) -> list:
    """
    Lists the expected files created during training for a particular model.
    :param model_name: The name of the model being checked.
    :return: List of patterns that we generate in the models directory during training.
    """

    model_mgr = jbfs.ModelManager(model_name)

    files = [
        '/'.join(model_mgr.get_training_out_file().parts[-2:]),
        '/'.join(model_mgr.get_training_log().parts[-2:]),
        model_mgr.get_model_path().name
    ]

    if model_name in ["imagenette_224x224_rgb_unit_test_tf_resnet50"]:
        return files

    if model_name in ["imagenette_160x160_rgb_unit_test_pyt_resnet18", "tabular_binary_sample"]:
        files.append('/'.join(model_mgr.get_training_summary_plot().parts[-2:]))
        return files

    elif "text_detect" in model_name:
        ext = [
            '/'.join(model_mgr.get_training_data_manifest_path().parts[-2:]),
            '/'.join(model_mgr.get_validation_data_manifest_path().parts[-2:])
        ]

        plat_conf = '/'.join(model_mgr.get_platform_training_config("py").parts[-2:])
        ext.append(plat_conf)

        files.extend(ext)
        return files

    else:
        logging.error(f"Internal error: {model_name} is unknown in system test. EXITING.")
        sys.exit(-1)


def get_model_dry_run_file_patterns(model_name: str) -> list:
    """
    Lists the expected files created during the dry run for a particular model.
    :param model_name: The name of the model being checked.
    :return: List of patterns that we generate in the models directory dry run.
    """

    # TODO: All the path splitting stuff has to change. It is too cryptic and fragile.
    model_mgr = jbfs.ModelManager(model_name)

    files = [
        '/'.join(model_mgr.get_training_dryrun_log_path().parts[-2:])
    ]

    if model_name == "imagenette_160x160_rgb_unit_test_pyt_resnet18":
        ext = [
            model_mgr.get_pytorch_model_summary_path().name,
            '/'.join(model_mgr.get_dryrun_imgs_dir().parts[-2:]) + "/"
        ]

    elif model_name == "imagenette_224x224_rgb_unit_test_tf_resnet50":
        ext = [
            model_mgr.get_pytorch_model_summary_path().name,
            '/'.join(model_mgr.get_dryrun_imgs_dir().parts[-2:]) + "/",
            '/'.join(model_mgr.get_training_data_manifest_path().parts[-2:]),
            '/'.join(model_mgr.get_validation_data_manifest_path().parts[-2:])
        ]

    elif model_name == "tabular_binary_sample":
        ext = [
            model_mgr.get_pytorch_model_summary_path().name
        ]

    elif "text_detect" in model_name:
        ext = [
            '/'.join(model_mgr.get_training_data_manifest_path().parts[-2:]),
            '/'.join(model_mgr.get_validation_data_manifest_path().parts[-2:])
        ]

        plat_conf = '/'.join(model_mgr.get_platform_training_config("py").parts[-2:])
        ext.append(plat_conf)

    else:
        logging.error(f"Internal error: {model_name} is unknown in system test. EXITING.")
        sys.exit(-1)

    files.extend(ext)

    return files


def clean_experiment(runner, experiment_name: str, workflow_flag=None) -> None:
    """
    Calls the clean mode of run_experiment.
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    :param workflow_flag: Optional flag for the workflow that needs to be cleaned.
    """
    show_banner(f"Running Experiment (CLEAN)...")
    args = [str(runner.bin_dir / "jb_run_experiment"), experiment_name, "--clean", "--commit"]
    if workflow_flag:
        args.append(workflow_flag)
    runner.run(args, add_roots=True)


def clean_pydoit_predictions(runner, experiment_name: str) -> None:
    """
    Cleans the predictions files produced by PyDoit
    :param runner: The runner object.
    :param experiment_name: The experiment to clean.
    """
    show_banner(f"Cleaning predictions files for PyDoit experiment {experiment_name}...")
    runner.run([str(runner.bin_dir / "jb_clean_predictions"), experiment_name])


def check_clean(runner, experiment_name: str, model_names: list, error_summary, check_experiment=True) -> int:
    """
    Checks to make sure that run_experiment cleaned all the appropriate files.
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    :param model_names: The models to check.
    :param error_summary: A summary of generated errors.
    :param check_experiment: Set to True to check experiment log files.
    :return: Number of files found that should have been cleaned.
    """
    # Look for files that SHOULD NOT be there.
    file_count = 0

    if check_experiment:
        experiment_dir = runner.workspace_root / 'experiments' / experiment_name
        file_count += count_found_patterns(experiment_dir, get_experiment_dry_run_list())
        file_count += count_found_patterns(experiment_dir, get_experiment_file_list(experiment_name))

    for model_name in model_names:
        model_mgr = jbfs.ModelManager(model_name)
        model_dir = model_mgr.get_model_dir()

        file_count += count_found_patterns(model_dir, get_model_dry_run_file_patterns(model_name))
        file_count += count_found_patterns(model_dir, get_model_train_file_patterns(model_name))

    if file_count > 0:
        logging.error(f"Failed to clean experiment. Found {file_count} files")
        error_summary.append(f"Failed to clean experiment. Found {file_count} files")

    return file_count


# This wasn't used anywhere.
# def check_clean_predictions(model_names: list, error_summary) -> int:
#     """
#     Checks to make sure that clean_pydoit_predictions cleaned all the appropriate files.
#     :param model_names: The models to check.
#     :param error_summary: A summary of generated errors
#     :return: Number of predictions files found that should have been cleaned
#     """
#     file_count = 0
#     for model_name in model_names:
#
#         # Get model info
#         model_manager = jbfs.ModelManager(model_name)
#         model_dir = model_manager.get_model_dir()
#         model_platform = model_manager.get_model_platform()
#
#         if model_platform == "detectron2":
#             file_count += count_found_patterns(model_dir, get_dt2_predictions_file_patterns())
#         else:
#             file_count += count_found_patterns(model_dir, get_dt2_predictions_file_patterns())
#
#     if file_count > 0:
#         logging.error(f"Failed to clean predictions files. Found {file_count} files")
#         error_summary.append(f"Failed to clean experiment. Found {file_count} files")
#
#     return file_count


def dry_run_experiment(runner, experiment_name: str) -> None:
    """
    Calls the dry run mode of run_experiment
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    """
    show_banner(f"Running Experiment (DRY RUN)...")

    args = [str(runner.bin_dir / "jb_run_experiment"), experiment_name, "--dryrun", "--commit"]
    runner.run(args, add_roots=True)


def check_for_files(runner, stage, experiment_name: str, experiment_patterns: list, model_names: list,
                    model_pattern_fn, error_summary, check_experiment=True) -> int:
    """
    Checks to see if files (patterns) exist in the directory.
    :param runner: The runner object.
    :param stage: The stage of the run for logging.
    :param experiment_name: The experiment we are examining.
    :param experiment_patterns: Experiments patterns.
    :param model_names: The models to check.
    :param model_pattern_fn: Function that returns the patterns of model files to check.
    :param error_summary: A summary of generated errors.
    :param check_experiment: Set to True to check experiment log files.
    :return: Number of missing files.
    """
    missing = 0

    if check_experiment:
        experiment_dir = runner.workspace_root / 'experiments' / experiment_name
        missing += count_missing_patterns(experiment_dir, experiment_patterns)

    for model_name in model_names:
        model_dir = runner.workspace_root / 'models' / model_name
        missing += count_missing_patterns(model_dir, model_pattern_fn(model_name))

    if missing > 0:
        logging.error(f"Failed {stage}. Missing {missing} files")
        error_summary.append(f"Failed {stage}. Missing {missing} files")

    return missing


def commit_experiment(runner, experiment_name) -> None:
    """
    Performs the training part of the experiment.
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    """
    show_banner(f"Running Experiment (COMMIT)...")

    args = [str(runner.bin_dir / "jb_run_experiment"), experiment_name, "--commit"]
    runner.run(args, add_roots=True)


def check_metric(test_set) -> int:
    """
    Checks that the metric for the model exceeds the minimum value for the test.
    :param test_set: The test set describing the models and expected metric values.
    :return: Number of failures.
    """
    # Remove all latest results
    for model_name, test_name, min_train_metric, min_eval_metric in test_set:
        model_mgr = jbfs.ModelManager(model_name)
        latest = model_mgr.get_latest_results()
        if latest.exists():
            latest.unlink()

    # Check the metric in the training runs
    failures = 0
    for model_name, test_name, min_train_metric, min_eval_metric in test_set:
        model_mgr = jbfs.ModelManager(model_name)
        eval_dir_mgr = model_mgr.get_eval_dir_mgr(test_name)
        if model_mgr.model_task == "classification":
            metric_file_path = eval_dir_mgr.get_predictions_path()
        elif model_mgr.model_task == "objectDetection":
            metric_file_path = eval_dir_mgr.get_metrics_path()
        else:
            logging.warning(f"The task type for model '{model_name}' could not be determined. Skipping metrics check.")
            continue

        # Check the metric against what we expect before we move forward
        # If good, make a latest results file that can be checked
        if check_training_metric(model_name, model_mgr, eval_dir_mgr, min_train_metric, min_eval_metric):
            convert_output_to_latest(model_mgr, metric_file_path)
        else:
            failures += 1

    return failures


def compare_latest_if_exists(model_names, error_summary):
    """
    Compares all the latest results with the known results.
    :param model_names: A list of the model names to check.
    :param error_summary: A summary of generated errors.
    :return: Number of failures.
    """
    # Check whatever latest we have.
    failures = 0
    for model_name in model_names:
        model_mgr = jbfs.ModelManager(model_name)
        latest = model_mgr.get_latest_results()
        if latest.exists():
            failures += compare_test_results(model_mgr, error_summary)

    return failures


def do_experiment(runner, init_known, experiment_name, test_set, error_summary):
    """
    Executes the specified experiment from clean, dry run, commit and checking results using the new PyDoit-backed
    jb_run_experiment script.
    :param runner: The runner object.
    :param init_known: Should the "known results" be initialized.
    :param experiment_name: The name of the experiment.
    :param test_set: A test set describing what the experiment should do.
    :param error_summary: A summary of generated errors.
    :return: Number of failures.
    """
    model_names = [x[0] for x in test_set]

    # Before we do anything remove all the "latest" results.
    for model_name in model_names:
        model_mgr = jbfs.ModelManager(model_name)
        latest = model_mgr.get_latest_results()
        if latest.exists():
            latest.unlink()

    failures = 0

    clean_experiment(runner, experiment_name, "--dryrun")
    clean_experiment(runner, experiment_name)
    failures += check_clean(runner, experiment_name, model_names, error_summary, check_experiment=False)

    dry_run_experiment(runner, experiment_name)
    failures += check_for_files(runner, "DRY RUN", experiment_name, get_experiment_dry_run_list(), model_names,
                                get_model_dry_run_file_patterns, error_summary, check_experiment=False)

    commit_experiment(runner, experiment_name)
    failures += check_for_files(runner, "MAIN", experiment_name, get_experiment_file_list(experiment_name), model_names,
                                get_model_train_file_patterns, error_summary, check_experiment=False)

    # Now at this point check the metrics. These produce 'latest' files if ok.
    show_banner("Checking metrics...")
    metric_failures = check_metric(test_set)
    failures += metric_failures

    # Now, if they want it inited, do it if everything is okay.
    if init_known:
        if metric_failures > 0:
            logging.error(f"{metric_failures} tests failed to meet min metric threshold. See log. NOT INITING.")
            error_summary.append(f"{metric_failures} tests failed to meet min metric threshold. See log. NOT INITING.")
        else:
            show_banner("Initing test results...")
            for model_name in model_names:
                model_mgr = jbfs.ModelManager(model_name)
                init_test_results(model_mgr, error_summary)
            logging.info("Tests INITED. This does NOT mean they passed.")
    else:
        # Since we have been inited, compare each set of output that we have
        failures += compare_latest_if_exists(model_names, error_summary)

    return failures


def do_reinit(test_set, error_summary) -> None:
    """
    Check the training accuracies and if good regenerate the latest results and copy to the known results.
    This is useful for when the experiment has been run successfully (manually).
    :param test_set: A test set describing what the experiment should do.
    :param error_summary: A summary of generated errors.
    """
    show_banner("Re-initializing")
    # To reinit we check that each training set satisfies the minimum metric, then convert to latest and copy over.
    models = [x[0] for x in test_set]

    # Before we do anything remove all the "latest" results.
    for model in models:
        model_mgr = jbfs.ModelManager(model)
        latest = model_mgr.get_latest_results()
        if latest.exists():
            latest.unlink()

    # Sanity check that we have output
    missing = 0
    for model in models:
        model_mgr = jbfs.ModelManager(model)
        out_path = model_mgr.get_training_out_file()
        if not out_path.exists():
            logging.error(f"Missing training output file in {model_mgr.get_model_dir()}.")
            error_summary.append(f"Missing training output file in {model_mgr.get_model_dir()}.")
            missing += 1

    if missing > 0:
        fake_model_mgr = jbfs.ModelManager(None)
        logging.error(f"Missing a {fake_model_mgr.get_training_out_file().name}. Terminating re-init. EXITING.")
        sys.exit(-1)

    # Okay, now check metrics
    show_banner("Checking metrics...")
    failures = check_metric(test_set)
    if failures > 0:
        logging.error(f"{failures} tests failed to meet min metric. See log. NOT RE-INITING")
        error_summary.append(f"{failures} tests failed to meet min metric. See log. NOT RE-INITING")
    else:
        show_banner("RE-INITING test results...")
        for model in models:
            model_mgr = jbfs.ModelManager(model)
            init_test_results(model_mgr, error_summary)


# The system test no longer does this.
# def do_generate_experiment(runner, error_summary) -> int:
#     """
#     Really really simple test for generate experiment.
#     :param runner: The test command runner.
#     :param error_summary: A summary of generated errors.
#     :return:
#     """
#     experiment_name = "generatedExperiment"
#     show_banner(f"Running generate experiment {experiment_name}...")
#
#     model_names = [
#         "generatedExperiment/" + x
#         for x in ["lr_0_lrSched_0", "lr_0_lrSched_1", "lr_1_lrSched_0", "lr_1_lrSched_1"]
#     ]
#
#     models_dir = runner.workspace_root / "models"
#
#     # Remove the config files
#     for model_name in model_names:
#         config_path = models_dir / model_name / "config.json"
#         if config_path.exists():
#             config_path.unlink()
#
#     # Run the generator
#     args = [str(runner.bin_dir / "jb_generate_experiments"), "generatedExperiment"]
#     runner.run(args)
#
#     # At this point we should have config files. We don't diff them or anything yet.
#     failures = 0
#     for model_name in model_names:
#         config_path = models_dir / model_name / "config.json"
#         if not config_path.exists():
#             logging.error(f"Expected to find config file at {config_path}")
#             error_summary.append(f"Expected to find config file at {config_path}")
#             failures += 1
#
#     # Do the dry run and see if it works
#     dry_run_experiment(runner, experiment_name)
#     failures += check_for_files(
#         runner,
#         "DRY RUN",
#         experiment_name,
#         get_experiment_dry_run_list(),
#         model_names,
#         get_model_dry_run_file_patterns(),
#         error_summary,
#     )
#
#     return failures


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Script for executing unit tests. Use --init to initialize known good files for this host. "
                    "We assume that there is a 'juneberry' directory as a peer to this workspace and that"
                    "is has a bin directory. We will use those scripts. If you wish to test a different juneberry"
                    "directory use the '-j' switch to select another.")
    parser.add_argument("-d", "--dataRoot", type=str, default=None,
                        help="Root of data directory. Overrides values pulled from config files.")
    parser.add_argument("-j", "--juneberry", type=str, default=None,
                        help="The juneberry directory with the executable to test.")
    parser.add_argument("--init", default=False, action="store_true",
                        help="Set to true to run experiment from scratch and init known files.")
    parser.add_argument("--reinit", default=False, action="store_true",
                        help="Set to true to initialize known good output WITHOUT re-running experiment. "
                             "Incompatible with init.")
    parser.add_argument("--initifneeded", default=False, action="store_true",
                        help="Set to true to automatically init if not inited. Incompatible with init or reinit.")

    args = parser.parse_args()

    # We need to find the juneberry directory so we can make the bin directory
    script_dir = Path(__file__).parent.absolute()
    workspace_root = script_dir.parent
    juneberry_dir = workspace_root.parent / "juneberry"
    if args.juneberry is not None:
        juneberry_dir = Path(args.juneberry).absolute()
    bin_dir = (juneberry_dir / "bin")
    data_root = None
    if args.dataRoot is not None:
        data_root = Path(args.dataRoot).absolute()

    os.chdir(workspace_root)

    # Set up our environment variables so that we are deterministic
    # https://github.com/NVIDIA/tensorflow-determinism
    sub_env = os.environ.copy()
    sub_env["TF_DETERMINISTIC_OPS"] = "1"
    sub_env["PYTHONHASHSEED"] = "0"

    # Set up the thing to run the command
    runner = CommandRunner(bin_dir, sub_env, workspace_root, data_root)
    error_summary = []

    if args.init and args.reinit:
        logging.error("Init and reinit may not be set at the same time. Exiting.")
        sys.exit(-1)

    if args.init and args.initifneeded:
        logging.error("Init and initifneeded may not be set at the same time. Exiting.")
        sys.exit(-1)

    if args.reinit and args.initifneeded:
        logging.error("Reinit and initifneeded may not be set at the same time. Exiting.")
        sys.exit(-1)

    if args.reinit:
        do_reinit(CLSFY_TEST_SET, error_summary)
        do_reinit(OD_TEST_SET, error_summary)
        return

    if not args.init:
        args.init = check_for_init(args.initifneeded, CLSFY_TEST_SET + OD_TEST_SET)

    # Test jb_run_experiment (PyDoit)
    failures = do_experiment(runner, args.init, "smokeTests/classify", CLSFY_TEST_SET, error_summary)
    failures += do_experiment(runner, args.init, OD_EXPERIMENT, OD_TEST_SET, error_summary)

    if args.init:
        if failures == 0:
            logging.info(f"###### Known results initialized. NOT A VALID TEST.")
        else:
            logging.info(f"###### Some errors during initialization! See log for details!")
            logging.info("Error summary: ")
            for item in error_summary:
                logging.error(item)
    else:
        if failures == 0:
            logging.info(f"###### SUCCESS - No test failures! :)")
        else:
            logging.info(f"###### FAILURE - Number of failures: {failures}")
            logging.info("Error summary: ")
            for item in error_summary:
                logging.error(item)

    sys.exit(failures)


if __name__ == "__main__":
    main()
