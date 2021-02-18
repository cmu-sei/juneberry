#! /usr/bin/env python3

"""
This could be done via pytest, but we really don't have a need for that level of detail right now.
* Cleans the directory
* Builds the model
* Runs the test
* Generates all other output
* Compares against the reference output
"""

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import argparse
import filecmp
import json
import logging
from pathlib import Path
import os
import subprocess
import sys

latest_file_name = 'latest_ut_results.json'
known_file_name = 'known_ut_results.json'

# model_names, test_names, min_accuracies
test_set = [
    ['imagenet_224x224_rgb_unit_test_pyt_resnet50', 'data_sets/imagenet_unit_test.json', 1.0],
    ['tabular_binary_sample', 'models/tabular_binary_sample/test_data_config.json', 0.95]
]


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

    def run(self, args: list, add_roots: bool = False, add_sub_env: bool = False):
        if add_roots:
            args.append('-w')
            args.append(str(self.workspace_root))

            if self.data_root is not None:
                args.append('-d')
                args.append(str(self.data_root))

        args = [sys.executable] + args
        if add_sub_env:
            result = subprocess.run(args, env=self.sub_env)
        else:
            result = subprocess.run(args)
        if result.returncode != 0:
            logging.error(f"Returned error code {result.returncode}. EXITING!!")
            exit(-1)


def check_for_init(init_if_needed) -> bool:
    """
    Checks to see if the model directory has been inited. If init_if_needed is False and the directory
    is not inited, the system will exit. If init_if_needed is set to True and the directory is not inited
    this function will return TRUE otherwise will return FALSE.
    :param init_if_needed: Set to true to not automatically exit and return true when there is any un-inited directory.
    :return: True if we should init, False if we don't need to.
    """
    models_root_dir = Path('models')
    model_names = [x[0] for x in test_set]

    missing_count = 0

    logging.info(f"Checking for known results in {models_root_dir}...")

    for model_name in model_names:
        model_dir = models_root_dir / model_name

        if not (model_dir / 'known_ut_results.json').exists():
            logging.info(f"No 'known_ut_results.json' in {model_name} to compare to! "
                         f"Consider using \'--init\' to initialize known.")
            missing_count += 1

    if missing_count > 0:
        if init_if_needed:
            logging.info("...missing inited test and '--initifneeded' set: Will init unit tests")
            return True
        else:
            logging.error("System test not inited, exiting. "
                          "Consider using \'--init\' to initialize known or '--initifneeded'")
            sys.exit(-1)

    # We are good!
    logging.info("...known results exist.")
    return False


def convert_output_to_latest(model_dir, prediction_file_path: Path):
    """
    Convert the test output to the "latest" file. The conversion will remove the volatile parts such as timestamps,
    model hash, etc.
    :param model_dir: The model directory to convert.
    :param prediction_file_path: The prediction file to convert.
    :return:
    """
    # Compare output. First, strip the timestamps and other variable stuff
    logging.info(f"Converting the output in {str(model_dir)}...")
    with open(str(prediction_file_path)) as pred_file:
        data = json.load(pred_file)
        del data['testTimes']
        del data['testOptions']['modelHash']
        with open(model_dir / latest_file_name, 'w') as out_file:
            json.dump(data, out_file, indent=4, sort_keys=True)


def init_test_results(model_dir: Path) -> None:
    """
    Renames "latest" results to "known" results.
    :param model_dir: The model dir to rename.
    """
    latest = model_dir / latest_file_name
    known = model_dir / known_file_name

    logging.info(f">>> Initializing known results in {model_dir} <<<")
    if known.exists():
        known.unlink()
    if not latest.exists():
        logging.error(f"Latest does NOT exist: {str(latest)}")
    latest.rename(known)


def compare_test_results(model_dir: Path) -> int:
    """
    Compares the latest results to the known results in the model directory.
    :param model_dir: The model dir
    :return: 0 if no errors, 1 if an error
    """
    latest = model_dir / latest_file_name
    known = model_dir / known_file_name

    if known.exists():
        if not filecmp.cmp(known, latest):
            logging.error(f">>> Unit test FAILED. 'diff -y {str(known)} {str(latest)} | more' <<<")
            return 1
        else:
            logging.info(f">>> Successful results for for {model_dir} <<<")
    else:
        logging.error(f"Unit test FAILED! No 'known_ut_results.json' to compare to! "
                      f"Consider using \'--init\' to initialize known.")
        return 1

    return 0


def check_training_accuracy(model_name, model_dir, min_accuracy):
    """
    Checks the accuracy of the training run from the output against the minimum accuracy and returns True
    if the accuracy exceeds the minimum accuracy. Also logs results.
    :param model_name: The model name for logging.
    :param model_dir: The model dir to check.
    :param min_accuracy: The minimum accuracy.
    :return:
    """
    output_path = model_dir / "train_out.json"
    with open(output_path) as json_file:
        results = json.load(json_file)

        # Our last accuracy should be 1.0
        found_acc = results['trainingResults']['accuracy'][-1]
        if found_acc >= min_accuracy:
            logging.info(f"Training accuracy for {model_name} of {found_acc} exceeded min accuracy: {min_accuracy}.")
            return True
        else:
            logging.info(
                f"Training accuracy for {model_name} of {found_acc} did NOT exceed min accuracy: {min_accuracy}."
                f"See 'train_out.json' for details.")

    return False


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
            logging.warning(f"Missing items in pattern {pattern}!")
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
            logging.warning(f"Found items in pattern {pattern}!")
            count += 1

    return count


def get_experiment_file_list():
    """ :return: List of expected files for the experiment directory. """
    return [
        "Simple binary test ROC.png",
        "System Test 1 ROC class 402,629,778,816,959.png",
        "System Test 1 ROC class 45,128,225,319,372.png",
        "System Test Summary.csv",
        "System Test Summary.md"
    ]


def get_experiment_dry_run_list():
    """ :return: Files expected in the experiment directory after the dry run. """
    return [
        "log_experiment_dryrun.txt"
    ]


def get_model_train_file_patterns() -> list:
    """ :return: List of patterns that we generate in the models directory during training """
    files = ['train_out.json',
             'train_out.png',
             'log_train.txt',
             'model.pt',
             'predictions_*.json',
             'log_predictions_*.txt']
    return files


def get_model_dry_run_file_patterns() -> list:
    """ :return:  List of patterns that we generate in the models directory dry run """
    files = ['log_train_dryrun.txt',
             'model_summary.txt']
    return files


def clean_experiment(runner, experiment_name: str) -> None:
    """
    Calls the clean mode of run_experiment
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    """
    show_banner(f"Running Experiment (CLEAN)...")

    args = [str(runner.bin_dir / 'jb_run_experiment'), experiment_name, "--clean", "--commit"]
    runner.run(args, add_roots=True)


def check_clean(runner, experiment_name: str, model_names: list) -> int:
    """
    Checks to make sure that run_experiment cleaned all the appropriate files.
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    :param model_names: The models to check.
    :return: Number of files found that should have been cleaned
    """
    # Look for files that SHOULD NOT be there
    experiment_dir = runner.workspace_root / 'experiments' / experiment_name
    file_count = count_found_patterns(experiment_dir, get_experiment_dry_run_list())
    file_count += count_found_patterns(experiment_dir, get_experiment_file_list())

    for model_name in model_names:
        model_dir = runner.workspace_root / 'models' / model_name
        file_count += count_found_patterns(model_dir, get_model_dry_run_file_patterns())
        file_count += count_found_patterns(model_dir, get_model_train_file_patterns())

    if file_count > 0:
        logging.error(f"Failed to clean experiment. Found {file_count} files")

    return file_count


def dry_run_experiment(runner, experiment_name: str) -> None:
    """
    Calls the dry run mode of run_experiment
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    """
    show_banner(f"Running Experiment (DRY RUN)...")

    args = [str(runner.bin_dir / 'jb_run_experiment'), experiment_name, "--dryrun"]
    runner.run(args, add_roots=True)


def check_for_files(runner, stage, experiment_name: str, experiment_patterns: list, model_names: list,
                    model_patterns: list) -> int:
    """
    Checks to see if files (patterns) exist in the directory.
    :param runner: The runner object.
    :param stage: The stage of the run for logging
    :param experiment_name: The experiment we are examining.
    :param experiment_patterns: Experiments patterns
    :param model_names: The models to check.
    :param model_patterns: Patterns of model files to check.
    :return: Number of missing files
    """
    experiment_dir = runner.workspace_root / 'experiments' / experiment_name
    missing = count_missing_patterns(experiment_dir, experiment_patterns)

    for model_name in model_names:
        model_dir = runner.workspace_root / 'models' / model_name
        missing += count_missing_patterns(model_dir, model_patterns)

    if missing > 0:
        logging.error(f"Failed {stage}. Missing {missing} files")

    return missing


def commit_experiment(runner, experiment_name) -> None:
    """
    Performs the training part of the experiment.
    :param runner: The runner object.
    :param experiment_name: The experiment to execute.
    """
    show_banner(f"Running Experiment (COMMIT)...")

    args = [str(runner.bin_dir / 'jb_run_experiment'), experiment_name, "--commit"]
    runner.run(args, add_roots=True)


def check_accuracies(runner, test_set) -> int:
    """
    Checks that the accuracies of all the models exceed the minimum accuracy for the test.
    :param runner: The runner object.
    :param test_set: The test set describing all the models.
    :return: Number of failures.
    """
    # Remove all latest results
    for model_name, test_name, min_accuracy in test_set:
        model_dir = runner.workspace_root / 'models' / model_name
        latest = model_dir / latest_file_name
        if latest.exists():
            latest.unlink()

    # Check the accuracies of the training runs
    failures = 0
    for model_name, test_name, min_accuracy in test_set:
        # Find all the directories and files
        model_dir = runner.workspace_root / 'models' / model_name
        predictions_file_name = Path(test_name).name
        prediction_file_path = model_dir / f"predictions_{predictions_file_name}"

        # Check the accuracy against what we expect before we move forward
        # If good, make a latest results file that can be checked
        if check_training_accuracy(model_name, model_dir, min_accuracy):
            convert_output_to_latest(model_dir, prediction_file_path)
        else:
            failures += 1

    return failures


def compare_latest_if_exists(model_dirs):
    """
    Compares all the latest results with the known results.
    :param model_dirs: The model directories to check.
    :return: Number of failures.
    """
    # Check whatever latest we have.
    failures = 0
    for model_dir in model_dirs:
        latest = model_dir / latest_file_name
        if latest.exists():
            failures += compare_test_results(model_dir)

    return failures


def do_experiment(runner, init_known, experiment_name, test_set):
    """
    Executes the specified experiment from clean, dry run, commit and checking results.
    :param runner: The runner object.
    :param init_known: Should the "known results" be initialized.
    :param experiment_name: The name of the experiment.
    :param test_set: A test set describing what the experiment should do.
    :return:
    """
    model_names = [x[0] for x in test_set]
    model_dirs = [runner.workspace_root / 'models' / x for x in model_names]

    # Before we do anything remove all the "latest" results.
    for model_dir in model_dirs:
        latest = model_dir / latest_file_name
        if latest.exists():
            latest.unlink()

    failures = 0

    clean_experiment(runner, experiment_name)
    failures += check_clean(runner, experiment_name, model_names)

    dry_run_experiment(runner, experiment_name)
    failures += check_for_files(runner, "DRY RUN", experiment_name, get_experiment_dry_run_list(),
                                model_names, get_model_dry_run_file_patterns())

    commit_experiment(runner, experiment_name)
    failures += check_for_files(runner, "COMMIT", experiment_name, get_experiment_file_list(),
                                model_names, get_model_train_file_patterns())

    # Now at this point check the accuracies. These produce 'latest' files if ok.
    show_banner("Checking accuracies...")
    acc_failures = check_accuracies(runner, test_set)
    failures += acc_failures

    # Now, if they want it inited, do it if everything is okay.
    if init_known:
        if acc_failures > 0:
            logging.error(f"{acc_failures} tests failed to meet min accuracy. See log. NOT INITING")
        else:
            show_banner("Initing test results...")
            for model_dir in model_dirs:
                init_test_results(model_dir)
            logging.info("Tests INITED. This does NOT mean they passed.")
    else:
        # Since we have been inited, compare each set of output that we have
        failures += compare_latest_if_exists(model_dirs)

    return failures


def do_reinit(runner, test_set) -> None:
    """
    Check the training accuracies and if good regenerate the latest results and copy to the known results.
    This is useful for when the experiment has been run successfully (manually)
    :param runner: The runner object.
    :param test_set: A test set describing what the experiment should do.
    """
    show_banner("Re-initializing")
    # To reinit we check that each training set meets minimum accuracy convert to latest and copy over.
    model_dirs = [runner.workspace_root / 'models' / x[0] for x in test_set]

    # Before we do anything remove all the "latest" results.
    for model_dir in model_dirs:
        latest = model_dir / latest_file_name
        if latest.exists():
            latest.unlink()

    # Sanity check that we have output
    missing = 0
    for model_dir in model_dirs:
        out_path = model_dir / 'train_out.json'
        if not out_path.exists():
            logging.error(f"Missing train_out.json in {model_dir}.")
            missing += 1

    if missing > 0:
        logging.error("Missing a train_out.json. Terminating re-init. EXITING")
        sys.exit(-1)

    # Okay, now check accuracies
    show_banner("Checking accuracies...")
    failures = check_accuracies(runner, test_set)
    if failures > 0:
        logging.error(f"{failures} tests failed to meet min accuracy. See log. NOT RE-INITING")
    else:
        show_banner("RE-INITING test results...")
        for model_dir in model_dirs:
            init_test_results(model_dir)


def do_generate_experiment(runner) -> int:
    """
    Really really simple test for generate experiment.
    :param runner:
    :return:
    """
    experiment_name = "generatedExperiment"
    show_banner(f"Running generate experiment {experiment_name}...")

    model_names = ['generatedExperiment/' + x for x in ['lr_0_lrSched_0', 'lr_0_lrSched_1', 'lr_1_lrSched_0',
                                                        'lr_1_lrSched_1']]

    models_dir = runner.workspace_root / 'models'

    # Remove the config files
    for model_name in model_names:
        config_path = models_dir / model_name / "config.json"
        if config_path.exists():
            config_path.unlink()

    # Run the generator
    args = [str(runner.bin_dir / 'jb_generate_experiments'), 'generatedExperiment']
    runner.run(args)

    # At this point we should have config files. We don't diff them or anything yet.
    failures = 0
    for model_name in model_names:
        config_path = models_dir / model_name / "config.json"
        if not config_path.exists():
            logging.error(f"Expected to find config file at {config_path}")
            failures += 1

    # Do the dry run and see if it works
    dry_run_experiment(runner, experiment_name)
    failures += check_for_files(runner, "DRY RUN",
                                experiment_name, get_experiment_dry_run_list(),
                                model_names, get_model_dry_run_file_patterns())

    return failures


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Script for executing unit tests. Use --init to initialize known '
                                                 'good files for this host.')
    parser.add_argument('-d', '--dataRoot', type=str, default=None,
                        help='Root of data directory. Overrides values pulled from config files.')
    parser.add_argument('--init', default=False, action='store_true',
                        help='Set to true to run experiment from scratch and init known files.')
    parser.add_argument('--reinit', default=False, action='store_true',
                        help='Set to true to initialize known good output WITHOUT re-running experiment. '
                             'Incompatible with init')
    parser.add_argument('--initifneeded', default=False, action='store_true',
                        help='Set to true to automatically init if not inited. Incompatible with init or reinit.')

    args = parser.parse_args()

    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    workspace_root = script_dir.parent.absolute()
    bin_dir = workspace_root / 'bin'
    data_root = None
    if args.dataRoot is not None:
        data_root = Path(args.dataRoot).absolute()

    os.chdir(workspace_root)

    # Set up our environment variables so that we are deterministic
    # https://github.com/NVIDIA/tensorflow-determinism
    sub_env = os.environ.copy()
    sub_env['TF_DETERMINISTIC_OPS'] = '1'
    sub_env['PYTHONHASHSEED'] = '0'

    # Set up the thing to run the command
    runner = CommandRunner(bin_dir, sub_env, workspace_root, data_root)

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
        do_reinit(runner, test_set)
        return

    if not args.init:
        args.init = check_for_init(args.initifneeded)

    # Now, run all the tests
    failures = do_experiment(runner, args.init, "pytorchSystemTest", test_set)
    failures += do_generate_experiment(runner)

    if args.init:
        if failures == 0:
            logging.info(f"###### Known results initialized. NOT A VALID TEST.")
        else:
            logging.info(f"###### Some errors during initialization! See log for details!")
    else:
        if failures == 0:
            logging.info(f"###### SUCCESS - No test failures! :)")
        else:
            logging.info(f"###### FAILURE - Number of failures: {failures}")

    sys.exit(failures)


if __name__ == "__main__":
    main()
