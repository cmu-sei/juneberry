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

import datetime
import juneberry
import logging
import os
from pathlib import Path
import sys

import juneberry.config_loader as ini_loader
import juneberry.jb_logging as jblogging
import juneberry.lab as Lab

logger = logging.getLogger(__name__)


def run_main(main_fn, the_logger):
    """
    Calls the main function handling highest level exceptions and logging as appropriate.
    :param main_fn: The function to call.
    :param the_logger: The logger to emit messages to.
    :return: None.
    """
    try:
        main_fn()
    except SystemExit as e:
        if e.code != 0:
            the_logger.exception(e)
            sys.exit(e.code)
    except BaseException as e:
        # By using BaseException we also get keyboard errors which allows us to log that someone cancelled.
        the_logger.exception(e)
        sys.exit(-1)


def setup_args(parser, add_data_root=True) -> None:
    parser.add_argument('-w', '--workspace', type=str, default=None,
                        help='Root of workspace directory. Overrides values pulled from config files.')
    if add_data_root:
        parser.add_argument('-d', '--dataRoot', type=str, default=None,
                            help='Root of data directory. Overrides values pulled from config files.')
    parser.add_argument('-t', '--tensorboard', type=str, default=None,
                        help='TensorBoard log directory. Overrides values pulled from config files.')
    parser.add_argument('-s', '--silent', default=False, action='store_true',
                        help='Silent flag to silence output to console. Default is to show to console.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Verbose flag that will log DEBUG messages. Default is off.')
    parser.add_argument('-c', '--machineClass', type=str, default=None,
                        help='The class of machine.')


def setup_workspace(args, *, log_file, log_prefix="", add_data_root=True, model_name=None, name="juneberry",
                    banner_msg=None) -> Lab:
    """
    Sets up the workspace package variable, sets the current working directory properly, and inits logging
    into the log file.
    :param args: The args from parseargs.
    :param log_file: A workspace relative log file to write to. Can be None to use no log file.
    :param log_prefix: OPTIONAL: A string to add as a prefix to log entries; can be "".
    :param add_data_root: True to identify and set the data root.
    :param model_name: OPTIONAL model name to use for juneberry options.
    :param name: OPTIONAL name for logger.
    :param banner_msg: OPTIONAL message for a startup banner.
    :return: The lab object with the configuration
    """
    if log_file is not None:
        log_file = Path(log_file)

    # Load workspace root and data root from config variables.
    # NOTE: We need to temporarily set up the logger to just use the console because we
    # don't know where to store it yet.
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    if banner_msg:
        jblogging.log_banner(logger, banner_msg)
    data_root = None
    if add_data_root:
        data_root = args.dataRoot
    overrides = {"WORKSPACE_ROOT": args.workspace, "DATA_ROOT": data_root, "TENSORBOARD_ROOT": args.tensorboard,
                 "MACHINE_CLASS": args.machineClass}
    lab, errors = juneberry.config_loader.setup_lab(overrides, model_name)
    logger.info(f"Lab configuration: {lab}")
    if errors > 0:
        print("Failed to set up Juneberry environment.  See console for details. EXITING!!")
        sys.exit(-1)

    # Change to the workspace root so we can find everything.
    os.chdir(lab.workspace())

    # Convert verbose argument to proper logging level.
    level = logging.DEBUG if args.verbose else logging.INFO

    # If there's an existing log_train file, rename it to include the last modified timestamp in the filename.
    # The current run will then get logged into a fresh log_train.txt file.
    if log_file is not None and log_file.exists():
        if int(os.environ.get('JB_REMOVE_OLD_LOG', 0)) == 1:
            log_file.unlink()
        else:
            time_val = datetime.datetime.fromtimestamp(os.path.getmtime(log_file)).strftime("%m%d%y_%H%M")
            new_file_path = Path(log_file.parent, f"{log_file.stem}_{time_val}.txt")
            os.rename(log_file, new_file_path)

    # Set up logging now that we have a workspace to log to.
    jblogging.setup_logger(log_file, log_prefix=log_prefix, log_to_console=not args.silent, level=level, name=name)

    # Indicates when DEBUG level messages have been enabled.
    logger.debug(f"DEBUG messages enabled.")

    logger.info(f"Default workspace: {lab.workspace()}")
    if add_data_root:
        logger.info(f"Default data_root: {lab.data_root()}")

    return lab


def setup_for_single_model(args, *, log_file, model_name, log_prefix="", add_data_root=True, banner_msg=None) -> Lab:
    # TODO: Come up with a better name for this call
    # Check that the model directory is there. We need to do this before setting up logging
    # because the logger wants to write to the directory.

    lab = setup_workspace(args, log_file=log_file, log_prefix=log_prefix, add_data_root=add_data_root,
                          model_name=model_name, banner_msg=banner_msg)
    mm = lab.model_manager(model_name)
    mm.ensure_model_directory()

    return lab


def setup_for_experiment_creation(args, experiment_manager, *, log_file, log_prefix="", add_data_root=True) -> Lab:
    # Check that the experiment directory is there.   We need to do this before setting up logging
    # because the logger wants to write to the directory.
    experiment_manager.ensure_experiment_directory()

    return setup_workspace(args, log_file=log_file, log_prefix=log_prefix, add_data_root=add_data_root)

# Log files are model relative
# jbfs determines where model log files belong
