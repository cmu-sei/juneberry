#! /usr/bin/env python3

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

import juneberry
import logging
import os
import sys

import juneberry.config_loader as ini_loader
import juneberry.jb_logging as jblogging


def setup_args(parser, add_data_root=True) -> None:

    parser.add_argument('-w', '--workspace', type=str, default=None,
                        help='Root of workspace directory. Overrides values pulled from config files.')
    if add_data_root:
        parser.add_argument('-d', '--dataRoot', type=str, default=None,
                            help='Root of data directory. Overrides values pulled from config files.')
    parser.add_argument('-t', '--tensorboard', type=str, default=None,
                        help='TensorBoard log directory. Overrides values pulled from config files.')
    parser.add_argument('-s', '--silent', default=True, action='store_false',
                        help='Silent flag to silence output to console. Default is to show to console.')


def setup_workspace(args, log_file, log_prefix, add_data_root=True, model_name=None) -> None:
    """
    Sets up the workspace package variable, sets the current working directory properly, and inits logging
    into the log file.
    :param args: The args from parseargs.
    :param log_file: The log file to write to.
    :param log_prefix: A string to add as a prefix to log entries.
    :param add_data_root: True to identify and set the data root.
    :param model_name: Optional model name to use for juneberry options.
    """
    # Load workspace root and data root from config variables.
    # NOTE: We need to temporarily set up the logger to just use the console because we
    # don't know where to store it yet.
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    data_root = None
    if add_data_root:
        data_root = args.dataRoot
    overrides = {"WORKSPACE_ROOT": args.workspace, "DATA_ROOT": data_root, "TENSORBOARD_ROOT": args.tensorboard}
    if juneberry.config_loader.load_and_set_configs_variables(overrides, model_name) > 0:
        print("Failed to set up Juneberry environment.  See console for details. EXITING!!")
        sys.exit(-1)

    # Change to the workspace root so we can find everything
    os.chdir(juneberry.WORKSPACE_ROOT)

    # Set up logging now that we have a workspace to log to.
    jblogging.setup_logger(log_file, log_prefix=log_prefix, log_to_console=args.silent)

    logging.info(f"Using workspace: {juneberry.WORKSPACE_ROOT}")


def setup_for_single_model(args, log_file, log_prefix, model_manager, add_data_root=True) -> None:
    # Check that the model directory is there.   We need to do this before setting up logging
    # because the logger wants to write the directory.
    model_manager.ensure_model_directory()

    setup_workspace(args, log_file, log_prefix, add_data_root, model_name=model_manager.model_name)

    # Now that we have a log file, log the workspace and data
    if add_data_root:
        logging.info(f"Using dataroot: {juneberry.DATA_ROOT}")


def setup_for_experiment_creation(args, log_file, log_prefix, experiment_creator, add_data_root=True) -> None:
    # Check that the experiment directory is there.   We need to do this before setting up logging
    # because the logger wants to write to the directory.
    experiment_creator.ensure_experiment_directory()

    setup_workspace(args, log_file, log_prefix, add_data_root)

    # Now that we have a log file, log the workspace and data
    if add_data_root:
        logging.info(f"Using dataroot: {juneberry.DATA_ROOT}")
