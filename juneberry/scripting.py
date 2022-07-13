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
import logging
import os
import platform
from pathlib import Path
import sys

import juneberry.jb_logging as jblogging
from juneberry.lab import Lab

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


def make_default_lab_values(workspace: str):
    """
    This routine generates "standard" locations for the basic sibling directory structure.
    :return: Default values for a script for workspace, data_root, tensorboard, profile name, etc.
    """
    # NOTE: These names match the Lab, NOT the command line names
    ws = Path(workspace)
    return {
        "workspace": str(ws),
        "data_root": str((ws.parent / "dataroot").absolute()),
        "tensorboard": str((ws.parent / "tensorboard").absolute()),
        "profile_name": "default",
        "model_zoo": None,
        "cache": str(Path.home() / ".juneberry" / "cache")
    }


def resolve_arg(arg, env_name, default, is_path: bool = False):
    """
    Determines the right value from the arg, environment variable, or default.
    :param arg: The passed in arg
    :param env_name: The name of an environment variable to check
    :param default: A default value
    :param is_path: If a path resolves to the absolute
    :return: The final value
    """
    if arg is not None:
        val = arg
    elif env_name in os.environ:
        val = os.environ[env_name]
    else:
        val = default

    if is_path:
        val = str(Path(val).absolute())

    return val


def resolve_lab_args(args):
    """
    This routine aggregates:
    - reasonable defaults
    - standard Juneberry environment variables
    - overrides
    into one set of values.
    NOTE: This does NOT require or check for directory existence.
    :return: A set of arguments for constructing the lab.
    """
    # STEP 1: Find the workspace first, so we can start with that structure.
    def_ws = resolve_arg(args.workspace, "JUNEBERRY_WORKSPACE", Path.cwd(), is_path=True)

    # STEP 2: Populate the "normal" structure based on the workspace relative values.
    vals = make_default_lab_values(def_ws)

    # STEP 3: Overlay the *remaining* (not ws) values from environment vars.
    # NOTE: This is where we KNOW the name of the args and are coupled with setup_args
    vals['data_root'] = resolve_arg(args.dataRoot, 'JUNEBERRY_DATA_ROOT', vals['data_root'], is_path=True)
    vals['tensorboard'] = resolve_arg(args.tensorboard, 'JUNEBERRY_TENSORBOARD', vals['tensorboard'], is_path=True)
    vals['profile_name'] = resolve_arg(args.profileName, 'JUNEBERRY_PROFILE_NAME', vals['profile_name'])
    vals['model_zoo'] = resolve_arg(args.zoo, 'JUNEBERRY_MODEL_ZOO', vals['model_zoo'])
    vals['cache'] = resolve_arg(args.cache, 'JUNEBERRY_CACHE', vals['cache'], is_path=True)

    # Now we have the basics, let's return those.
    return vals


def setup_args(parser) -> None:
    parser.add_argument('-w', '--workspace', type=str, default=None,
                        help='Root of workspace directory. Overrides other values.')
    parser.add_argument('-d', '--dataRoot', type=str, default=None,
                        help='Root of data directory. Overrides other values.')
    parser.add_argument('-t', '--tensorboard', type=str, default=None,
                        help='TensorBoard log directory. Overrides other values.')
    parser.add_argument('-s', '--silent', default=False, action='store_true',
                        help='Silent flag to silence output to console. Default is to show to console.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Verbose flag that will log DEBUG messages. Default is off.')
    parser.add_argument('-p', '--profileName', type=str, default=None,
                        help='The name of the lab profile.')
    parser.add_argument('-l', '--logDir', default=Path.cwd(), required=False,
                        help="Directory where the log file will be saved. Default is the current working directory.")
    parser.add_argument('--zoo', default=None, required=False,
                        help="Url for the optional juneberry model zoo.")
    parser.add_argument('--cache', default=None, required=False,
                        help="Directory where to cache files.  The default is ~/.juneberry/cache")


def construct_validated_lab_from_args(args):
    """
    Constructs and validates the lab object from the arguments
    :param args: The parse args arguments
    :return: The constructed lab.
    """
    # Resolve all the args we want for the lab.
    lab_args = resolve_lab_args(args)

    # Check the lab args and make the lab. NOTE: validate_args will exit if it fails
    Lab.validate_args(**lab_args)
    lab = Lab(**lab_args)

    return lab


def standard_line_prefix(dryrun):
    if dryrun:
        return "<<DRYRUN>> "
    else:
        return ""


def setup_workspace(args, *, log_file, log_prefix="", name="juneberry", banner_msg=None) -> Lab:
    """
    Sets up the lab (workspace, etc) and inits the log file.
    :param args: The args from setup args and parseargs.
    :param log_file: OPTIONAL: Path to log file. If a relative path, must be relative ot the workspace.
    :param log_prefix: OPTIONAL: A string to add as a prefix to log entries; can be "".
    :param name: OPTIONAL name for logger.
    :param banner_msg: OPTIONAL message for a startup banner.
    :return: The lab object with the configuration
    """
    # LOGGING NOTE:
    # Everything up to the comment needs to be quiet and work without logging, UNLESS it fails and exits.
    # Once we have the log file all logging is fair game.

    # Load workspace root and data root from config variables.
    # NOTE: We need to temporarily set up the logger to just use the console because we
    # don't know where to store it yet.
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    # Make the lab
    lab = construct_validated_lab_from_args(args)

    # Change to the workspace root so we can find everything.
    os.chdir(lab.workspace())

    # =======================================
    # Logging prep

    # Use the user supplied function to return the log name.
    if log_file is not None:
        log_file = Path(log_file)

    # Make sure the path to the log file is ready
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # If there's an existing log file, rename it to include the last modified timestamp in the filename.
    # The current run will then get logged into a fresh file.
    add_log_keeping_logs = False
    if log_file is not None and log_file.exists():
        if int(os.environ.get('JUNEBERRY_REMOVE_OLD_LOGS', 0)) == 1:
            log_file.unlink()
        else:
            add_log_keeping_logs = True
            time_val = datetime.datetime.fromtimestamp(os.path.getmtime(log_file)).strftime("%m%d%y_%H%M")
            new_file_path = Path(log_file.parent, f"{log_file.stem}_{time_val}.txt")
            os.rename(log_file, new_file_path)

    # Convert verbose argument to proper logging level.
    level = logging.DEBUG if args.verbose else logging.INFO

    # Set up logging now that we have a workspace to log to.
    jblogging.setup_logger(log_file, log_prefix=log_prefix, log_to_console=not args.silent, level=level, name=name)

    # =======================================
    # Log away!
    if banner_msg:
        jblogging.log_banner(logger, banner_msg)

    logger.info(f"Changing directory to workspace: '{lab.workspace()}'")
    if log_file:
        logger.info(f"Using log file: '{log_file}'")
    if add_log_keeping_logs:
        logger.info("Keeping old log files. Specify 'JUNEBERRY_REMOVE_OLD_LOGS=1' to remove them.")

    # Indicates when DEBUG level messages have been enabled.
    logger.debug(f"DEBUG messages enabled.")

    logger.info(f"Using workspace:    {lab.workspace()}")
    logger.info(f"Using data root:    {lab.data_root()}")
    logger.info(f"Using tensorboard:  {lab.tensorboard}")
    logger.info(f"Using profile name: {lab.profile_name}")
    logger.info(f"Hostname:           {platform.uname().node}")

    return lab


def setup_logging_for_script(args, script_name: str = None):
    """
    This function is responsible for setting up logging in a Juneberry script. This function is typically used when
    the root juneberry logger cannot be set up by other means.
    :param args: The args from parseargs.
    :param script_name: A string indicating which script the log messages are from. Used to build
    the name of the log file.
    :return: Nothing.

    """
    # Build the location of the log using the desired log directory (from args) and the name of
    # the script calling this function.
    if script_name is None:
        script_name = Path(sys.argv[0]).stem
    log_file = Path(args.logDir) / f"log_{script_name}.txt"

    # Set the logging level and setup the logger.
    level = logging.DEBUG if args.verbose else logging.INFO
    jblogging.setup_logger(log_file, log_prefix="", log_to_console=not args.silent, level=level)

    # A simple test message indicating where the log messages are being saved to.
    logger.info(f"Logging initialized for {Path(sys.argv[0]).absolute()}")
    logger.info(f"Log messages are beings saved to {log_file}")


def setup_workspace_and_model(args, *, log_file, model_name, log_prefix="", banner_msg=None):
    # Setup up the workspace like normal
    lab = setup_workspace(args, log_file=log_file, log_prefix=log_prefix, banner_msg=banner_msg)

    # Double check the model.
    mm = lab.model_manager(model_name)
    mm.ensure_model_directory()
    mm.setup()
    return lab
