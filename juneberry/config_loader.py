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

import argparse
import configparser
import logging
import os
from pathlib import Path

from juneberry.lab import Lab

logger = logging.getLogger(__name__)

# This describes what our valid types are and the keys for those types. This is the authoritative
# list of stuff we pull in.
KEYS_TYPES = {"WORKSPACE_ROOT": str,
              "DATA_ROOT": str,
              "NUM_WORKERS": int,
              "TENSORBOARD_ROOT": str,
              "MACHINE_CLASS": str
              }


def get_configs(env_config='JUNEBERRY_CONFIG', ini_name='juneberry.ini'):
    """
    Locates the available config files and returns them in an order of highest priority last.
    :param env_config: The name of the environment variable to check.
    :param ini_name: The name of the ini file.
    :return: List of config file Path objects.
    """
    configs = []
    juneberry_config = os.environ.get(env_config)
    if juneberry_config is not None:
        logger.info(f"Found ini file via environment variable: {juneberry_config}")
        configs.append(Path(juneberry_config))

    home_config = Path.home() / ini_name
    if home_config.exists():
        logger.info(f"Found '{ini_name}' file in home directory: {home_config}")
        configs.append(home_config)

    working_config = Path.cwd() / ini_name
    if working_config.exists():
        logger.info(f"Found '{ini_name}' file in working directory: {working_config}")
        configs.append(working_config)

    return configs


def setup_lab(overrides: dict, section_name: str = None):
    """
    Finds the list of config files, loads the variables from those files, and sets in the lab object.
    :param overrides: A dictionary of values to be used as overrides. Usually from the command line.
    :param section_name: Optional section name to look up in the config file for model or experiment specific overrides.
    :return: Lab object, number of configuration errors
    """

    # NOTES:
    # Config keys and apis in the file are normalized to lower case.
    # Config values in the juneberry __init__.py are upper case because that's the way python likes it.
    # We want to keep values as set in the actual config file, so we only want to overwrite them if they
    #   are actually provided to us as overrides or in the config file. So we are very sensitive to None.
    # So the basic process is to:
    # - Just read all the files
    # - If an override exists, use it, else use the value from the ini else leave default in file.
    # - Make sure the type matches that in the vars.

    # Find our config files. The built in one will read ones that don't exist but we like the extra
    # diagnostics of doing it ourselves.
    configs = get_configs()

    # Read values from the config files
    config = configparser.ConfigParser()
    config.read(configs)

    # If there isn't a specific section then just add it. Everything will just pass through to defaults.
    if section_name is None:
        section_name = "nosection"
    if not config.has_section(section_name):
        config.add_section(section_name)
    else:
        logger.info(f"Found section for '{section_name}'")

    # We want a version without the Nones
    overrides_copy = {k: v for k, v in overrides.items() if v is not None}

    lab_arg_mapper = {
        "WORKSPACE_ROOT": 'workspace',
        "DATA_ROOT": 'data_root',
        "TENSORBOARD_ROOT": 'tensorboard',
        "NUM_WORKERS": 'num_workers',
        "NUM_GPUS": 'num_gpus',
        "MACHINE_CLASS": 'machine_class'
    }
    lab_args = {}


    # Walk each input value and create arguments for the lab object
    errors = 0
    for k in KEYS_TYPES.keys():
        value = None
        source = ''

        # Look for new values
        if k in overrides_copy:
            value = overrides_copy[k]
            del overrides_copy[k]
            source = 'overrides'

        # Grab environment variable if machine_class not specified from command line
        elif k == 'MACHINE_CLASS' and k not in overrides_copy and os.environ.get('JUNEBERRY_MACHINE_CLASS') is not None:
            machine_class = os.environ.get('JUNEBERRY_MACHINE_CLASS')
            value = str(machine_class)
            source = 'env var'

        elif config.has_option(section_name, k):
            # They always come through as strings and we don't want quotes.
            value = config[section_name][k].strip('"')
            source = 'ini file'

        # If we have a value then try to cast it to the right type
        if value is not None:
            jb_type = KEYS_TYPES[k]
            try:
                converted = jb_type(value)
                logger.info(f"Found '{k}' in {source}. Setting '{lab_arg_mapper[k]}' to {converted}")
                lab_args[lab_arg_mapper[k]] = converted

            except ValueError:
                logger.error(f"Failed to convert {k} to type {jb_type}, leaving default.")
                errors += 1

    if len(overrides_copy) > 0:
        logger.error(f"We have some unused overrides: {overrides_copy}")

    # Now at this point we MUST have some specific values to continue
    for required in ['workspace', 'data_root']:
        if required not in lab_args:
            logger.error(f"Required value {required} not set from INI or override!")
            lab_args[required] = None
            errors += 1

    # Check environment variable if machine_class not found in command line or ini
    if 'machine_class' not in lab_args:
        machine_class = os.environ.get('JUNEBERRY_MACHINE_CLASS')
        if machine_class is not None:
            lab_args['machine_class'] = str(machine_class)

    # Now return a lab args object initialized to these values
    return Lab(**lab_args), errors


def main():
    # TODO: This should go into a config debugging utility like "jb_ini_tool" or something that can also
    #  make a pre-populated one...
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Shows what the config values would be via files.")
    parser.add_argument('-s', '--section', default="nomodel", help="Section name")

    # Args specific to this script.
    args = parser.parse_args()
    lab, errors = setup_lab({}, args.section)

    logging.info(f"Loaded the config with {errors} errors.  Values:")
    logging.info(lab)


if __name__ == "__main__":
    main()
