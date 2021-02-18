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

import configparser
import os
import juneberry
import argparse
import logging

from pathlib import Path

# This describes what our valid types are and the keys for those types. This is the authoritative
# list of stuff we pull in.
KEYS_TYPES = {"WORKSPACE_ROOT": str,
              "DATA_ROOT": str,
              "NUM_WORKERS": int,
              "TENSORBOARD_ROOT": str
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
        logging.info(f"Found ini file via environment variable: {juneberry_config}")
        configs.append(Path(juneberry_config))

    home_config = Path.home() / ini_name
    if home_config.exists():
        logging.info(f"Found '{ini_name}' file in home directory: {home_config}")
        configs.append(home_config)

    working_config = Path.cwd() / ini_name
    if working_config.exists():
        logging.info(f"Found '{ini_name}' file in working directory: {working_config}")
        configs.append(working_config)

    return configs


def load_and_set_configs_variables(overrides, section_name=None):
    """
    Finds the list of config files, loads the variables from those files, and sets them in the package namespace.
    :param overrides: A dictionary of values to be used as overrides. Usually from the command line.
    :param section_name: Optional section name to look up in the config file for model or experiment specific overrides.
    :return: Number of configuration errors.
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
        logging.info(f"Found section for '{section_name}'")

    # We want a version without the Nones
    overrides_copy = {k: v for k, v in overrides.items() if v is not None}

    # Walk all the vars on juneberry and look for those in the config
    errors = 0
    for k in KEYS_TYPES.keys():
        value = None
        source = ''

        # Look for new values
        if k in overrides_copy:
            value = overrides_copy[k]
            del overrides_copy[k]
            source = 'overrides'
        elif config.has_option(section_name, k):
            # They always come through as strings and we don't want quotes.
            value = config[section_name][k].strip('"')
            source = 'ini file'

        # If we have a value then try to cast it to the right type
        if value is not None:
            jb_type = KEYS_TYPES[k]
            try:
                converted = jb_type(value)
                vars(juneberry)[k] = converted
                logging.info(f"Found '{k}' in {source}. Setting to {converted}")

            except ValueError:
                logging.error(f"Failed to convert {k} to type {jb_type}, leaving default.")
                errors += 1

    if len(overrides_copy) > 0:
        logging.error(f"We have some unused overrides: {overrides_copy}")

    # Now at this point we MUST have some specific values to continue
    for required in ['WORKSPACE_ROOT', 'DATA_ROOT']:
        if vars(juneberry)[required] is None:
            logging.error(f"Required value {required} not set from INI or override!")
            errors += 1

    return errors


def main():
    # TODO: This should go into a config debugging utility like "jb_ini_tool" or something that can also
    #  make a pre-populated one...
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Shows what the config values would be via files.")
    parser.add_argument('-s', '--section', default="nomodel", help="Section name")

    # Args specific to this script.
    args = parser.parse_args()
    load_and_set_configs_variables({}, args.section)

    print("Found the following values")
    for k in KEYS_TYPES:
        print(f"-- {k} - {vars(juneberry)[k]}")


if __name__ == "__main__":
    main()
