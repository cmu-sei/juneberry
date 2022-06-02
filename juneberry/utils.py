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
General utilities.
"""

import logging
from pathlib import Path
import random
import re
import sys

import numpy as np

logger = logging.getLogger(__name__)


def set_seeds(seed: int):
    """
    Sets all the random seeds used by all the various pieces.
    :param seed: A random seed to use. Can not be None.
    """
    if seed is None:
        logger.error("Request to initialize with a seed of None. Exiting")
        sys.exit(-1)

    logger.debug(f"Setting numpy and random seed to: {str(seed)}")
    random.seed(seed)
    np.random.seed(seed)


def rekey(struct, key_map: dict, reverse=False) -> None:
    """
    Traverses the structure changing any dictionary keys as specified in the map.
    NOTE: Changes the struct in-place.
    :param struct: The structure to update.
    :param key_map: The keymap of keys that need to be changed.
    :param reverse: Boolean indicating whether or not to swap the keys and values.
    :return: None
    """
    if reverse:
        key_map = {v: k for k, v in key_map.items()}

    if isinstance(struct, (list, tuple)):
        for x in struct:
            rekey(x, key_map)

    # If dict, convert the key then the value
    elif isinstance(struct, dict):
        for k in list(struct.keys()):
            rekey(struct[k], key_map)
            if k in key_map:
                struct[key_map[k]] = struct[k]
                del struct[k]


def mixed_to_snake_struct_keys(struct) -> dict:
    """
    Traverses the structure converting any dictionary keys from mixedCase to snake_case
    using an automatic (non-reversible!) converter.
    NOTE: Changes the struct in-place.
    :param struct: The structure to traverse.
    :return: Dict of converted keys before -> after.
    """
    converted = {}

    # If the thing is a list or tuple, try to convert every element
    if isinstance(struct, (list, tuple)):
        for x in struct:
            converted.update(mixed_to_snake_struct_keys(x))

    # If dict, convert the key then the value
    elif isinstance(struct, dict):
        for k in list(struct.keys()):
            converted.update(mixed_to_snake_struct_keys(struct[k]))
            new_key = mixed_to_snake(k)
            if new_key != k:
                struct[new_key] = struct[k]
                del struct[k]
                converted[k] = new_key

    return converted


SNAKE_CASE_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
SNAKE_CASE_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')


def mixed_to_snake(name: str):
    """
    Converts the input name from CamelCase to snake_case.  NOTE: Not reversible!
    :param name: The name to convert.
    :return: Converted name
    """
    name = SNAKE_CASE_PATTERN_1.sub(r'\1_\2', name)
    return SNAKE_CASE_PATTERN_2.sub(r'\1_\2', name).lower()


def dict_cleaner(obj):
    """
    Creates a version of the structure converting non-serializable things.
    :param obj: The struct to convert.
    :return: A struct with all tuples instead of lists
    """
    if isinstance(obj, list):
        return list([dict_cleaner(x) for x in obj])
    if isinstance(obj, tuple):
        return tuple([dict_cleaner(x) for x in obj])
    elif isinstance(obj, dict):
        return {k: dict_cleaner(v) for k, v in obj.items()}
    else:
        if isinstance(obj, Path):
            return str(obj)
        elif type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                # This unpacks floats, etc.
                return obj.item()
        else:
            return obj


def none_stripper(obj):
    """
    Creates a version of the structure skipping keys with None values.
    :param obj: The struct to convert.
    :return: A struct with all tuples instead of lists.
    """
    if isinstance(obj, list):
        return list([none_stripper(x) for x in obj])
    if isinstance(obj, tuple):
        return tuple([none_stripper(x) for x in obj])
    elif isinstance(obj, dict):
        return {k: none_stripper(v) for k, v in obj.items() if v is not None}
    else:
        return obj


#  ____       _                       _
# |  _ \  ___| |__  _   _  __ _  __ _(_)_ __   __ _
# | | | |/ _ \ '_ \| | | |/ _` |/ _` | | '_ \ / _` |
# | |_| |  __/ |_) | |_| | (_| | (_| | | | | | (_| |
# |____/ \___|_.__/ \__,_|\__, |\__, |_|_| |_|\__, |
#                         |___/ |___/         |___/

def show_type(rhs, indent=0) -> None:
    """
    Prints all the types of the data structure for debugging.
    :param rhs: The structure to show.
    :param indent: Current indent level for nesting.
    :return: None
    """
    if isinstance(rhs, dict):
        show_dict(rhs, indent)
    elif isinstance(rhs, (list, tuple)):
        show_list(rhs, indent)


def show_dict(rhs, indent=0) -> None:
    """
    Prints the types of all the values in the dictionary.
    :param rhs: The dict to scan.
    :param indent: Current indent level for nesting.
    :return:
    """
    for k, v in rhs.items():
        print(f"{' ' * indent}{k} {type(v)}")
        show_type(v, indent + 4)


def show_list(rhs, indent=0) -> None:
    """
    Prints a list of all the types in the list. Only shows each type once.
    :param rhs: The list to scan.
    :param indent: Current indent level for nesting.
    :return: None
    """
    if len(rhs) > 0:
        found_types = {}
        for i in range(len(rhs)):
            elem_type = type(rhs[i])
            if elem_type not in found_types:
                print(f"{' ' * indent}[{i}] - {elem_type}")
                found_types[elem_type] = i
                show_type(rhs[i], indent + 4)

# ======================================================================================================================
# RANDOM
#  ____                 _
# |  _ \ __ _ _ __   __| | ___  _ __ ___
# | |_) / _` | '_ \ / _` |/ _ \| '_ ` _ \
# |  _ < (_| | | | | (_| | (_) | | | | | |
# |_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_|


def wrap_seed(seed: int):
    """ :return: A numerically wraps the seed if it exceeds the maximum value. """
    # Some system has a maximum value of 32 bits
    return seed & 0xFFFFFFFF
