#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn 
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow 
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

"""
General utilities.
"""

import numpy as np
from pathlib import Path
import re


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
