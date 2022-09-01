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
Common utilities used by config parsers and generators.
"""

import logging
from pathlib import Path
import pkgutil
import sys

import hjson
import jsonschema
import pkg_resources
from prodict import Prodict

import juneberry.filesystem as jb_fs
from juneberry.utils import none_stripper
import juneberry.version_system as jb_vs

logger = logging.getLogger(__name__)


def require_version(data, min_version, file_path: str, module_str: str):
    data_version = jb_vs.VersionNumber(data['format_version'])
    min_version = jb_vs.VersionNumber(min_version)

    if data_version < min_version:
        logger.error(f"{module_str} at {file_path} has version {data_version} and we require {min_version}. EXITING.")
        sys.exit(-1)


def require_tags(label, data: dict, tags: list) -> int:
    """
    Simple function to check to see if the required tags are in the data dictionary and if not,
    log an error.
    :param label: A label to use during logging.
    :param data: The data dictionary.
    :param tags: The tags that are required in the dictionary.
    :return: The number of missing tags or errors.
    """
    error_count = 0
    for tag in tags:
        if tag not in data:
            logger.error(f"Didn't find required tag. tag='{tag}', structure='{label}'.")
            error_count += 1

    return error_count


def validate_schema(data, schema_name, die_on_error=False):
    # Load the schema.
    # TODO: Should this be routed through the jb_fs load chokepoint?
    schema = hjson.loads(pkgutil.get_data('juneberry', f"schemas/{schema_name}"))

    # Set up a reference resolver to simplify how Juneberry schema files reference each other.
    resource = pkg_resources.resource_filename('juneberry', f"schemas/{schema_name}")
    schema_path = f"file:{resource}"
    resolver = jsonschema.RefResolver(schema_path, schema)

    # While we are trying to use the latest, jsonschema seems to only have 7.
    validator = jsonschema.Draft7Validator(schema, resolver=resolver)
    error_count = 0
    for error in validator.iter_errors(data):
        logger.error(f"{error.message} at {list(error.path)}")
        error_count += 1

    if error_count:
        logger.error(f"Found {error_count} errors validating against schema={schema_name}")
        if die_on_error:
            type_name = type(data).__name__
            logger.error(f"Validation errors in {type_name}. See log. EXITING!")
            sys.exit(-1)

        return False

    return True


def prodict_to_dict(data: Prodict):
    """
    Converts the prodict data to a normal dict.
    :param data: The prodict data to convert.
    :return: A normal (non-prodict) dictionary.
    """
    as_dict = data.to_dict(exclude_none=True, exclude_none_in_lists=True, is_recursive=True)
    as_dict = none_stripper(as_dict)
    return as_dict


def validate_and_save_json(json_data: dict, data_path: str, schema_name: str) -> None:
    """
    Used to validate and export a json version of the file.
    :param json_data: A json suitable data structure usually acquired by "to_json"
    :param data_path: Where to save the file. Format will be determined by suffix.
    :param schema_name: The name of the schema to use for validation, None to skip validation
    :return: None
    """
    data_path = Path(data_path)

    # TODO: Add support in filesystem for different serializers - yaml, toml
    if data_path.suffix != ".json":
        logger.error(f"We currently only support saving configs to json. {data_path}")
        sys.exit(-1)

    # We have to validate BEFORE we rekey
    if schema_name is not None:
        validate_schema(json_data, schema_name, die_on_error=True)

    # Use the filesystem to figure out how to save it
    jb_fs.save_json(json_data, data_path)
