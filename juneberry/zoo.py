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

import logging
from typing import Tuple

from juneberry.lab import Lab

logger = logging.getLogger(__name__)

"""
Model zoo urls are of the format:
https://<zoo>/models/<model-name>/
|-------------------|------------|
  -- zoo-url --      -- model --

Where the LAST /models/ splits the zoo url from the model name with models in the
zoo-url. If we want to find a model on the zoo, we put the model zoo url together 
with the model name name.

When we store them in the cache they are
<path-to-cache>/model-name.zip

If the model name has slashes in it, then the last part is the part before the zip. So
<path-to-cache>/mode/name.zip
"""


# This module uses two environment/lab variables
# JUNEBERRY_MODEL_ZOO - An url to the model zoo
# JUNEBERRY_CACHE - A location on disk for the cache. Default ~/.juneberry/cache

def _setup_zoo_and_model_name(lab: Lab, model_name: str) -> Tuple[str, str]:
    """
    If a model name contains an URL splits it into a zoo path or model name.
    We assume the word models exists at least once and we split the path
    based on the LAST location of models into the zoo portion and model name.
    If the model name is just the model, use it and the zoo from the lab
    :param lab: The lab object.
    :param model_name: The model name (may have URL) to analyze.
    :return: Zoo base URL, model-name
    """
    # TODO: Find last occurrence of '/models' and split after it
    pass


def _install_from_cache(lab: Lab, model_name: str) -> bool:
    """
    Installs the model from the cache if it exists. Returns true if model
    is installed.
    :return: True if installed from the zoo.
    """
    # Check existence in cache and if not there, return false
    # Make the model path in the models dir - Path.mkdir
    # Unzip the contents into that directory - extract all
    pass


def _download_to_cache(lab: Lab, zoo_url: str, model_name: str) -> bool:
    """
    Download the model from the zoo into the cache.
    :param lab: The lab that has the zoo url and cache path
    :param zoo_url: The url to the zoo or None to use the Lab
    :param model_name: The name of the model to download.
    :return: True if the model was downloaded into the cache.
    """
    # If cache path is none, log warning and return
    # Make url
    #
    # Something like
    #
    # import requests
    # URL = "https://instagram.com/favicon.ico"
    # response = requests.get(URL)
    # open("instagram.ico", "wb").write(response.content)
    pass


def ensure_model(lab: Lab, model_name: str, no_cache: bool = False) -> None:
    """
    This routine installs the model from the cache if available, if not, downloads
    it from the model zoo and installs it.
    :param lab: The lab that contains the cache directory and model zoo url
    :param model_name: The model name to download.
    :param no_cache: Set to true to ignore what is in the cache and download again
    :return:
    """
    # Set up the proper zoo and model names

    # If exists in model directory - done
    # If not in cache or "no_cache" then pull to cache
    # Install from cache
    pass


def prepare_model_for_zoo(model_name: str, model_file: str, staging_zoo_dir: str) -> str:
    """
    Prepares a model and config into an archive for being placed into a zoo for downlaod.
    :param model_name: The name of the model. The config from this directory will be placed in the archive.
    :param model_file: The path to the model file to place directly into the archive.
    :param staging_zoo_dir: A directory that mirrors the deployed "models" directory in which to place the zipped
    model and maps to the "zoo-url" root. The model path will be created inside the directory as appropriate.
    :return: Path to resulting archive
    """
    # Create the model name directory in the staging directory int the staging directory.
    # Open a zip file.
    # -- add the config
    # -- add the model file
    # return the path to the archive file
    pass
