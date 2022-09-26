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
import logging
from pathlib import Path
import re
from typing import Tuple
from zipfile import ZipFile

import requests

from juneberry.config.hashes import Hashes
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
from juneberry.onnx.utils import ONNXPlatformDefinitions
from juneberry.pytorch.utils import PyTorchPlatformDefinitions
from juneberry.tensorflow.utils import TensorFlowPlatformDefinitions

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

def cache_models_dir(lab: Lab) -> Path:
    """
    Returns the directory to the cache from the lab object
    :param lab: The lab object
    :return: Path object to the models directory in the cache
    """
    return lab.cache / 'models'


def cache_model_zip_path(lab: Lab, model_name: str) -> Path:
    """
    Provides a path to the model zip file in the cache.
    :param lab: The lab object
    :param model_name: The model name
    :return: Path to the zip file in the models directory in the cache.
    """
    return cache_models_dir(lab) / (model_name + ".zip")


def ensure_cache_dir(lab: Lab, model_name) -> None:
    """
    Makes sure that we have all the directories we need in the cache to download the model zip.
    :param lab: The lab
    :param model_name: The model name for the zip file.
    :return: None
    """
    cache_dir = cache_models_dir(lab) / Path(model_name).parent
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)


def _setup_zoo_and_model_name(lab: Lab, model_name: str) -> Tuple[str, str]:
    """
    If a model name contains a URL, split it into a zoo path or model name.
    We assume the word models exists at least once and we split the path
    based on the LAST location of models into the zoo portion and model name.
    If the model name is just the model, use it and the zoo from the lab
    :param lab: The lab object.
    :param model_name: The model name (may have URL) to analyze.
    :return: Zoo base URL, model name
    """
    if model_name.startswith("http") or model_name.startswith("https"):
        m = re.match("(.*)/models/(.*)", model_name)
        if m:
            zoo_url = f"{m.group(1)}/models"
            model_name = m.group(2)
            return zoo_url, model_name
        else:
            raise RuntimeError(f"Given url for model, failed to find 'models' part. {model_name}")

    # If we got this far, then the model name is just the model
    return lab.model_zoo, model_name


def _download_to_cache(lab: Lab, zoo_url: str, model_name: str) -> None:
    """
    Download the model from the zoo into the cache.
    :param lab: The lab that has the cache path
    :param zoo_url: The url to the zoo or None to use the Lab
    :param model_name: The name of the model to download.
    :return: None
    """
    zip_url = f"{zoo_url}/{model_name}.zip"
    response = requests.get(zip_url)
    cache_zip_path = cache_model_zip_path(lab, model_name)
    if response.status_code == 200:
        ensure_cache_dir(lab, model_name)
        with open(cache_zip_path, "wb") as zip_file:
            zip_file.write(response.content)
    else:
        raise RuntimeError(f"Failed to download item from model zoo. item={zip_url}, status={response.status_code}")


def _install_from_cache(lab: Lab, model_name: str) -> bool:
    """
    Installs the model from the cache if it exists. Returns true if model
    is installed.
    :return: True if installed from the zoo.
    """
    cache_zip_path = cache_model_zip_path(lab, model_name)

    # Check existence in cache and if not there, return false
    if not cache_zip_path.exists():
        return False

    # Make the model path in the models dir - Path.mkdir
    model_mgr = lab.model_manager(model_name)
    model_mgr.model_dir_path.mkdir(parents=True, exist_ok=True)

    # The contents are in.
    with ZipFile(cache_zip_path) as myzip:
        myzip.extractall(model_mgr.model_dir_path)


def ensure_model(lab: Lab, model_name: str, no_cache: bool = False) -> None:
    """
    This routine installs the model from the cache if available, if not, downloads
    it from the model zoo and installs it.
    :param lab: The lab that contains the cache directory and model zoo url
    :param model_name: The model name to download.
    :param no_cache: Set to true to ignore what is in the cache and download again
    :return: None
    """
    # Set up the proper zoo and model names
    zoo_url, model_name = _setup_zoo_and_model_name(lab, model_name)

    # If the model config exists, we are done
    model_mgr = lab.model_manager(model_name)
    if model_mgr.get_model_config().exists():
        return

    if zoo_url is None:
        logger.info("No juneberry zoo specified, cannot download model.")
        return

    # By this point we are going to pull things and put into the cache.
    # The cache dir needs to exist.
    if lab.cache is None:
        logger.error(f"Cache directory not defined. Please set the JUNEBERRY_CACHE value or use --cache switch.")
        raise RuntimeError("No juneberry cache defined, can't download model.")

    # If not in cache or "no_cache" then pull to cache
    cache_file_path = cache_model_zip_path(lab, model_name)
    if no_cache or not cache_file_path.exists():
        _download_to_cache(lab, zoo_url, model_name)

    # Install from cache
    _install_from_cache(lab, model_name)


def check_allow_load_model(model_manager, summary_hash_fn) -> bool:
    """
    Checks to see if the model should be installed. If a hashes config file exists and
    it has a model_archiecture key and that matchs the archiecture, then we are good to
    go.
    :param lab: The lab that contains the cache directory and model zoo url
    :param model_name: The name of the model
    :param summary_hash_fn: A function to call that will return the appropriate hash
    value of the model summary. This will only be called if the a hash value exists to
    compare agaist.
    :return:
    """
    hashes_path = Path(model_manager.get_hashes_config())
    if not hashes_path.exists():
        # If no hashes config file exists just return.
        logger.debug("No hashes.json file found, approving model load.")
        return True

    hashes = Hashes.load(hashes_path)
    if hashes.model_architecture is None:
        # No model architecture hash exists, retrn
        logger.debug("No model_archirecture found in hashes.json, approving model load.")
        return True

    # Okay, now compare
    if summary_hash_fn() == hashes.model_architecture:
        logger.info("Model architecture hash found and matches architecture allow load.")
        return True
    else:
        logger.info("Model architecture hash found and DOES NOT matches architecture, preventing load.")
        return False


def prepare_model_for_zoo(model_name: str, staging_zoo_dir: str, onnx: bool = True) -> str:
    """
    Prepares a model and config into an archive for being placed into a zoo for download.
    :param model_name: The name of the model. The config from this directory will be placed in the archive.
    :param staging_zoo_dir: A directory that mirrors the deployed "models" directory in which to place the zipped
    model and maps to the "zoo-url" root. The model path will be created inside the directory as appropriate.
    :param onnx: If true, add onnx file if it exists
    :return: Path to resulting archive
    """
    # Create the model name directory in the staging directory int the staging directory.
    model_zip_path = Path(staging_zoo_dir) / (model_name + ".zip")

    # Make the directory
    model_parent_dir = model_zip_path.parent
    if not model_parent_dir.exists():
        model_parent_dir.mkdir(parents=True)

    # We assume that the workspace is the cwd.
    model_mgr = jb_fs.ModelManager(model_name)

    with ZipFile(model_zip_path, "w") as zip_file:
        zip_file.write(model_mgr.get_model_config(), model_mgr.get_model_config_filename())

        # TODO: This is fragile in that we won't zip just anything. We need some better way
        #  to do this. Of course, the user can just zip it themselves, so this is just a convenience.
        path = model_mgr.get_model_path(PyTorchPlatformDefinitions())
        if path.exists():
            zip_file.write(path, PyTorchPlatformDefinitions().get_model_filename())

        path = model_mgr.get_model_path(TensorFlowPlatformDefinitions())
        if path.exists():
            zip_file.write(path, TensorFlowPlatformDefinitions().get_model_filename())

        if onnx:
            path = model_mgr.get_model_path(ONNXPlatformDefinitions())
            if path.exists():
                zip_file.write(path, ONNXPlatformDefinitions().get_model_filename())

        # Add the hash file if it exists
        hashes_path = model_mgr.get_hashes_config()
        if hashes_path.exists():
            zip_file.write(hashes_path, model_mgr.get_hashes_config_filename())
        else:
            # If we have a latest hashes, then use it.
            latest_hashes_path = model_mgr.get_latest_hashes_config()
            if latest_hashes_path.exists():
                zip_file.write(latest_hashes_path, model_mgr.get_hashes_config_filename())

    return str(model_zip_path.absolute())


def update_hashes_file(hashes_path, model_architecture_hash: str = None):
    if hashes_path.exists():
        hashes = Hashes.load(hashes_path)
    else:
        hashes = Hashes()

    if model_architecture_hash is not None:
        hashes.model_architecture = model_architecture_hash
    hashes.save(hashes_path)

def update_hashes_after_training(model_mgr, model_architecture_hash: str = None):
    # If we have an existing hash file, update it.
    # Always update 'latest' as it is what we use when package a model for the zoo.
    hashes_path = model_mgr.get_hashes_config()
    if hashes_path.exists():
        update_hashes_file(hashes_path, model_architecture_hash)
    update_hashes_file(model_mgr.get_latest_hashes_config(), model_architecture_hash)


def package_model():
    parser = argparse.ArgumentParser(description="A simple utility to package an archive. "
                                                 "The CWD MUST be the workspace.")
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("staging_dir", type=str, help="Path to a directory like 'models' in which to create the zip.")
    args = parser.parse_args()
    path = prepare_model_for_zoo(args.model_name, args.staging_dir)
    print(f"Model stored in zip at: {path}")


if __name__ == "__main__":
    package_model()
