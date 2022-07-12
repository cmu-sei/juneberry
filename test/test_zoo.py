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
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from juneberry.lab import Lab
import juneberry.zoo as zoo
import requests


def setup_tmp_dir(tmp_path: Path):
    ws_path = tmp_path / "workspace"
    ws_path.mkdir(parents=True)
    ws_models = ws_path / 'models'
    ws_models.mkdir()
    os.chdir(ws_path)

    cache_path = tmp_path / "cache"
    cache_path.mkdir(parents=True)
    cache_models = cache_path / 'models'
    cache_models.mkdir()

    return ws_path, cache_path


def make_sample_zip(tmp_path, target_path):
    # Make a sample zip in the path called "sample_model.zip"
    # A tiny config.json
    # An a model.pt file which is itself a zip of a file sample.txt with hello world.

    # Make a sample model file
    with ZipFile(tmp_path / "model.pt", "w") as zip_file:
        zip_file.writestr("sample.txt", "Hello world.")

    # Now make the zip
    if not target_path.exists():
        target_path.mkdir(parents=True)

    with ZipFile(target_path / 'sample_model.zip', "w") as zip_file:
        zip_file.writestr("config.json", "{'data' : 'foo'}")
        zip_file.write(tmp_path / "model.pt", "model.pt")


def test_model_name_parsing():
    lab = Lab(model_zoo="test_zoo")
    zoo_url, model_name = zoo._setup_zoo_and_model_name(lab, "test_model")
    assert zoo_url == "test_zoo"
    assert model_name == "test_model"

    zoo_url, model_name = zoo._setup_zoo_and_model_name(lab, "http://foo.com/models/other_test_model")
    assert zoo_url == "http://foo.com/models"
    assert model_name == "other_test_model"

    zoo_url, model_name = zoo._setup_zoo_and_model_name(lab, "https://bar.com/models/other_test_model")
    assert zoo_url == "https://bar.com/models"
    assert model_name == "other_test_model"


def test_download_to_cache(tmp_path, monkeypatch):
    def mock_get(url):
        # NOTE: We return as a binary string because the content is a zip
        return SimpleNamespace(status_code=200, content=b"Hello world")

    # Make out get provide something stupid
    monkeypatch.setattr(requests, "get", mock_get)
    lab = Lab(cache=tmp_path / "cache")

    zoo._download_to_cache(lab, "zoo", "my_model")

    # Now, in the models subdirectory of the cache directory we should have the model
    cache_file_path = tmp_path / 'cache' / 'models' / 'my_model.zip'
    assert cache_file_path.exists()


def test_install_from_cache(tmp_path):
    # Set up the workspace so we can install it there
    ws_path, cache_path = setup_tmp_dir(tmp_path)

    # Make the sample model right in the cache
    make_sample_zip(tmp_path, cache_path / 'models')

    lab = Lab(workspace=ws_path, cache=cache_path)
    zoo._install_from_cache(lab, 'sample_model')

    # Now, we should have a config and model in the models dir
    model_dir = ws_path / "models" / "sample_model"
    assert (model_dir / "config.json").exists()
    assert (model_dir / "model.pt").exists()


def test_ensure_model(tmp_path, monkeypatch):
    # Set up the workspace so we can install it there
    ws_path, cache_path = setup_tmp_dir(tmp_path)

    # Make the sample model right in the tmp)path and we'll "download" from there
    make_sample_zip(tmp_path, tmp_path)

    def mock_get(url):
        if url == "https://my.zoo.com/models/model_path/sample_model.zip":
            # NOTE: We return as a binary string because the content is a zip
            with open(tmp_path / 'sample_model.zip', "rb") as zip_file:
                data = zip_file.read()
                return SimpleNamespace(status_code=200, content=data)
        else:
            print(f"*********************** {url}")
            return SimpleNamespace(status_code=404, content=None)

    monkeypatch.setattr(requests, "get", mock_get)

    lab = Lab(workspace=ws_path, cache=cache_path, model_zoo="https://my.zoo.com/models")
    zoo.ensure_model(lab, 'model_path/sample_model')

    # Now it should be in the model dir in the "model_path" subdir
    model_dir = ws_path / "models" / "model_path" / "sample_model"
    assert (model_dir / "config.json").exists()
    assert (model_dir / "model.pt").exists()


def test_prepare_model(tmp_path):
    # Set up the workspace so we can install it there
    ws_path, cache_path = setup_tmp_dir(tmp_path)

    # Make up a fake model in the workspace
    dummy_model_path = ws_path / 'models' / 'foo' / 'staged_model'
    dummy_model_path.mkdir(parents=True, exist_ok=True)

    with open(dummy_model_path / "config.json", "w") as out_file:
        out_file.write("{'data' : 'foo'}")
    with ZipFile(dummy_model_path / "model.pt", "w") as zip_file:
        zip_file.writestr("sample.txt", "Hello world.")
    with open(dummy_model_path / "model.h5", "w") as out_file:
        out_file.write("A tensorflow model")

    staging_path = tmp_path / "staging"
    staging_path.mkdir(parents=True)

    zip_path = zoo.prepare_model_for_zoo("foo/staged_model", str(staging_path))

    # Now, we need to make sure that it was constructed. Open the zip and see that
    # everything is there
    with ZipFile(zip_path) as zip_file:
        assert "config.json" in zip_file.namelist()
        assert "model.pt" in zip_file.namelist()
        assert "model.h5" in zip_file.namelist()
