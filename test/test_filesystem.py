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

import json
import hjson
import numpy as np
from pathlib import Path

import juneberry.filesystem as jbfs
from juneberry.platform import PlatformDefinitions


def test_eval_dir():
    mm = jbfs.ModelManager('TestModel')
    root = Path('models') / 'TestModel'
    eval_dir_root = root / 'eval' / 'TestDataset'

    eval_dir_mgr = mm.get_eval_dir_mgr('TestDataset')
    eval_dir_mgr.setup()

    assert eval_dir_mgr.get_dir() == eval_dir_root
    assert eval_dir_mgr.get_scratch_dir() == eval_dir_root / 'scratch'
    assert eval_dir_mgr.get_dryrun_imgs_dir() == str(eval_dir_root / 'dryrun_imgs')
    assert eval_dir_mgr.get_platform_config() == str(eval_dir_root / "platform_config.json")
    assert eval_dir_mgr.get_manifest_path() == str(eval_dir_root / "eval_manifest.json")
    assert eval_dir_mgr.get_detections_path() == str(eval_dir_root / "detections.json")
    assert eval_dir_mgr.get_detections_anno_path() == str(eval_dir_root / "detections_anno.json")
    assert eval_dir_mgr.get_log_path() == str(eval_dir_root / "log.txt")
    assert eval_dir_mgr.get_log_path("TestTool") == str(eval_dir_root / "log_TestTool.txt")
    assert eval_dir_mgr.get_dryrun_log_path() == str(eval_dir_root / "log_dryrun.txt")
    assert eval_dir_mgr.get_dryrun_log_path("TestTool") == str(eval_dir_root / "log_dryrun_TestTool.txt")
    assert eval_dir_mgr.get_metrics_path() == str(eval_dir_root / "metrics.json")
    assert eval_dir_mgr.get_predictions_path() == str(eval_dir_root / "predictions.json")
    assert eval_dir_mgr.get_sample_detections_dir() == str(eval_dir_root / "sample_detections")


class DummyPlatformDefinitions(PlatformDefinitions):
    def get_model_filename(self) -> str:
        return "foo.txt"

    def get_config_suffix(self) -> str:
        return ".bak"

    def has_platform_config(self) -> bool:
        return True


def test_model_manager():
    mm = jbfs.ModelManager('TestModel')
    root = Path('models') / 'TestModel'
    plots = root / 'plots'

    assert mm.get_model_name() == 'TestModel'
    assert mm.get_model_dir() == root
    assert mm.get_plots_dir() == plots
    assert mm.get_dryrun_imgs_dir() == root / 'train' / 'dryrun_imgs'
    assert mm.get_model_summary_path() == root / 'model_summary.txt'
    assert mm.get_model_config() == root / 'config.json'
    assert mm.get_model_diagram() == root / 'model_diagram.png'
    assert mm.get_training_out_file() == root / 'train' / 'output.json'
    assert mm.get_training_summary_plot() == root / 'train' / 'output.png'
    assert mm.get_training_log() == root / 'train' / 'log.txt'
    assert mm.get_training_dryrun_log_path() == root / 'train' / 'log_dryrun.txt'

    assert mm.get_model_path(DummyPlatformDefinitions()) == root / 'foo.txt'


def test_model_manager_clean():
    mm = jbfs.ModelManager('TestModel')
    root = Path('models') / 'TestModel' / '1999'
    test_dir = root / 'test_dir'
    woot_file = root / 'woot.txt'
    ok_file = test_dir / 'ok.txt'

    if not test_dir.exists():
        test_dir.mkdir(parents=True)
    mm.get_model_config().touch()
    mm.get_model_path(DummyPlatformDefinitions()).touch()
    woot_file.touch()
    ok_file.touch()

    mm.clean(DummyPlatformDefinitions(), dry_run=True)
    assert mm.get_model_config().exists()
    assert mm.get_model_path(DummyPlatformDefinitions()).exists()
    assert ok_file.exists()
    assert woot_file.exists()

    mm.clean(DummyPlatformDefinitions())
    assert mm.get_model_config().exists()
    assert not mm.get_model_path(DummyPlatformDefinitions()).exists()
    assert ok_file.exists()
    assert woot_file.exists()

    import shutil
    shutil.rmtree(Path('models') / 'TestModel')


def test_experiment_manager():
    em = jbfs.ExperimentManager("millikan_oil_drop")
    root = Path('experiments') / "millikan_oil_drop"

    assert em.get_experiment_name() == "millikan_oil_drop"
    assert em._get_experiment_dir() == root
    assert em.get_experiment_config() == root / 'config.json'
    assert em.get_log_path() == root / "logs" / "log_experiment.txt"
    assert em.get_dryrun_log_path() == root / "logs" / "log_experiment_dryrun.txt"


def test_experiment_manager_clean():
    em = jbfs.ExperimentManager("millikan_oil_drop")
    root = Path('experiments') / "millikan_oil_drop"

    test_dir = root / 'test_dir'
    woot_file = root / 'woot.txt'
    ok_file = test_dir / 'ok.txt'

    if not test_dir.exists():
        test_dir.mkdir(parents=True)
    em.get_experiment_config().touch()
    em.get_dryrun_log_path().touch()
    woot_file.touch()
    ok_file.touch()

    em.clean(dry_run=True)
    assert em.get_experiment_config().exists()
    assert em.get_dryrun_log_path().exists()
    assert ok_file.exists()
    assert woot_file.exists()
    assert test_dir.exists()

    em.clean()
    assert em.get_experiment_config().exists()
    assert not em.get_dryrun_log_path().exists()
    assert ok_file.exists()
    assert woot_file.exists()
    assert test_dir.exists()

    import shutil
    shutil.rmtree(Path('experiments') / 'millikan_oil_drop')


def test_hash_function(tmp_path):
    # This is the same as "shasum -a 256" on "echo 'Hello World' > foo.txt"

    file_path = Path(tmp_path) / "hash_test.txt"
    with open(file_path, 'w') as tmp_file:
        tmp_file.write("Hello World\n")

    out_str = jbfs.generate_file_hash(file_path)
    assert "d2a84f4b8b650937ec8f73cd8be2c74add5a911ba64df27458ed8229da804a26" == out_str


def test_json_cleaner():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    str_results = json.dumps(data, indent=4, default=jbfs.json_cleaner)
    # TODO: Should this be routed through the jbfs load chokepoint?
    results = hjson.loads(str_results)
    assert results['path'] == 'models'
    assert results['np'] == [1, 2, 3]


def test_open_file():
    cwd = Path.cwd()
    file = cwd / "test.json"
    with open(file, "w") as f:
        f.write("test")
    with jbfs.open_file(file, "r") as f:
        assert f.read() == "test"
    file.unlink()


def test_save_json():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()

    path = cwd / "test_save_json.json"
    jbfs.save_json(data, path)
    assert path.exists()
    path.unlink(True)


def test_save_hjson():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()

    path = cwd / "test_save_hjson.hjson"
    jbfs.save_hjson(data, path)
    assert path.exists()
    path.unlink(True)


def test_save_yaml():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()

    path = cwd / "test_save_yaml.yaml"
    jbfs.save_yaml(data, path)
    assert path.exists()
    path.unlink(True)

    path = cwd / "test_save_yaml.yml"
    jbfs.save_yaml(data, path)
    assert path.exists()
    path.unlink(True)


def test_save_toml():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()

    path = cwd / "test_save_toml.toml"
    jbfs.save_toml(data, path)
    assert path.exists()
    path.unlink(True)

    path = cwd / "test_save_toml.tml"
    jbfs.save_toml(data, path)
    assert path.exists()
    path.unlink(True)

    path = cwd / "test_save_toml.toml.gz"
    jbfs.save_toml(data, path)
    assert path.exists()
    path.unlink(True)


def test_save_file():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()
    # JSON case
    path = cwd / "test_save_file.json"
    path.unlink(missing_ok=True)
    jbfs.save_file(data, path)
    assert path.exists()
    path.unlink()
    assert not path.exists()
    # HJSON case
    path = cwd / "test_save_file.hjson"
    path.unlink(missing_ok=True)
    jbfs.save_file(data, path)
    assert path.exists()
    path.unlink()
    assert not path.exists()
    # GZIP case
    path = cwd / "test_save_file.json.gzip"
    path.unlink(missing_ok=True)
    jbfs.save_file(data, path)
    assert path.exists()
    path.unlink()
    assert not path.exists()
    # YAML case
    path = cwd / "test_save_file.yaml"
    path.unlink(missing_ok=True)
    jbfs.save_file(data, path)
    assert path.exists()
    path.unlink()
    assert not path.exists()
    # TOML case
    path = cwd / "test_save_file.toml"
    path.unlink(missing_ok=True)
    jbfs.save_file(data, path)
    assert path.exists()
    path.unlink()
    assert not path.exists()


def test_load_file():
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()
    # JSON case
    path = cwd / "test_save_file.json"
    jbfs.save_file(data, path)
    loaded_data = jbfs.load_file(path)
    assert np.array_equal(loaded_data['np'], data['np'])
    assert loaded_data['path'] == str(data['path'])
    path.unlink()
    # HJSON case
    path = cwd / "test_save_file.hjson"
    jbfs.save_file(data, path)
    loaded_data = jbfs.load_file(path)
    assert np.array_equal(loaded_data['np'], data['np'])
    assert loaded_data['path'] == str(data['path'])
    path.unlink()
    # GZIP case
    path = cwd / "test_save_file.json.gz"
    jbfs.save_file(data, path)
    loaded_data = jbfs.load_file(path)
    assert np.array_equal(loaded_data['np'], data['np'])
    assert loaded_data['path'] == str(data['path'])
    path.unlink()
    # YAML case
    path = cwd / "test_save_file.yaml"
    jbfs.save_file(data, path)
    loaded_data = jbfs.load_file(path)
    assert np.array_equal(loaded_data['np'], data['np'])
    assert loaded_data['path'] == data['path']
    path.unlink()
    # TOML case
    path = cwd / "test_save_file.toml"
    jbfs.save_file(data, path)
    loaded_data = jbfs.load_file(path)
    assert np.array_equal(list(map(int, loaded_data['np'])), data['np'])
    # TOML converty the object PosixPath('models') into the string "PosixPath('models')"
    # assert loaded_data['path'] == str(data['path']) 
    path.unlink()
    # TOML.GZ case
    path = cwd / "test_save_file.toml.gz"
    jbfs.save_file(data, path)
    loaded_data = jbfs.load_file(path)
    assert np.array_equal(list(map(int, loaded_data['np'])), data['np'])
    # TOML converty the object PosixPath('models') into the string "PosixPath('models')"
    # assert loaded_data['path'] == str(data['path']) 
    path.unlink()


def test_load_file_look_for_gz():
    # Load file should look for a fil.ext.gz file if fil.ext cannot be found
    data = {"np": np.array([1, 2, 3]), "path": Path('models')}
    cwd = Path.cwd()

    # saved JSON.GZIP, load JSON
    # load_file should recognize that there is no 'test_save_file.json' and
    # load 'test_save_file.json.gzip' instead
    gz_path = cwd / "test_save_file.json.gz"
    json_path = cwd / "test_save_file.json"
    jbfs.save_file(data, gz_path)
    assert not json_path.exists()
    loaded_data = jbfs.load_file(json_path)
    assert np.array_equal(loaded_data['np'], data['np'])
    assert loaded_data['path'] == str(data['path'])
    gz_path.unlink()
