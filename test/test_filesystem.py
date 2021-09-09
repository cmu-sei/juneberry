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

import json
import numpy as np
from pathlib import Path

import juneberry.filesystem as jbfs


def test_model_manager():
    mm = jbfs.ModelManager('TestModel', '1999')
    root = Path('models') / 'TestModel' / '1999'
    plots = root / 'plots'

    assert mm.get_model_name() == 'TestModel'
    assert mm.get_model_dir() == root
    assert mm.get_plots_dir() == plots
    assert mm.get_dryrun_imgs_dir() == root / 'train' / 'dryrun_imgs'
    assert mm.get_pytorch_model_path() == root / 'model.pt'
    assert mm.get_pytorch_model_summary_path() == root / 'model_summary.txt'
    assert mm.get_model_config() == root / 'config.json'
    assert mm.get_model_diagram() == root / 'model_diagram.png'
    assert mm.get_training_out_file() == root / 'train' / 'output.json'
    assert mm.get_training_summary_plot() == root / 'train' / 'output.png'
    assert mm.get_training_log() == root / 'train' / 'log.txt'
    assert mm.get_training_dryrun_log_path() == root / 'train' / 'log_dryrun.txt'


def test_model_manager_clean():
    mm = jbfs.ModelManager('TestModel', '1999')
    root = Path('models') / 'TestModel' / '1999'
    test_dir = root / 'test_dir'
    woot_file = root / 'woot.txt'
    ok_file = test_dir / 'ok.txt'

    if not test_dir.exists():
        test_dir.mkdir(parents=True)
    mm.get_model_config().touch()
    mm.get_pytorch_model_path().touch()
    woot_file.touch()
    ok_file.touch()

    mm.clean(dry_run=True)
    assert mm.get_model_config().exists()
    assert mm.get_pytorch_model_path().exists()
    assert ok_file.exists()
    assert woot_file.exists()

    mm.clean()
    assert mm.get_model_config().exists()
    assert not mm.get_pytorch_model_path().exists()
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
    assert em.get_experiment_log_path() == root / "log_experiment.txt"
    assert em.get_experiment_dryrun_log_path() == root / "log_experiment_dryrun.txt"


def test_experiment_manager_clean():
    em = jbfs.ExperimentManager("millikan_oil_drop")
    root = Path('experiments') / "millikan_oil_drop"

    test_dir = root / 'test_dir'
    woot_file = root / 'woot.txt'
    ok_file = test_dir / 'ok.txt'

    if not test_dir.exists():
        test_dir.mkdir(parents=True)
    em.get_experiment_config().touch()
    em.get_experiment_dryrun_log_path().touch()
    woot_file.touch()
    ok_file.touch()

    em.clean(dry_run=True)
    assert em.get_experiment_config().exists()
    assert em.get_experiment_dryrun_log_path().exists()
    assert ok_file.exists()
    assert woot_file.exists()
    assert test_dir.exists()

    em.clean()
    assert em.get_experiment_config().exists()
    assert not em.get_experiment_dryrun_log_path().exists()
    assert ok_file.exists()
    assert woot_file.exists()
    assert test_dir.exists()

    import shutil
    shutil.rmtree(Path('experiments') / 'millikan_oil_drop')


def test_data_manager():
    data_set_config = {}
    dm = jbfs.DataManager(data_set_config)
    assert dm.version_path == Path()
    dm = jbfs.DataManager(data_set_config, version='009')
    assert dm.version_path == Path('009')

    data_set_config['dataSetPath'] = 'testDataSet'
    dm = jbfs.DataManager(data_set_config)
    assert dm.version_path == Path('testDataSet')
    dm = jbfs.DataManager(data_set_config, version='009')
    assert dm.version_path == Path('testDataSet') / '009'

    category = 'cars'
    assert dm.get_directory_path(category) == Path('testDataSet') / '009' / 'cars'

    image = 'miata.png'
    assert dm.get_file_path(category, image) == Path('testDataSet') / '009' / 'cars' / 'miata.png'

    data_set_config['imageData'] = {}
    data_set_config['imageData']['properties'] = {}
    data_set_config['imageData']['properties']['dimensions'] = '64,64'
    data_set_config['imageData']['properties']['colorspace'] = 'gray'
    dm = jbfs.DataManager(data_set_config, version='009')

    temp_dir = Path.cwd() / 'test' / 'cache' / 'testDataSet' / '009' / '64x64_gray' / 'cars'
    temp_image = temp_dir / 'miata.png'

    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)
    temp_image.touch()

    import shutil
    shutil.rmtree(Path.cwd() / 'test' / 'cache')


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
    results = json.loads(str_results)
    assert results['path'] == 'models'
    assert results['np'] == [1, 2, 3]
