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

import juneberry.pytorch.model_transforms
from pathlib import Path
import torch
import torchvision

from juneberry import filesystem as jbfs


def make_dummy_resent_18(model_mgr):
    model = torchvision.models.resnet18()
    state_dict = model.state_dict()
    half_bias = torch.full(state_dict['fc.bias'].size(), 0.5)
    state_dict['fc.bias'] = half_bias

    if not model_mgr.get_model_dir().exists():
        model_mgr.get_model_dir().mkdir()

    torch.save(state_dict, model_mgr.get_pytorch_model_path())
    return half_bias


def test_load_model_from_url():
    kwargs = {
        "modelURL": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "excludePatterns": ['fc.weight'],
        "strict": False
    }
    model = torchvision.models.resnet18()

    # The weights should be unchanged, so stash off a copy
    fc_weights = model.state_dict()['fc.weight'].clone()

    # Clear out the bias so we can check to see it is set
    state_dict = model.state_dict()
    nan_bias = torch.full(state_dict['fc.bias'].size(), float('nan'))
    state_dict['fc.bias'] = nan_bias
    model.load_state_dict(state_dict)

    # Let the transform do its work
    transform = juneberry.pytorch.model_transforms.LoadModel(**kwargs)
    model = transform(model)

    # All the values should be the same as the original
    assert torch.all(fc_weights.eq(model.state_dict()['fc.weight']))
    assert not torch.any(nan_bias.eq(model.state_dict()['fc.bias']))


def test_load_model_from_model_name():
    model_mgr = jbfs.ModelManager("model_transform_test")
    kwargs = {
        "modelName": "model_transform_test",
        "excludePatterns": ['fc.weight'],
        "strict": False
    }

    # Make the dummy model to load
    dummy_bias = make_dummy_resent_18(model_mgr)

    # Make a new model
    model = torchvision.models.resnet18()

    # The weights should be unchanged, so stash off a copy
    fc_weights = model.state_dict()['fc.weight'].clone()

    # Clear out the bias in this model so we can check to see it is set
    zero_bias = torch.zeros_like(model.state_dict()['fc.bias'])
    state_dict = model.state_dict()
    state_dict['fc.bias'] = zero_bias
    model.load_state_dict(state_dict)

    # Let the transform do its work
    transform = juneberry.pytorch.model_transforms.LoadModel(**kwargs)
    model = transform(model)

    # Check that the new values are as we expected
    assert torch.all(fc_weights.eq(model.state_dict()['fc.weight']))
    assert torch.all(dummy_bias.eq(model.state_dict()['fc.bias']))


def test_include_pattern():
    kwargs = {
        "modelName": "model_transform_test",
        "includePatterns": "ba",
        "strict": False
    }

    transform = juneberry.pytorch.model_transforms.LoadModel(**kwargs)
    fake_keys = ["foo", "bar", "baz"]
    keep_map = transform.filter_keys(fake_keys)
    assert len(keep_map) == 2
    assert 'bar' in keep_map
    assert 'baz' in keep_map

    # Ordering should be the same
    assert list(keep_map.keys())[0] == 'bar'
    assert list(keep_map.keys())[1] == 'baz'


def test_exclude_pattern():
    kwargs = {
        "modelName": "model_transform_test",
        "excludePatterns": "ba",
        "strict": False
    }

    transform = juneberry.pytorch.model_transforms.LoadModel(**kwargs)
    fake_keys = ["foo", "bar", "baz"]
    keep_map = transform.filter_keys(fake_keys)
    assert len(keep_map) == 1
    assert 'foo' in keep_map

    # Ordering should be the same
    assert list(keep_map.keys())[0] == 'foo'


def test_rename_pattern():
    kwargs = {
        "modelName": "model_transform_test",
        "renamePatterns": ['ba', 'ugh'],
        "strict": False
    }

    transform = juneberry.pytorch.model_transforms.LoadModel(**kwargs)
    fake_keys = ["foo", "bar", "baz"]
    keep_map = transform.filter_keys(fake_keys)
    assert len(keep_map) == 3
    assert keep_map['foo'] == 'foo'
    assert keep_map['bar'] == 'ughr'
    assert keep_map['baz'] == 'ughz'

    # Ordering should be the same
    assert list(keep_map.keys())[0] == 'foo'
    assert list(keep_map.keys())[1] == 'bar'
    assert list(keep_map.keys())[2] == 'baz'


def test_log_model_summary():
    model = torchvision.models.resnet18()

    # Let the transform do its work
    transform = juneberry.pytorch.model_transforms.LogModelSummary(imageShape=(3, 224, 224))
    model = transform(model)


def test_save_model_path(tmpdir):
    path = Path(tmpdir) / "junkmodel.pt"
    str_path = str(path.resolve())
    kwargs = {
        "modelPath": str_path,
        "overwrite": True
    }

    model = torchvision.models.resnet18()

    transform = juneberry.pytorch.model_transforms.SaveModel(**kwargs)
    model = transform(model)

    # Now, we should have a file in the spot
    assert path.exists()


def test_save_model_name(tmpdir):
    model_mgr = jbfs.ModelManager("model_transform_test")
    if not model_mgr.get_model_dir().exists():
        model_mgr.get_model_dir().mkdir()

    # If one already exists, delete it
    if model_mgr.get_pytorch_model_path().exists():
        model_mgr.get_pytorch_model_path().unlink()

    kwargs = {
        "modelName": "model_transform_test",
        "overwrite": False
    }

    model = torchvision.models.resnet18()
    transform = juneberry.pytorch.model_transforms.SaveModel(**kwargs)
    model = transform(model)

    # Now, we should have a file in the spot
    assert model_mgr.get_pytorch_model_path().exists()
