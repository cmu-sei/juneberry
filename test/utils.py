#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM22-0856
#
# ======================================================================================================================

from contextlib import contextmanager
import functools
import inspect
import os
from pathlib import Path

from juneberry.config.model import ModelConfig


@contextmanager
def set_directory(path: Path):
    """
    A context manager that:
    1) Captures the current working directory
    2) Sets the current working directory to 'path' on __enter__
    3) Reverts the current working directory to the one captured in step 1 on __exit__
    :param path: The desired path to be used inside the context
    :return: None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


tabular_model_config = {
    'batch_size': 1024,
    'description': 'Sample config for unit testing',
    'epochs': 100,
    'evaluator': {
        'fqcn': 'juneberry.dummy.evaluator'
    },
    'evaluation_transforms': [
        {
            'fqcn': 'juneberry.transforms.tabular.RemoveColumns',
            'kwargs': {'indexes': [1, 3]}
        }
    ],
    'format_version': '0.2.0',
    'lab_profile': {'max_gpus': 1},
    'model_architecture': {
        'fqcn': 'juneberry-example-workspace.architectures.pytorch.sample_tabular_nn.BinaryModel',
        'kwargs': {'num_classes': 2}
    },
    'platform': 'pytorch',
    'pytorch': {
        'deterministic': True,
        'loss_fn': 'torch.nn.BCELoss',
        'optimizer_args': {'lr': 0.1},
        'optimizer_fn': 'torch.optim.SGD'
    },
    'seed': 4210592948,
    'timestamp': '2021-03-01T10:00:00',
    'trainer': {
        'fqcn': 'juneberry.dummy.trainer'
    },
    'training_dataset_config_path': 'data_sets/train_data_config.json',
    'training_transforms': [
        {
            'fqcn': 'juneberry.transforms.tabular.RemoveColumns',
            'kwargs': {'indexes': [3, 1]}
        }
    ],
    'validation': {
        'algorithm': 'random_fraction',
        'arguments': {'fraction': 0.2, 'seed': 3554237221}
    }
}

text_detect_dt2_config = {
    'batch_size': 4,
    'description': 'Unit test for an object detector on the TextDetection data '
                   'set.',
    'detectron2': {'overrides': ['SOLVER.BASE_LR',
                                 0.0025,
                                 'SOLVER.REFERENCE_WORLD_SIZE',
                                 1]},
    'epochs': 1,
    'evaluation_metrics': [{'fqcn': 'juneberry.metrics.metrics.Coco',
                            'kwargs': {'iou_threshold': 0.5,
                                       'max_det': 100,
                                       'tqdm': False}}],
    'evaluation_metrics_formatter': {'fqcn': 'juneberry.metrics.format.DefaultCocoFormatter',
                                     'kwargs': {}},
    'evaluation_transforms': [],
    'evaluator': {'fqcn': 'juneberry.detectron2.evaluator.Evaluator'},
    'format_version': '0.2.0',
    'model_architecture': {
        'fqcn': 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
        'kwargs': {'num_classes': 3}
    },
    'platform': 'detectron2',
    'pytorch': {'deterministic': True},
    'seed': 4210592948,
    'timestamp': '2021-03-03T10:00:00',
    'trainer': {'fqcn': 'juneberry.detectron2.trainer.Detectron2Trainer'},
    'training_dataset_config_path': 'data_sets/text_detect_val.json',
    'training_transforms': [],
    'validation': {'algorithm': 'random_fraction',
                   'arguments': {'fraction': 0.2, 'seed': 3554237221}}
}

tabular_dataset_config = {
    'data_root': 'relative',
    'data_type': 'tabular',
    'format_version': '0.2.0',
    'label_names': {
        '0': 'outer',
        '1': 'inner'
    },
    'num_model_classes': 2,
    'sampling': {'algorithm': 'none', 'arguments': {}},
    'tabular_data': {
        'label_index': 4,
        'sources': [
            {'path': 'train_data.csv', 'root': 'relative'}
        ]
    }
}

text_detect_dataset_config = {
    'data_type': 'image',
    'description': 'Juneberry metadata files created using detectron2 text '
                   'detection validation set.',
    'format_version': '0.2.0',
    'image_data': {
        'sources': [
            {'directory': 'detectron2-text-detection/val/coco_annotations.json'}
        ],
        'task_type': 'object_detection'
    },
    'label_names': {'0': 'HINDI', '1': 'ENGLISH', '2': 'OTHER'},
    'num_model_classes': 3,
    'sampling': {'algorithm': 'none', 'arguments': {}},
    'timestamp': '2021-03-05T10:00:00',
    'url': 'https://cocodataset.org/#download'
}

training_output = {
    'format_version': "0.2.0",
    'options': {
        'batch_size': 0,
        'epochs': 0,
        'model_architecture': {
            "fqcn": "test_arch",
            "kwargs": {}
        },
        'model_name': "TBD",
        'seed': 12345,
        'training_dataset_config_path': "TBD"
    },
    'results': {
        'accuracy': [],
        'loss': [],
        'val_accuracy': [],
        'val_loss': [],

    }
}


def make_basic_model_config(add_transforms: bool = False) -> dict:
    mc = {
        "batch_size": 16,
        "training_dataset_config_path": "path/to/data/set",
        "epochs": 50,
        'evaluator': {
            'fqcn': 'juneberry.dummy.evaluator'
        },
        "format_version": ModelConfig.FORMAT_VERSION,
        "platform": "pytorch",
        "model_architecture": {
            "fqcn": "sample.module",
            "kwargs": {"num_classes": 1000}
        },
        "seed": 1234,
        "timestamp": "optional ISO time stamp for when this was generated generated",
        'trainer': {
            'fqcn': 'juneberry.dummy.evaluator'
        },
        "validation": {
            "algorithm": "random_fraction",
            "arguments": {
                "seed": 1234,
                "fraction": 0.5
            }
        }
    }

    if add_transforms:
        transform_dict = {
            "training_transforms": [
                {
                    "fqcn": "my.fqg",
                    "kwargs": {"arg1": "hello"}
                }
            ],
            "evaluation_transforms": [
                {
                    "fqcn": "my.fqg",
                    "kwargs": {"arg1": "hello"}
                }
            ]
        }
        mc.update(transform_dict)

    return mc


def make_basic_dataset_config(image_data=True, classification=True, torchvision=False):
    config = {
        "num_model_classes": 4,
        "description": "Unit test",
        "timestamp": "never",
        "format_version": "3.2.0",
        "label_names": {"0": "frodo", "1": "sam"},
    }

    if image_data:
        config['data_type'] = 'image'
        if classification:
            config['image_data'] = {
                "task_type": "classification",
                "sources": [{"directory": "some/path", "label": 0}]
            }
        else:
            config['image_data'] = {
                "task_type": "object_detection",
                "sources": [{"directory": "some/path"}]
            }
    elif torchvision:

        kwargs = {
            "size": 2,
            "image_size": (1, 2, 2),
            "num_classes": 4,
        }

        config['data_type'] = 'torchvision'
        config['torchvision_data'] = {
            "eval_kwargs": kwargs,
            "fqcn": "torchvision.datasets.FakeData",  # A fake dataset that returns randomly generated PIL images.
            "root": "",
            "train_kwargs": kwargs,
            "val_kwargs": kwargs
        }

        config['torchvision_data']['task_type'] = 'classification' if classification else 'object_detection'

    else:
        config['data_type'] = 'tabular'
        config['tabular_data'] = {
            "sources": [{"path": "some/path"}],
            "label_index": 0
        }

    return config


def make_basic_experiment_config():
    return {
        "description": "simple description",
        "models": [
            {
                "name": "tabular_binary_sample",
                "tests": [
                    {
                        "tag": "pyt50",
                        "dataset_path": "data_sets/train_data_config.json",
                        "classify": 3
                    }
                ]
            }
        ],
        "reports": [
            {
                "description": "basic description",
                "fqcn": 'juneberry.reporting.roc.ROCPlot',
                "kwargs": {
                    "output_filename": "sample_roc_1.png",
                    "plot_title": "Sample ROC Plot"
                },
                "tests": [
                    {
                        "tag": "pyt50",
                        "classes": "0"
                    }
                ],
            }
        ],
        "tuning": [
            {
                "model": "",
                "tuning_config": ""
            }

        ],
        "format_version": "1.5.0"
    }
    # NOTE: We provide the formatVersion manually to force an update of the unit test when
    # the version changes.


def get_fn_name(fn):
    for k, v in inspect.getmembers(fn):
        if k == "__name__":
            return v
    return "Unknown"


log_step = 0


def log_func(func):
    func_name = get_fn_name(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global log_step
        # Use this to get a list of all calls in order
        # print(f">> {log_step} {func_name}")
        log_step += 1
        return func(*args, **kwargs)

    return wrapper
