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

from contextlib import contextmanager
import json
import os
from pathlib import Path


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
        'args': {'num_classes': 2},
        'module': 'juneberry-example-workspace.architectures.pytorch.sample_tabular_nn.BinaryModel'
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
    'model_architecture': {'args': {'num_classes': 3},
                           'module': 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'},
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


def setup_test_workspace(tmp_path) -> None:
    """
    Creates a test workspace *structure* within tmp_path.
    :param tmp_path: Path to where to setup the temporary workspace.
    :return: Nothing.
    """
    ws_path = Path(tmp_path)
    model_dir_path = ws_path / "models"
    tbs_dir_path = model_dir_path / "tabular_binary_sample"
    tbs_dir_path.mkdir(parents=True, exist_ok=True)

    dt_conf_path = model_dir_path / "text_detect" / "dt2" / "ut"
    dt_conf_path.mkdir(parents=True, exist_ok=True)

    data_sets_path = ws_path / "data_sets"
    data_sets_path.mkdir(exist_ok=True)


def make_tabular_workspace(tmp_path) -> None:
    """
    Creates a sample model config and dataset config for the tabular model.
    :param tmp_path: Path to workspace directory.
    :return: Nothing.
    """
    ws_path = Path(tmp_path)
    model_conf_path = ws_path / "models" / "tabular_binary_sample" / "config.json"
    with open(str(model_conf_path), "w") as out_file:
        json.dump(tabular_model_config, out_file)

    ds_conf_path = ws_path / "data_sets" / "train_data_config.json"
    with open(str(ds_conf_path), "w") as out_file:
        json.dump(tabular_dataset_config, out_file)


def make_dt2_workspace(tmp_path) -> None:
    """
    Creates a sample model config and dataset config for the detectron2 text detection model.
    :param tmp_path: Path to workspace directory.
    :return: Nothing.
    """
    ws_path = Path(tmp_path)

    model_conf_path = ws_path / "models" / "text_detect" / "dt2" / "ut" / "config.json"
    with open(str(model_conf_path), "w") as out_file:
        json.dump(text_detect_dt2_config, out_file)

    ds_conf_path = ws_path / "data_sets" / "text_detect_val.json"
    with open(str(ds_conf_path), "w") as out_file:
        json.dump(text_detect_dataset_config, out_file)
