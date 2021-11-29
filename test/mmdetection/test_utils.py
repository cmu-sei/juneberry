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

import juneberry.mmdetection.utils as mmd_utils


def test_tupleize():
    sample = {
        'array': [1, 2, 3],
        'tuple': (4, 5, 6),
        'nested': {
            'arrays': [[10, 11], (12, 13), [14, 15]],
            'dicts': ({'foo': [100, 101]}, {'bar': (200, 201)})
        }
    }

    result = mmd_utils.tupleize(sample)

    assert isinstance(result['array'], tuple)
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['nested']['arrays'][0], tuple)
    assert isinstance(result['nested']['arrays'][1], tuple)
    assert isinstance(result['nested']['dicts'][0]['foo'], tuple)
    assert isinstance(result['nested']['dicts'][1]['bar'], tuple)


def test_add_train_stages():
    changes = [
        {
            "name": "RandomFlip",
            "stage": {"type": "BeforeRandomFlip", "count": 1138}
        },
        {
            "name": "Resize",
            "stage": {"type": "BeforeResize", "count": 1234}
        },
        {
            "name": "Pad",
            "stage": {"type": "AfterPad", "size": [1, 2, 3, 4]},
            "mode": "after",
            "tupleize": True
        },
        {
            "name": "Normalize",
            "mode": "delete"
        },
        {
            "name": "Pad",
            "stage": {"type": "ReplacePad", "size_divisor": 123456789},
            "mode": "replace"
        },
        {
            "name": "Collect",
            "stage": {"foo": 1138},
            "mode": "update"
        }
    ]

    train_in = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                       (1333, 768), (1333, 800)],
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

    train_ans = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='BeforeResize', count=1234),
        dict(
            type='Resize',
            img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                       (1333, 768), (1333, 800)],
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='BeforeRandomFlip', count=1138),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='ReplacePad', size_divisor=123456789),
        dict(type='AfterPad', size=(1, 2, 3, 4)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], foo=1138)
    ]

    mmd_utils.add_stages(train_in, changes)
    assert train_in == train_ans


def test_add_val_stages():
    changes = [
        {
            "name": "RandomFlip",
            "stage": {"type": "BeforeRandomFlip", "count": 1138}
        },
        {
            "name": "Resize",
            "stage": {"type": "BeforeResize", "count": 1234}
        },
        {
            "name": "Pad",
            "stage": {"type": "AfterPad", "size": [1, 2, 3, 4]},
            "mode": "after",
            "tupleize": True
        },
        {
            "name": "Normalize",
            "mode": "delete"
        },
        {
            "name": "Pad",
            "stage": {"type": "ReplacePad", "size_divisor": 123456789},
            "mode": "replace"
        },
        {
            "name": "Collect",
            "stage": {"foo": 1138},
            "mode": "update"
        },
        {
            "name": "MultiScaleFlipAug",
            "stage": {"img_scale": (224,224)},
            "mode": "update",
            "tupleize": True

        }

    ]

    val_in = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]

    val_ans = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(224, 224),
            flip=False,
            transforms=[
                dict(type='BeforeResize', count=1234),
                dict(type='Resize', keep_ratio=True),
                dict(type='BeforeRandomFlip', count=1138),
                dict(type='RandomFlip'),
                dict(type='ReplacePad', size_divisor=123456789),
                dict(type='AfterPad', size=(1, 2, 3, 4)),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'], foo=1138)
            ])
    ]

    mmd_utils.add_stages(val_in, changes)
    assert val_in == val_ans
