#! /usr/bin/env python3

"""
Unit test for the trainer base class.
"""

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import inspect
import logging
import time

from juneberry.trainer import EpochTrainer

from juneberry.config.dataset import DatasetConfig
from juneberry.config.training import TrainingConfig

import test_data_set
import test_training_config

import functools


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


class EpochTrainerHarness(EpochTrainer):
    def __init__(self, training_config, data_set_config, **kw_args):
        super().__init__(training_config, data_set_config, **kw_args)

        self.setup_calls = []
        self.dry_run_calls = []
        self.start_epoch_training_calls = []
        self.start_epoch_evaluation_calls = []
        self.process_batch_calls = []
        self.update_metrics_calls = []
        self.update_model_calls = []
        self.summarize_metrics_calls = []
        self.checkpoint_calls = []
        self.finalize_results_calls = []
        self.close_calls = []

        self.expected_metrics = None
        self.expected_results = None

        self.epochs = 3

        self.step = 0

    @log_func
    def setup(self):
        self.setup_calls.append(self.step)
        self.step += 1

        # We need to mock up the "data generator"
        # Since it is just an iterable, we can just do that
        self.training_iterable = [
            [[1, 2], [0, 1]],
            [[2, 4], [1, 0]],
            [[3, 6], [0, 1]],
        ]
        self.evaluation_iterable = [
            [[10, 20], [1, 0]],
            [[11, 22], [0, 1]]
        ]

    @log_func
    def dry_run(self) -> None:
        self.dry_run_calls.append(self.step)
        self.step += 1

    @log_func
    def start_epoch_phase(self, train: bool):
        if train:
            self.start_epoch_training_calls.append(self.step)
        else:
            self.start_epoch_evaluation_calls.append(self.step)
        self.step += 1
        self.expected_metrics = "DummyTrainMetrics"
        return self.expected_metrics

    @log_func
    def process_batch(self, train, data, targets):
        time.sleep(0.1)

        self.process_batch_calls.append(self.step)
        self.step += 1

        # Two pieces of data
        # Two is twice one
        # If first entry odd, then 0,1 else 1,0 for targets
        assert len(data) == 2
        assert data[0] != 0
        assert data[1] == data[0] * 2
        if data[0] % 2 == 0:
            assert targets == [1, 0]
        else:
            assert targets == [0, 1]

        self.expected_results = data[0] * 3
        return self.expected_results

    @log_func
    def update_metrics(self, train, metrics, results) -> None:
        self.update_metrics_calls.append(self.step)
        self.step += 1

        assert self.expected_metrics == metrics
        assert self.expected_results == results

    @log_func
    def update_model(self, results) -> None:
        self.update_model_calls.append(self.step)
        self.step += 1

        assert self.expected_results == results

    @log_func
    def summarize_metrics(self, train: bool, metrics) -> None:
        self.summarize_metrics_calls.append(self.step)
        self.step += 1

    @log_func
    def end_epoch(self, elapsed_secs: float) -> None:
        self.checkpoint_calls.append(self.step)
        self.step += 1
        self.epochs -= 1
        if self.epochs == 0:
            self.done = True

    @log_func
    def finalize_results(self):
        self.finalize_results_calls.append(self.step)
        self.step += 1

    @log_func
    def close(self) -> None:
        self.close_calls.append(self.step)
        self.step += 1

    # -----------------------------------------------
    # Utility functions I don't know what to do with
    @log_func
    def get_labels(self, targets):
        return []


def test_epoch_trainer():
    """
    >> 0 setup
    >> 1 start_epoch_training
    >> 2 process_batch
    >> 3 update_metrics
    >> 4 update_model
    >> 5 process_batch
    >> 6 update_metrics
    >> 7 update_model
    >> 8 process_batch
    >> 9 update_metrics
    >> 10 update_model
    >> 11 summarize_metrics
    >> 12 start_epoch_evaluation
    >> 13 process_batch
    >> 14 update_metrics
    >> 15 process_batch
    >> 16 update_metrics
    >> 17 summarize_metrics
    >> 18 end_epoch
    >> 19 start_epoch_training
    >> 20 process_batch
    >> 21 update_metrics
    >> 22 update_model
    >> 23 process_batch
    >> 24 update_metrics
    >> 25 update_model
    >> 26 process_batch
    >> 27 update_metrics
    >> 28 update_model
    >> 29 summarize_metrics
    >> 30 start_epoch_evaluation
    >> 31 process_batch
    >> 32 update_metrics
    >> 33 process_batch
    >> 34 update_metrics
    >> 35 summarize_metrics
    >> 36 end_epoch
    >> 37 start_epoch_training
    >> 38 process_batch
    >> 39 update_metrics
    >> 40 update_model
    >> 41 process_batch
    >> 42 update_metrics
    >> 43 update_model
    >> 44 process_batch
    >> 45 update_metrics
    >> 46 update_model
    >> 47 summarize_metrics
    >> 48 start_epoch_evaluation
    >> 49 process_batch
    >> 50 update_metrics
    >> 51 process_batch
    >> 52 update_metrics
    >> 53 summarize_metrics
    >> 54 end_epoch
    >> 55 finalize_results
    """

    print("Starting")
    logging.basicConfig(level=logging.INFO)
    dsc = test_data_set.make_basic_config()
    data_set_config = DatasetConfig(dsc)
    tc = test_training_config.make_basic_config()
    training_config = TrainingConfig("simple_model", tc)

    trainer = EpochTrainerHarness(training_config, data_set_config)

    trainer.train_model()
    assert trainer.setup_calls == [0]
    assert trainer.start_epoch_training_calls == [1, 19, 37]

    # Check all the results.  Sequencing is important
    assert trainer.process_batch_calls == [2, 5, 8, 13, 15, 20, 23, 26, 31, 33, 38, 41, 44, 49, 51]
    assert trainer.start_epoch_evaluation_calls == [12, 30, 48]
    assert trainer.update_metrics_calls == [3, 6, 9, 14, 16, 21, 24, 27, 32, 34, 39, 42, 45, 50, 52]
    assert trainer.update_model_calls == [4, 7, 10, 22, 25, 28, 40, 43, 46]
    assert trainer.summarize_metrics_calls == [11, 17, 29, 35, 47, 53]
    assert trainer.checkpoint_calls == [18, 36, 54]
    assert trainer.finalize_results_calls == [55]
    assert trainer.close_calls == [56]

    trainer.timer.log_metrics()

    print("Done")


def test_dry_run():
    logging.basicConfig(level=logging.INFO)
    dsc = test_data_set.make_basic_config()
    data_set_config = DatasetConfig(dsc)
    tc = test_training_config.make_basic_config()
    training_config = TrainingConfig("simple_model", tc)

    trainer = EpochTrainerHarness(training_config, data_set_config, dry_run=True)

    trainer.train_model()
    assert trainer.setup_calls == [0]
    assert trainer.dry_run_calls == [1]

    print("Done")
