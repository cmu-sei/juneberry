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

"""
Unit test for the trainer base class.
"""

import logging
import time

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
from juneberry.training.trainer import EpochTrainer
import utils


class EpochTrainerHarness(EpochTrainer):
    def __init__(self, lab, model_manager, model_config, dataset_config, log_level):
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        self.setup_calls = []
        self.start_epoch_phase_calls = []
        self.start_epoch_evaluation_calls = []
        self.process_batch_calls = []
        self.update_metrics_calls = []
        self.update_model_calls = []
        self.summarize_metrics_calls = []
        self.checkpoint_calls = []
        self.finalize_results_calls = []
        self.serialize_results_calls = []

        self.expected_metrics = None
        self.expected_results = None

        self.epochs = 3

        self.step = 0

    @utils.log_func
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

    @utils.log_func
    def start_epoch_phase(self, train: bool):
        if train:
            self.start_epoch_phase_calls.append(self.step)
        else:
            self.start_epoch_evaluation_calls.append(self.step)
        self.step += 1
        self.expected_metrics = "DummyTrainMetrics"
        return self.expected_metrics

    @utils.log_func
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

    @utils.log_func
    def update_metrics(self, train, metrics, results) -> None:
        self.update_metrics_calls.append(self.step)
        self.step += 1

        assert self.expected_metrics == metrics
        assert self.expected_results == results

    @utils.log_func
    def update_model(self, results) -> None:
        self.update_model_calls.append(self.step)
        self.step += 1

        assert self.expected_results == results

    @utils.log_func
    def summarize_metrics(self, train: bool, metrics) -> None:
        self.summarize_metrics_calls.append(self.step)
        self.step += 1

    @utils.log_func
    def end_epoch(self) -> None:
        self.checkpoint_calls.append(self.step)
        self.step += 1
        self.epochs -= 1
        if self.epochs == 0:
            self.done = True

    @utils.log_func
    def finalize_results(self):
        self.finalize_results_calls.append(self.step)
        self.step += 1

    # -----------------------------------------------
    # Utility functions I don't know what to do with
    @utils.log_func
    def get_labels(self, targets):
        return []

    @utils.log_func
    def _serialize_results(self):
        # override so we don't actually write to disk
        self.serialize_results_calls.append(self.step)
        self.step += 1


def test_epoch_trainer(tmp_path):
    """
    >> 0 setup
    >> 1 start_epoch_phase
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
    >> 19 start_epoch_phase
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
    >> 37 start_epoch_phase
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
    >> 56 serialize_results_calls
    """

    print("Starting")
    logging.basicConfig(level=logging.INFO)
    lab = Lab(workspace='.', data_root=',')
    dsc = utils.make_basic_dataset_config()
    data_set_config = DatasetConfig.construct(dsc)

    mc = utils.make_basic_model_config()
    model_config = ModelConfig.construct(mc)

    model_manager = jb_fs.ModelManager("foo")
    lab.setup_lab_profile(model_config=model_config)

    trainer = EpochTrainerHarness(lab, model_manager, model_config, data_set_config, log_level=logging.INFO)

    trainer.train_model(None)
    assert trainer.setup_calls == [0]
    assert trainer.start_epoch_phase_calls == [1, 19, 37]

    # Check all the results.  Sequencing is important
    assert trainer.process_batch_calls == [2, 5, 8, 13, 15, 20, 23, 26, 31, 33, 38, 41, 44, 49, 51]
    assert trainer.start_epoch_evaluation_calls == [12, 30, 48]
    assert trainer.update_metrics_calls == [3, 6, 9, 14, 16, 21, 24, 27, 32, 34, 39, 42, 45, 50, 52]
    assert trainer.update_model_calls == [4, 7, 10, 22, 25, 28, 40, 43, 46]
    assert trainer.summarize_metrics_calls == [11, 17, 29, 35, 47, 53]
    assert trainer.checkpoint_calls == [18, 36, 54]
    assert trainer.finalize_results_calls == [55]
    assert trainer.serialize_results_calls == [56]

    trainer.timer.log_metrics()

    print("Done")
