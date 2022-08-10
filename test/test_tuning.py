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
import datetime
import json
from pathlib import Path
from unittest import TestCase

import pytest
from ray.tune.sample import Categorical as CategoricalSampler
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from juneberry.config.tuning import TuningConfig
import juneberry.config.tuning_output as tuning_output
from juneberry.script_tools.tuning_sprout import TuningSprout
from juneberry.tuning.tuner import Tuner
from test_sprouts import TestTuningSprout


def make_basic_tuning_config():
    """
    This function returns a dictionary representing the contents of a very basic Juneberry hyperparameter
    tuning configuration file.
    """
    return {
        "description": "A simple tuning config for unit tests.",
        "num_samples": 1,
        "scheduler": {
            "fqcn": "ray.tune.schedulers.AsyncHyperBandScheduler",
            "kwargs": {}
        },
        "search_algorithm": {
            "fqcn": "ray.tune.suggest.basic_variant.BasicVariantGenerator",
            "kwargs": {}
        },
        "search_space": [
            {
                "hyperparameter_name": "batch_size",
                "fqcn": "ray.tune.choice",
                "kwargs": {
                    "categories": [10, 15]
                }
            }
        ],
        "trial_resources": {
            "cpu": 1,
            "gpu": 0
        },
        "tuning_parameters": {
            "checkpoint_interval": 1,
            "metric": "loss",
            "mode": "min",
            "scope": "last"
        }
    }


def verify_basic_tuning_config(tuning_config: TuningConfig):
    """
    This function examines a TuningConfig object and checks if its attributes match the properties
    described in JSON content defined above in make_basic_tuning_config().
    :param tuning_config: The Juneberry TuningConfig object to examine.
    :return: Nothing.
    """
    # Number of samples
    assert tuning_config.num_samples == 1

    # Scheduler
    assert tuning_config.scheduler.fqcn == "ray.tune.schedulers.AsyncHyperBandScheduler"
    assert tuning_config.scheduler.kwargs == {}

    # Search algorithm
    assert tuning_config.search_algorithm.fqcn == "ray.tune.suggest.basic_variant.BasicVariantGenerator"
    assert tuning_config.search_algorithm.kwargs == {}

    # Search space
    assert len(tuning_config.search_space) == 1
    assert tuning_config.search_space[0].hyperparameter_name == "batch_size"
    assert tuning_config.search_space[0].fqcn == "ray.tune.choice"
    assert tuning_config.search_space[0].kwargs == {"categories": [10, 15]}

    # Trial resources
    assert tuning_config.trial_resources.cpu == 1
    assert tuning_config.trial_resources.gpu == 0

    # Tuning parameters
    assert tuning_config.tuning_parameters.checkpoint_interval == 1
    assert tuning_config.tuning_parameters.metric == "loss"
    assert tuning_config.tuning_parameters.mode == "min"
    assert tuning_config.tuning_parameters.scope == "last"


class TestTuningConfig(TestCase):
    """
    This group of tests exercise various aspects related to Juneberry TuningConfig objects.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, tmp_path):
        """
        The purpose of this method is to make the pytest tmp_path fixture available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path
        self.tuning_config_path = Path(self.tmp_path, "tuning_config.json")

        config = make_basic_tuning_config()
        with open(self.tuning_config_path, 'w') as out_file:
            json.dump(config, out_file, indent=4)

    def test_basic_tuning_config_construct(self):
        """
        This test exercises construction of a TuningConfig object using a dictionary of data.
        """
        # Construct the TuningConfig object using the "contents of the JSON file".
        tuning_config = TuningConfig.construct(make_basic_tuning_config())

        # Verify the attributes in the TuningConfig.
        verify_basic_tuning_config(tuning_config)

    def test_basic_tuning_config_load(self):
        """
        This test exercises the loading of a TuningConfig from a JSON file.
        """
        # Load the TuningConfig data from the JSON file.
        tuning_config = TuningConfig.load(str(self.tuning_config_path))

        # Verify the attributes in the TuningConfig.
        verify_basic_tuning_config(tuning_config)

    def test_tuning_config_default_values(self):
        """
        This test exercises the setting of default values in the TuningConfig when
        the parameters aren't present in the input data.
        """
        # Start with the basic TuningConfig data.
        config = make_basic_tuning_config()

        # Remove some fields from the data and confirm that they're gone.
        del_keys = ["tuning_parameters", "trial_resources"]
        for key in del_keys:
            config.pop(key)
            assert key not in config.keys()

        # Build a TuningConfig object using adjusted data.
        tuning_config = TuningConfig.construct(config, str(self.tuning_config_path))

        # Confirm the default tuning_parameters have been set.
        assert tuning_config.tuning_parameters.checkpoint_interval == 0
        assert tuning_config.tuning_parameters.metric == "loss"
        assert tuning_config.tuning_parameters.mode == "min"
        assert tuning_config.tuning_parameters.scope == "last"

        # Confirm the default trial_resources have been set.
        assert tuning_config.trial_resources.cpu == 1
        assert tuning_config.trial_resources.gpu == 0


class TestTuningOutput(TestCase):
    """
    This group of tests exercise various aspects of the output generated during a tuning run.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        """
        The purpose of this method is to establish a TuningOutputBuilder to exercise the ability
        to produce tuning output data.
        """
        self.tuning_output_builder = tuning_output.TuningOutputBuilder()
        self.output = self.tuning_output_builder.output

    def test_output_init(self):
        """
        The purpose of this test is to verify that the tuning output has been initialized
        correctly.
        """
        # Check the attributes of the TuningOutput object.
        assert self.output.format_version == tuning_output.TuningOutput.FORMAT_VERSION
        assert type(self.output.options) == tuning_output.Options
        assert type(self.output.times) == tuning_output.Times
        assert type(self.output.results) == tuning_output.Results
        assert self.output.results.trial_results == []

    def test_builder_set_tuning_options(self):
        """
        This test exercises the ability to set the tuning options via the TuningOutputBuilder.
        """
        # Some strings to use for testing.
        model_name = "test_model"
        tuning_config_name = "test_tuning_config.json"

        # Set the tuning options via the TuningOutputBuilder.
        self.tuning_output_builder.set_tuning_options(model_name=model_name, tuning_config=tuning_config_name)

        # Confirm the options have been set.
        assert self.output.options.model_name == model_name
        assert self.output.options.tuning_config == tuning_config_name

    def test_builder_set_times(self):
        """
        This test exercises the ability to set the tuning Times via the TuningOutputBuilder.
        """
        # Create some timestamps.
        start_time = datetime.datetime.now().replace(microsecond=0)
        end_time = datetime.datetime.now().replace(microsecond=0)

        # Set the timing information in the tuning output.
        self.tuning_output_builder.set_times(start_time=start_time, end_time=end_time)

        # Verify the start time, end time, and duration have been set properly in the tuning output.
        assert self.output.times.start_time == start_time.isoformat()
        assert self.output.times.end_time == end_time.isoformat()
        assert self.output.times.duration == (end_time - start_time).total_seconds()

    def test_builder_append_trial_result(self):
        """
        This test exercises the ability to append trial data to the Results section of the tuning output.
        """
        # Create some data to feed into the TuningOutputBuilder.
        directory = "test_dirname"
        params = {"batch_size": 10}
        trial_data = [
            {"trial_id": "abc_123", "loss": 10, "accuracy": 0.1, "lr": 0.01},
            {"trial_id": "abc_123", "loss": 5, "accuracy": 0.2, "lr": 0.001}
        ]

        # Use the sample data to append a trial result to the tuning output.
        self.tuning_output_builder.append_trial_result(directory=directory, params=params, trial_data=trial_data)

        # Verify the trial result has been set properly.
        assert self.output.results.trial_results == [
            {
                "directory": "test_dirname",
                "id": "abc_123",
                "num_iterations": 2,
                "params": {"batch_size": 10},
                "result_data": {
                    "trial_id": ['abc_123', 'abc_123'],
                    "loss": [10, 5],
                    "accuracy": [0.1, 0.2],
                    "lr": [0.01, 0.001]
                }
            }
        ]


class TestTuner(TestCase):
    """
    This group of tests exercise various aspects related to performing hyperparameter tuning using a
    Juneberry Tuner.
    """

    def test_simple_tuner_setup(self):
        """
        This test exercises basic functionality of the Juneberry Tuner.
        """
        # Initialize the Tuner.
        tuner = Tuner()

        # Confirm the default values for all Tuner attributes.
        assert tuner.trainer_factory is None
        assert tuner.sprout is None
        assert tuner.tuning_config is None
        assert tuner.search_space is None
        assert tuner.search_algo is None
        assert tuner.scheduler is None
        assert tuner.best_result is None
        assert type(tuner.output_builder) == tuning_output.TuningOutputBuilder
        assert type(tuner.output) == tuning_output.TuningOutput
        assert tuner.tuning_start_time is None
        assert tuner.tuning_end_time is None

        # Construct a TuningConfig and associate it with the Tuner.
        tuner.tuning_config = TuningConfig.construct(make_basic_tuning_config())

        # Construct a TuningSprout, associate it with the Tuner, and add data to the Sprout.
        tuner.sprout = TuningSprout()
        tuner.sprout.grow_from_args(TestTuningSprout.build_tuning_namespace())

        # Now perform a tuning dryrun using the Tuner.
        tuner.tune(dryrun=True)

        # Confirm the model name and tuning config name were set correctly in the tuning output.
        assert tuner.output.options.model_name == "test_model"
        assert tuner.output.options.tuning_config == "test_tuning_config.json"

        # Confirm the Tuner's search space was set up correctly.
        assert tuner.search_space is not None
        assert 'batch_size' in tuner.search_space.keys()
        assert type(tuner.search_space['batch_size']) == CategoricalSampler

        # Confirm the Tuner's scheduler was set up correctly.
        assert type(tuner.scheduler) is AsyncHyperBandScheduler

        # Confirm the Tuner's search algorithm was set up correctly.
        assert type(tuner.search_algo) is BasicVariantGenerator
