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
from datetime import datetime
import logging
import sys

from prodict import List, Prodict

import juneberry.config.util as jb_config_utils
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class Options(Prodict):
    model_name: str
    tuning_config: str


class TrialResult(Prodict):
    directory: str
    id: str
    num_iterations: int
    params: Prodict
    result_data: Prodict


class Results(Prodict):
    best_trial_id: str
    best_trial_params: Prodict
    trial_results: List[TrialResult]


class Times(Prodict):
    duration: float
    end_time: str
    start_time: str


class TuningOutput(Prodict):
    FORMAT_VERSION = '0.1.0'
    SCHEMA_NAME = 'tuning_output_schema.json'

    format_version: str
    options: Options
    results: Results
    times: Times

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Validate and construct an object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed object.
        """
        # Validate with our schema
        jb_config_utils.require_version(data, TuningOutput.FORMAT_VERSION, file_path, 'TuningOutput')
        if not jb_config_utils.validate_schema(data, TuningOutput.SCHEMA_NAME):
            logger.error(f"Validation errors in TuningOutput from {file_path}. See log. Exiting.")
            sys.exit(-1)

        return TuningOutput.from_dict(data)

    @staticmethod
    def load(data_path: str):
        """
        Loads the config from the provided path, validates, and constructs the object.
        :param data_path: Path to config.
        :return: Loaded, validated, and constructed object.
        """
        logger.info(f"Loading TUNING OUTPUT from {data_path}")
        data = jb_fs.load_file(data_path)

        # Validate and construct the model.
        return TuningOutput.construct(data)

    def save(self, data_path: str) -> None:
        """
        Save the TuningOutput to the specified resource path.
        :param data_path: The path to the resource.
        :return: Nothing.
        """
        jb_config_utils.validate_and_save_json(self.to_json(), data_path, TuningOutput.SCHEMA_NAME)

    def to_json(self):
        return jb_config_utils.prodict_to_dict(self)


class TuningOutputBuilder:
    """ A helper class used to aid in the construction of the training output. """

    def __init__(self):
        self.output = TuningOutput()

        self.output.format_version = TuningOutput.FORMAT_VERSION
        self.output.options = Options()
        self.output.times = Times()
        self.output.results = Results()
        self.output.results.trial_results = []

    def set_tuning_options(self, model_name: str = None, tuning_config: str = None) -> None:
        """
        Sets the tuning options to the desired values.
        :param model_name: The name of the model in the 'models' directory being tuned.
        :param tuning_config: The name of the tuning config in the workspace that was used to tune the model.
        :return: Nothing.
        """
        if model_name is not None:
            self.output.options.model_name = model_name

        if tuning_config is not None:
            self.output.options.tuning_config = tuning_config

    def set_times(self, start_time: datetime, end_time: datetime) -> None:
        """
        Sets the appropriate start and stop times in the training output.
        :param start_time: The time training started.
        :param end_time: The time training ended.
        :return: Nothing.
        """
        self.output.times.start_time = start_time.isoformat()
        self.output.times.end_time = end_time.isoformat()
        self.output.times.duration = (end_time - start_time).total_seconds()

    def append_trial_result(self, directory: str, params: dict, trial_data: list) -> None:
        """
        Appends a Trial result to the list of Trial results in the tuning output.
        :param directory: The directory name for the trial result.
        :param params: The dictionary describing the hyperparameter values selected for this Trial.
        :param trial_data: The JSON Lines data retrieved from the Trial's result.json file.
        :return: Nothing.
        """
        cur_result = TrialResult()

        cur_result.directory = directory
        cur_result.id = trial_data[0]["trial_id"]
        cur_result.num_iterations = len(trial_data)
        cur_result.params = params
        cur_result.result_data = self._process_results(trial_data)

        self.output.results.trial_results.append(cur_result)

    @staticmethod
    def _process_results(results: list) -> dict:
        """
        Converts a Trial's JSON Lines data into the desired format for the tuning output file.
        :param results: The Trial's JSON Lines data.
        :return: A dictionary that would be appropriate to insert into the result_data attribute of
        a TrialResult.
        """
        # Start with an empty dictionary.
        result_dict = {}

        # Loop through all keys in the JSON Lines data and create and initialize an empty list in
        # the result dictionary for each key.
        for k, v in results[0].items():
            result_dict[k] = []

        # Now loop through all the JSON objects in the JSON Lines data and append the value for
        # each key into the appropriate list.
        for result in results:
            for k, v in result.items():
                result_dict[k].append(v)

        return result_dict

    def to_dict(self) -> dict:
        """ :return: Returns a pure version of the data structure as a dict. """
        return jb_config_utils.prodict_to_dict(self.output)

    def to_json(self) -> dict:
        """ :return: A pure dictionary version suitable for serialization to CURRENT JSON. """
        as_dict = jb_config_utils.prodict_to_dict(self.output)

        return as_dict

    def save(self, data_path: str) -> None:
        """
        Saves the training output to the specified path.
        :param data_path: The path to save to.
        :return: None
        """
        self.output.save(data_path)
