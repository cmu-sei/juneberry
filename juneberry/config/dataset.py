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

from collections import namedtuple, defaultdict
import datetime
from enum import Enum
import logging
from pathlib import Path
import random
import sys

from prodict import Prodict, List

import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


# Having them subclass string makes them convert to string by default.
class DataType(str, Enum):
    UNDEFINED = 'unknown'
    IMAGE = 'image'
    TORCHVISION = 'torchvision'
    TABULAR = 'tabular'


class TaskType(str, Enum):
    UNDEFINED = 'undefined'
    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'


class SamplingAlgo(str, Enum):
    NONE = 'none'
    RANDOM_FRACTION = 'random_fraction'
    RANDOM_QUANTITY = 'random_quantity'
    ROUND_ROBIN = 'round_robin'


SamplingConfig = namedtuple('SamplingType', 'algo args randomizer')


class Plugin(Prodict):
    fqcn: str
    kwargs: Prodict


class ImagesSource(Prodict):
    description: str
    url: str
    remove_image_ids: List[int]
    directory: str
    label: int
    sampling_count: int
    sampling_fraction: float


class ImageData(Prodict):
    task_type: str
    sources: List[ImagesSource]


class TabularSource(Prodict):
    root: str
    path: str


class TabularData(Prodict):
    label_index: int
    sources: List[TabularSource]


class TensorFlowData(Prodict):
    name: str
    load_kwargs: Prodict


class TorchvisionData(Prodict):
    fqcn: str
    root: str
    train_kwargs: Prodict
    val_kwargs: Prodict
    eval_kwargs: Prodict


class DataTransforms(Prodict):
    seed: int
    transforms: List[Plugin]


class Sampling(Prodict):
    algorithm: str
    arguments: Prodict


class DatasetConfig(Prodict):
    FORMAT_VERSION = '0.3.0'
    SCHEMA_NAME = 'dataset_schema.json'

    data_transforms: DataTransforms
    data_type: str
    description: str
    file_path: Path
    format_version: str
    image_data: ImageData
    label_names: dict
    num_model_classes: int
    relative_path: Path
    sampling: Sampling
    tabular_data: TabularData
    tensorflow_data: TensorFlowData
    timestamp: str
    torchvision_data: TorchvisionData
    url: str

    def _finish_init(self, relative_path: Path = None, file_path: str = None):
        """
        Initializes a dataset configuration object from a config data structure.
        :param relative_path: Path for "relative" tabular operations.
        :param file_path: Optional - string indicating the dataset config file
        used to construct the object.
        """
        found_errors = 0

        self.relative_path = Path(relative_path) if relative_path is not None else None
        self.file_path = Path(file_path) if file_path is not None else None

        # Attempt to extract some optional fields
        self.is_binary = self.num_model_classes == 2

        # Some checks that are specific to the Image data type
        if self.is_image_type():
            # TODO: Move to schema checks
            # Classification also requires label
            if self.is_classification_task():
                for idx, entry in enumerate(self.image_data.sources):
                    if 'label' not in entry:
                        logger.error(f"imageData source entry {idx} missing required keyword 'label'.")
                        found_errors += 1

        if found_errors > 0:
            self.valid = False
            logger.error(f"Found {found_errors} error(s) in config file. See log for details.")
            sys.exit(-1)

    @staticmethod
    def construct(data: dict, relative_path: Path = None, file_path: str = None):
        """
        Validate and construct a DatasetConfig object.
        :param data: The data to use to construct the object.
        :param relative_path: Used for datasets with relative path data.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: A constructed object.
        """

        # We currently do not have any non-compatible versions
        # conf_utils.require_version(data, DatasetConfig.FORMAT_VERSION, file_path, 'DatasetConfig')

        # Validate with our schema
        if not conf_utils.validate_schema(data, DatasetConfig.SCHEMA_NAME):
            logger.error(f"Validation errors in DatasetConfig from {file_path}. See log. EXITING!")
            sys.exit(-1)

        # Finally, construct the object
        dataset = DatasetConfig.from_dict(data)
        dataset._finish_init(relative_path, file_path)
        return dataset

    @staticmethod
    def load(data_path: str, relative_path: Path = None):
        """
        Load the config from the provided path, validate and construct the config.
        :param data_path: Path to config.
        :param relative_path: Used for datasets with relative path data.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading DATASET CONFIG from {data_path}")
        data = jbfs.load_file(data_path)

        # Validate and construct the model.
        return DatasetConfig.construct(data, relative_path, data_path)

    def to_json(self):
        """ :return: A pure dictionary version suitable for serialization to json."""
        as_dict = conf_utils.prodict_to_dict(self)

        # Strip our some other stuff
        ignore_attrs = ["found_errors", "valid", "relative_path", "file_path"]
        for attr_name in ignore_attrs:
            if attr_name in as_dict:
                del as_dict[attr_name]

        return as_dict

    def save(self, data_path: str) -> None:
        """
        Save the DatasetConfig to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, DatasetConfig.SCHEMA_NAME)

    def retrieve_label_names(self) -> dict:
        """ :return: The label names as a dict of int -> string."""
        # TODO: Deal with str vs label issues
        label_names = {}
        for k, v in self.label_names.items():
            label_names[int(k)] = v
            if int(k) >= self.num_model_classes:
                logger.error(f"Label name entry {k} exceeds num_model_classes {self.num_model_classes}.")
        return label_names

    def update_label_names(self, label_names: dict):
        # NOTE: Prodict requires we store them as strings.
        self.label_names = {str(k): v for k, v in label_names.items()}
        self.num_model_classes = max([int(k) for k in label_names.keys()]) + 1

    def is_image_type(self):
        """ :return: True if this configuration is for an image dataset."""
        return self.data_type == DataType.IMAGE

    def is_tabular_type(self):
        """ :return: True if this configuration is for a tabular dataset."""
        return self.data_type == DataType.TABULAR

    def is_classification_task(self):
        """ :return: True if this configuration is for a classifications task."""
        if self.is_image_type():
            return self.image_data.task_type == TaskType.CLASSIFICATION
        else:
            return False

    def is_object_detection_task(self):
        """ :return: True if this configuration is for an object detection task."""
        if self.is_image_type():
            return self.image_data.task_type == TaskType.OBJECT_DETECTION
        else:
            return False

    def get_image_sources(self):
        """ :return: Image sources. """
        return self.image_data.sources

    def _resolve_tabular_source_path(self, lab, source):
        """
        "root": <OPTIONAL: [ dataroot (default) | workspace | relative ] >
        "path": < glob path within the above defined root >

        :param lab: The Juneberry lab in which to resolve files.
        :param source: The data source stanza.
        :return: A list of resolved paths and the label index for all.
        """
        root = source.get('root', 'dataroot')
        path = str(source['path'])
        paths = []
        if root == 'dataroot':
            paths = list(lab.data_root().glob(path))
        elif root == 'workspace':
            paths = list(lab.workspace().glob(path))
        elif root == 'relative':
            if self.relative_path is None:
                logger.error("No relative path set for relative tabular file loading. EXITING")
                sys.exit(-1)
            paths = list(self.relative_path.glob(path))
        else:
            logger.error(f"Unknown source root '{root}'")

        return paths

    def get_resolved_tabular_source_paths_and_labels(self, lab):
        """
        Generates a set of file, index pairs from the source section of the configuration.
        :param lab: The Juneberry lab in which to resolve files.
        :return: A list of the resolved file paths and the label.
        """
        if not self.is_tabular_type():
            logger.error(f"Trying to resolve tabular source on a non-tabular data set. Type is: "
                         f"{self.data_type}. EXITING.")

        file_list = []
        for source in self.tabular_data.sources:
            file_list.extend(self._resolve_tabular_source_path(lab, source))

        return file_list, self.tabular_data.label_index

    def has_sampling(self):
        """ :return: True if this has a sampling stanza. """
        return self.sampling is not None

    def get_sampling_config(self):
        """
        :return: Sampling algorithm name, sampling algorithm arguments, and a randomizer for sampling
        """
        if self.sampling is None:
            return SamplingConfig(None, None, None)

        sampling_algo = self.sampling['algorithm']
        sampling_args = self.sampling['arguments']

        # Set seed if there is one
        randomizer = None
        if sampling_algo in (SamplingAlgo.RANDOM_FRACTION, SamplingAlgo.RANDOM_QUANTITY, SamplingAlgo.ROUND_ROBIN):
            seed = sampling_args.get('seed', None)
            if seed is None:
                logger.error("Sampler 'randomFraction' and 'randomQuantity' require a 'seed' in 'arguments'")
                sys.exit(-1)
            else:
                logger.info("Setting SAMPLING seed to: " + str(sampling_args['seed']))
                randomizer = random.Random()
                randomizer.seed(sampling_args['seed'])

        return SamplingConfig(sampling_algo, sampling_args, randomizer)


class DatasetConfigBuilder:
    def __init__(self, data_type: DataType, task_type: TaskType, description=''):
        self.warnings = defaultdict(list)
        self.errors = defaultdict(list)

        self.config = DatasetConfig.from_dict({'data_type': data_type.value,
                                               'description': description,
                                               'num_model_classes': 0,
                                               'label_names': {},
                                               'image_data': {'task_type': task_type.value, 'sources': []},
                                               'format_version': DatasetConfig.FORMAT_VERSION,
                                               'timestamp': str(datetime.datetime.now())})

        self._log_error('label_names', 'Label index to name mapping is required', False)
        self._log_error('image_data.sources', 'No sources in config, at least one source required', False)

    def _log_warn(self, key, msg, to_logger=True):
        self.warnings[key].append('WARNING: ' + msg)
        if to_logger:
            logger.warning(msg)

    def _log_error(self, key, msg, to_logger=True):
        self.errors[key].append('ERROR: ' + msg)
        if to_logger:
            logger.error(msg)

    def _clear_config_comments(self, key):
        if not key:
            return
        self.warnings.pop(key, None)
        self.errors.pop(key, None)

    def set_label_mapping(self, label_dict):
        self._clear_config_comments('label_names')
        if not label_dict:
            self.config.label_names = {}
            self.config.num_model_classes = 0
            self._log_error('label_names', 'Label index to name mapping is required')
            return

        self.config.label_names = {str(k): v for k, v in label_dict.items()}
        self.config.num_model_classes = len(label_dict)
        for i in range(len(label_dict)):
            if str(i) not in self.config.label_names:
                self._log_warn('label_names', 'Label indexing is missing index {i}; '
                                              'model configs that reference this dataset will need '
                                              'to include a preprocessor to handle this.')

    def add_source(self, lab, directory, label, *, source_description='', remove_image_ids=None,
                   sampling_count=None, sampling_fraction=None):
        if remove_image_ids is None:
            remove_image_ids = []
        idx = len(self.config.image_data.sources)
        self._clear_config_comments(f'image_data.sources_{idx}')
        if label and self.config.image_data.task_type == TaskType.OBJECT_DETECTION:
            self._log_warn(f'image_data.sources_{idx}', f'')

        new_source = DatasetConfigBuilder.create_source(lab, directory, label, source_description, remove_image_ids,
                                                        sampling_count, sampling_fraction)
        self.config.image_data.sources.append(ImagesSource.from_dict(new_source))
        self._clear_config_comments('image_data.sources')
        return new_source

    @staticmethod
    def create_source(lab, directory, label=0, source_description='', remove_image_ids=None,
                      sampling_count=None, sampling_fraction=None):
        abs_path = lab.data_root() / directory
        if remove_image_ids is None:
            remove_image_ids = []
        if not abs_path.exists():
            logger.error(f'Source directory {str(abs_path)} does not exist. '
                         f'Directory path should be relative to data root. '
                         f'A data root of {str(lab.data_root())} was passed into this '
                         'dataset\'s config builder through the lab.')
        config = {
            'description': source_description,
            'directory': str(directory),
            'label': label,
        }
        if remove_image_ids is not None:
            config['remove_image_ids'] = remove_image_ids
        if sampling_count is not None:
            config['sampling_count'] = sampling_count
        if sampling_fraction is not None:
            config['sampling_fraction'] = sampling_fraction

        return config

    def set_sampling(self, algorithm, arguments=None):
        if algorithm is None:
            algorithm = SamplingAlgo.NONE
        if arguments is None:
            arguments = {}

        # noinspection PyTypeChecker
        if algorithm not in list(SamplingAlgo):
            self._log_error('sampling.algorithm', f'{algorithm} is not currently supported')
            self.config.sampling = Sampling.from_dict({'algorithm': 'none', 'arguments': {}})
            return

        if algorithm != SamplingAlgo.NONE:
            if 'seed' not in arguments:
                self._log_warn('sampling.arguments',
                               'Missing seed field; an integer seed value is recommended for random samplers')

        if algorithm == SamplingAlgo.RANDOM_QUANTITY:
            if 'count' not in arguments:
                self._log_error('sampling.arguments',
                                'Missing count field; specific count of images to provide, can be overridden in source')
        elif algorithm == SamplingAlgo.RANDOM_FRACTION:
            if 'fraction' not in arguments:
                self._log_error('sampling.arguments',
                                'Missing fraction field; decimal fraction of images to provide, can be overridden '
                                'in source')
        elif algorithm == SamplingAlgo.ROUND_ROBIN:
            if 'groups' not in arguments:
                self._log_error('sampling.arguments', 'Missing groups field; number of groups to split dataset into')
            if 'position' not in arguments:
                self._log_error('sampling.arguments',
                                'Missing position field; group to select from result in range [0 - (groups - 1)]')

        self.config.sampling = Sampling.from_dict({'algorithm': algorithm.value, 'arguments': arguments})

    def validate_config(self):
        return conf_utils.validate_schema(self.config, 'dataset_schema.json')

    def to_json(self):
        return self.config.to_json()
