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

import logging
import random
import sys
import torch

from torch.utils import data

from juneberry.config.dataset import DataType, DatasetConfig, TaskType
from juneberry.lab import Lab
from juneberry.pytorch.image_dataset import ImageDataset
from juneberry.pytorch.tabular_dataset import TabularDataset
import juneberry.pytorch.utils as pyt_utils

from juneberry.config.dataset import SamplingConfig, TorchvisionData
from juneberry.config.model import ModelConfig, SplittingAlgo, SplittingConfig
import juneberry.data as jb_data
import juneberry.loader as jbloader
from juneberry.transform_manager import TransformManager

logger = logging.getLogger(__name__)


# ======================================================================================================================
#  ____        _        _                    _
# |  _ \  __ _| |_ __ _| |    ___   __ _  __| | ___ _ __ ___
# | | | |/ _` | __/ _` | |   / _ \ / _` |/ _` |/ _ \ '__/ __|
# | |_| | (_| | || (_| | |__| (_) | (_| | (_| |  __/ |  \__ \
# |____/ \__,_|\__\__,_|_____\___/ \__,_|\__,_|\___|_|  |___/


def make_transform_manager(model_cfg: ModelConfig, ds_cfg: DatasetConfig, set_size: int,
                           opt_args: dict, eval: bool = False):
    """
    Constructs the appropriate transform manager for the this data.
    :param model_cfg: The model config.
    :param ds_cfg: The datasett config.
    :param set_size: The size of the data set.
    :param opt_args: Optional args to pass into the construction of the plugin.
    :param eval: Are we in train (False) or eval mode (True).
    :return: Transform manager.
    """
    # Convenience to call the base on with our custom stage transform
    return jb_data.make_transform_manager(model_cfg, ds_cfg, set_size, opt_args,
                                          pyt_utils.PyTorchStagedTransform, eval)


def make_training_data_loaders(lab, ds_cfg, model_cfg, data_lst, split_lst, *,
                               no_paging=False, collate_fn=None, sampler_args=None):
    """
    Creates the appropriate data loaders from the training and validation data sets.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param ds_cfg: The data set config that describes the data.
    :param model_cfg: A Juneberry ModelConfig object that may contain transforms and validation options.
    :param data_lst: The data list to load. Pre-shuffled.
    :param split_lst: The list of data to load that was split from the main dataset.
    :param no_paging: Set to true to read all the data at once. Good for small data sets or large memory.
    :param collate_fn: Function that controls how samples are collated into batches.
    :param sampler_args: Tuple of integers (world_size, rank) if a sampler is to be used; None for no sampler
    :return: Returns the training and validation data loaders.
    """
    opt_args = {'path_label_list': list(data_lst)}
    logger.info("Constructing TRAINING data loader.")
    data_loader = make_data_loader(lab, ds_cfg, data_lst,
                                   make_transform_manager(model_cfg, ds_cfg, len(data_lst), opt_args, False),
                                   model_cfg.batch_size, no_paging=no_paging, collate_fn=collate_fn,
                                   sampler_args=sampler_args)

    logger.info("Constructing VALIDATION data loader.")
    opt_args = {'path_label_list': list(split_lst)}
    split_loader = make_data_loader(lab, ds_cfg, split_lst,
                                    make_transform_manager(model_cfg, ds_cfg, len(split_lst), opt_args, True),
                                    model_cfg.batch_size, no_paging=no_paging, collate_fn=collate_fn)

    return data_loader, split_loader


def make_eval_data_loader(lab, dataset_config, model_config, data_lst, *,
                          no_paging=False, collate_fn=None, sampler_args=None):
    """
    Constructs the appropriate data loader for evaluation
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The data set config that describes the data.
    :param model_config: A Juneberry ModelConfig object that may contain transforms and validation options.
    :param data_lst: The data list to load. Pre-shuffled.
    :param no_paging: Set to true to read all the data at once. Good for small data sets or large memory.
    :param collate_fn: Function that controls how samples are collated into batches.
    :param sampler_args: Tuple of integers (world_size, rank) if a sampler is to be used; None for no sampler
    :return: Returns the data loader.
    """
    # TODO: Should we use collate and sampler?
    opt_args = {'path_label_list': list(data_lst)}
    logger.info("Constructing data loader from EVALUATION data set using prediction transforms.")
    return make_data_loader(lab.profile, dataset_config, data_lst,
                            make_transform_manager(model_config, dataset_config, len(data_lst), opt_args, True),
                            model_config.batch_size, no_paging=no_paging)


def make_data_loader(lab: Lab, dataset_config: DatasetConfig, data_list, transform_manager, batch_size,
                     *, no_paging=False, collate_fn=None, sampler_args=None):
    """
    A convenience method to:
    1) Shuffle the data
    2) Construct the appropriate pytorch data set with a transform manager
    3) Wrap the data set in a data loader.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The data set configuration file used to construct the data list.
    :param data_list: The data list to load. Pre-shuffled.
    :param transform_manager: A transform manager to use.
    :param batch_size: The batch size.
    :param no_paging: Should the loader not page data
    :param collate_fn: Function that controls how samples are collated into batches.
    :param sampler_args: Tuple of integers (world_size, rank) if a sampler is to be used, else None
    :return: PyTorch DataLoader
    """

    # Convenience function to wrap these
    dataset = manifest_to_pytorch_dataset(dataset_config, data_list, transform_manager, no_paging=no_paging)
    # NOTE: We do not shuffle since the dataset conversion above already did
    return wrap_dataset_in_dataloader(lab, dataset, batch_size, collate_fn=collate_fn,
                                      sampler_args=sampler_args)


def manifest_to_pytorch_dataset(dataset_config: DatasetConfig, data_list, transform_manager, *, no_paging=False):
    """
    Wraps the data_list in a Juneberry specific custom pytorch dataset:
    1) Shuffle the data
    2) Construct a transform manager (if transforms is not None)
    3) Construct the appropriate dataset
    :param dataset_config: The dataset configuration file used to construct the data list.
    :param data_list: The data list to load. Pre-shuffled.
    :param transform_manager: A transform manager to add to the data set.
    :param no_paging: Boolean indicating if paging should be disabled.
    :return: PyTorch dataset
    """
    logger.info(f"...shuffling data...")
    random.shuffle(data_list)

    if dataset_config.data_type == DataType.IMAGE and dataset_config.image_data.task_type == TaskType.CLASSIFICATION:
        logger.info(f"...constructing ImageDataset...")
        dataset = ImageDataset(data_list, transform_manager, no_paging)

    elif dataset_config.data_type == DataType.TABULAR:
        logger.info(f"...constructing TabularDataset...")
        dataset = TabularDataset(data_list, transform_manager)

    else:
        logger.error(f"Unsupported DataType - '{dataset_config.data_type}' or task type "
                     f"{dataset_config.task_type}. EXITING!")
        sys.exit(-1)

    logger.info(f"...dataset has len={len(dataset)} from data_list len={len(data_list)}")

    return dataset


def wrap_dataset_in_dataloader(lab: Lab, dataset, batch_size, *,
                               collate_fn=None, sampler_args=None, shuffle=False):
    """
    Wraps the dataset in a DataLoader with the correct configuration.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset: A pytorch dataset to wrap.
    :param batch_size: The batch size.
    :param collate_fn: Function that controls how samples are collated into batches.
    :param sampler_args: Tuple of integers (world_size, rank) if a sampler is to be used, else None
    :param shuffle: True to shuffle data.  By default we assume it is shuffled.
    :return: A pytorch DataLoader
    """

    # A sampler is usually associated with distributed training. The world size indicates the number of
    # distributed processes, while the rank is a way to identify which process is which. The sampler is
    # responsible for making sure the input data is distributed across all processes.
    sampler = None
    if sampler_args is not None:
        world_size, rank = sampler_args
        logger.info(f"...constructing data sampler rank={rank}, world_size={world_size}")
        # TODO: The DistributedSampler has its own "shuffle" arg that defaults to True.
        #  So during distributed training the data gets shuffled, which may lead to
        #  differing results and might be unnecessary.
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': lab.profile.num_workers,
              'collate_fn': collate_fn,
              'sampler': sampler,
              'worker_init_fn': pyt_utils.worker_init_fn}

    # This is required to make sure pieces of sampled data don't show up in multiple processes.
    if sampler_args is not None:
        params['drop_last'] = True

    # We want to pass in batches from the input set so we tell the data loader
    # to provide us with batches.
    logger.info(f"...constructing DataLoader with {params} ...")
    loader = data.DataLoader(dataset, **params)
    logger.info(f"Constructed data loader has len={len(loader)}")
    return loader


#  _____              _          _     _
# |_   _|__  _ __ ___| |____   _(_)___(_) ___  _ __
#   | |/ _ \| '__/ __| '_ \ \ / / / __| |/ _ \| '_ \
#   | | (_) | | | (__| | | \ V /| \__ \ | (_) | | | |
#   |_|\___/|_|  \___|_| |_|\_/ |_|___/_|\___/|_| |_|


class DatasetView(data.Dataset):
    """
    This dataset takes an ordered set of indices and returns a dataset with the items in that order.
    """

    def __init__(self, dataset, remap_list: list):
        """
        Constructs a view into the dataset.
        :param dataset: The dataset of items.
        :param remap_list: A sequence of items to return as the new view.
        """
        self.remap_list = remap_list
        self.dataset = dataset

    def __len__(self):
        """
        :return: Total number of samples.
        """
        return len(self.remap_list)

    def __getitem__(self, index):
        """
        :param index: The index to fetch.
        :return: The item at this index.
        """
        if index >= len(self.remap_list):
            logger.error(f"Asked for item greater than the our len. "
                         f"index={index}, view_len={len(self.remap_list)}, ds_len={len(self.dataset)}")
            sys.exit(-1)
        return self.dataset[self.remap_list[index]]


def construct_transform_manager(transform_list):
    # TODO: Continue refitting the other construct transform manager everywhere
    if transform_list is not None:
        return TransformManager(transform_list)
    return None


def sample_and_split(count, *, sampling_config: SamplingConfig = None, splitting_config: SplittingConfig = None):
    """
    Returns two lists of indices sampled based on the sampling config and then split based on the
    splitting config.
    :param count: The number of indices to sample and split.
    :param sampling_config: Optional sampling config.
    :param splitting_config: Optional splitting config.
    :return: Unshuffled list of training indices and validation indices.
    """

    # Make a list of all the items
    training_indices = list(range(count))
    validation_indices = []

    logger.info(f"Sample and Split: Initially {count} items.")

    # Sample if they gave us a config
    if sampling_config and sampling_config.algo:
        training_indices = jb_data.sample_data_list(training_indices, **sampling_config._asdict())
        logger.info(f"Sample and Split: Sampled down to {len(training_indices)} training items.")
    else:
        logger.info(f"Sample and Split: No sampling")

    # Split if they gave us a config
    if splitting_config and splitting_config.algo:
        # The training_indices are modified IN PLACE
        validation_indices = jb_data.split_list(training_indices, **splitting_config._asdict())
        logger.info(f"Sample and Split: Split to {len(training_indices)} training and {len(validation_indices)} "
                    f"validation items.")
    else:
        logger.info(f"Sample and Split: No splitting")

    # Hand them back
    return training_indices, validation_indices


def construct_torchvision_dataloaders(lab: Lab, tv_data: TorchvisionData, model_config: ModelConfig,
                                      sampling_config: SamplingConfig = None,
                                      *, collate_fn=None, sampler_args=None):
    """
    Constructs the training and validation torchvision dataloaders.
    :param lab: The lab in which this takes place.
    :param tv_data: The torchvision data object.
    :param model_config: The model config.
    :param sampling_config: A config for how to sample the training dataset.
    :param collate_fn: Optional collate function. (Usually for distributed.)
    :param sampler_args: Optional sampler arguments. (Usually for distributed.)
    :return: The constructed training and validation data loaders.
    """
    if tv_data is None:
        logger.error("Asked to construct torchvision datasets but torchvision_data not supplied. EXITING.")
        sys.exit(-1)

    # Construct the training set
    train_dataset = construct_torchvision_dataset(
        lab, tv_data.fqcn, tv_data.root, tv_data.train_kwargs,
        data_transforms=construct_transform_manager(model_config.training_transforms),
        target_transforms=construct_transform_manager(model_config.training_target_transforms))

    # Look at the validation split and see where we want to get the validation data from
    splitting_config = model_config.get_validation_split_config()

    if splitting_config.algo == SplittingAlgo.FROM_FILE:
        logger.error(f"Torchvision datasets do NOT support 'from_file' splitting.")
        sys.exit(-1)
    elif splitting_config.algo == SplittingAlgo.NONE:
        val_dataset = []
    elif splitting_config.algo == SplittingAlgo.RANDOM_FRACTION:
        # Construct a list of indices and sample and split them. Then create two views based on those
        train_indices, val_indices = sample_and_split(len(train_dataset), sampling_config=sampling_config,
                                                      splitting_config=splitting_config)
        orig_dataset = train_dataset
        train_dataset = DatasetView(orig_dataset, train_indices)
        val_dataset = DatasetView(orig_dataset, val_indices)
    elif splitting_config.algo == SplittingAlgo.TORCHVISION:
        # construct and wrap the validation dataset
        val_dataset = construct_torchvision_dataset(
            lab, tv_data.fqcn, tv_data.root, tv_data.val_kwargs,
            data_transforms=construct_transform_manager(model_config.evaluation_transforms),
            target_transforms=construct_transform_manager(model_config.evaluation_target_transforms))
    else:
        logger.error(f"Unknown splitting algo: {splitting_config.algo}")
        sys.exit(-1)

    # Now, wrap the dataset in data loaders
    training_iterable = wrap_dataset_in_dataloader(
        lab, train_dataset, model_config.batch_size, collate_fn=collate_fn, sampler_args=sampler_args,
        shuffle=True)
    validation_iterable = wrap_dataset_in_dataloader(
        lab, val_dataset, model_config.batch_size, collate_fn=collate_fn, sampler_args=sampler_args,
        shuffle=True)

    return training_iterable, validation_iterable


def construct_torchvision_dataset(lab, fqcn: str, data_path: str, kwargs: dict, *,
                                  data_transforms=None,
                                  target_transforms=None):
    """
    Constructs a torchvision style dataset.
    :param lab: The lab in which this takes place.
    :param fqcn: The FQCN to the class to construct.
    :param data_path: A branch path within the data root.
    :param kwargs: A set of kwargs to pass in during construction.
    :param data_transforms: Optional set of data transforms.  Callable that takes 'data'.
    :param target_transforms: Optional set of target transforms. Callable that takes 'target'.
    :return: The constructed instance.
    """
    # Example
    # imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')

    # Make a copy of their args
    if kwargs is not None:
        kwargs = kwargs.copy()
    else:
        kwargs = {}

    # Strip out the three we overwrite so we don't leave any old junk
    for key in ['root', 'transform', 'target_transform']:
        if key in kwargs:
            del kwargs[key]

    # NOTE: The root is an optional argument for most but not all
    opt_args = {'root': str(lab.data_root() / data_path)}

    # Now add our versions
    if data_transforms is not None:
        kwargs['transform'] = data_transforms
    if target_transforms is not None:
        kwargs['target_transform'] = target_transforms

    logger.info(f"Constructing torchvision dataset: '{fqcn}' with args: {kwargs}, {opt_args}")
    # Load the module with the args
    return jbloader.construct_instance(fqcn, kwargs, opt_args)
