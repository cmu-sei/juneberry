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
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import balanced_accuracy_score
import sys
import torch
import traceback

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torchsummary import summary
from torchvision import transforms

from juneberry.config.dataset import DataType, DatasetConfig, SamplingConfig, TaskType, TorchvisionData
from juneberry.config.model import ModelConfig, PytorchOptions, SplittingAlgo, SplittingConfig
import juneberry.data as jb_data
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
import juneberry.loader as jbloader
import juneberry.loader as model_loader
from juneberry.pytorch.evaluation.util import compute_accuracy
from juneberry.pytorch.image_dataset import ImageDataset
from juneberry.pytorch.tabular_dataset import TabularDataset
from juneberry.transform_manager import TransformManager
import juneberry.utils as jb_utils

logger = logging.getLogger(__name__)


def make_data_loaders(lab, dataset_config, model_config, data_lst, split_lst, *, no_paging=False, predict_mode=False,
                      collate_fn=None, sampler_args=None):
    """
    Creates the appropriate data loaders based on provided data lists and the configuration switches.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The data set config that describes the data.
    :param model_config: A Juneberry ModelConfig object that may contain transforms and validation options.
    :param data_lst: The data list to load. Pre-shuffled.
    :param split_lst: The list of data to load that was split from the main dataset.
    :param no_paging: Set to true to read all the data at once. Good for small data sets or large memory.
    :param collate_fn: Function that controls how samples are collated into batches.
    :param sampler_args: Tuple of integers (world_size, rank) if a sampler is to be used; None for no sampler
    :param predict_mode: Boolean indicating if function is being called in prediction mode.
    :return: When unit_test is False, this function returns data_loader, split_loader.
    """
    split_loader = None
    if predict_mode:
        logger.info("Constructing data loader from EVALUATION data set using prediction transforms.")
        data_loader = make_data_loader(lab, dataset_config, data_lst, model_config.evaluation_transforms,
                                       model_config.batch_size, no_paging=no_paging)

    else:
        logger.info("Constructing TRAINING data loader.")
        data_loader = make_data_loader(lab, dataset_config, data_lst, model_config.training_transforms,
                                       model_config.batch_size, no_paging=no_paging, collate_fn=collate_fn,
                                       sampler_args=sampler_args)

        logger.info("Constructing VALIDATION data loader.")
        split_loader = make_data_loader(lab, dataset_config, split_lst, model_config.evaluation_transforms,
                                        model_config.batch_size, no_paging=no_paging, collate_fn=collate_fn)

    return data_loader, split_loader


def make_data_loader(lab: Lab, dataset_config: DatasetConfig, data_list, transform_config, batch_size,
                     *, no_paging=False, collate_fn=None, sampler_args=None):
    """
    A convenience method to:
    1) Shuffle the data
    2) Construct a transform manager (if transforms is not None)
    3) Construct the appropriate data set
    4) Wrap the data set in a data loader.
    :param lab: The Juneberry Lab in which this operation occurs.
    :param dataset_config: The data set configuration file used to construct the data list.
    :param data_list: The data list to load. Pre-shuffled.
    :param transform_config: A configuration for a TransformationManager.
    :param batch_size: The batch size.
    :param no_paging: Should the loader not page data
    :param collate_fn: Function that controls how samples are collated into batches.
    :param sampler_args: Tuple of integers (world_size, rank) if a sampler is to be used, else None
    :return: PyTorch DataLoader
    """

    # Convenience function to wrap these
    dataset = manifest_to_pytorch_dataset(dataset_config, data_list, transform_config, no_paging=no_paging)
    # NOTE: We do not shuffle since the dataset conversion above already did
    return wrap_dataset_in_dataloader(lab, dataset, batch_size, collate_fn=collate_fn, sampler_args=sampler_args)


def manifest_to_pytorch_dataset(dataset_config: DatasetConfig, data_list, transform_config, *, no_paging=False):
    """
    Wraps the data_list in a Juneberry specific custom pytorch dataset:
    1) Shuffle the data
    2) Construct a transform manager (if transforms is not None)
    3) Construct the appropriate dataset
    :param dataset_config: The dataset configuration file used to construct the data list.
    :param data_list: The data list to load. Pre-shuffled.
    :param transform_config: A configuration for a TransformationManager.
    :param no_paging: Boolean indicating if paging should be disabled.
    :return: PyTorch dataset
    """
    logger.info(f"...shuffling data...")
    random.shuffle(data_list)

    transform_manager = None

    if transform_config is not None:
        logger.info(f"...found transforms - attempting construction...")
        transform_manager = TransformManager(transform_config)

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
              'num_workers': lab.num_workers,
              'collate_fn': collate_fn,
              'sampler': sampler,
              'worker_init_fn': worker_init_fn}

    # This is required to make sure pieces of sampled data don't show up in multiple processes.
    if sampler_args is not None:
        params['drop_last'] = True

    # We want to pass in batches from the input set so we tell the data loader
    # to provide us with batches.
    logger.info(f"...constructing DataLoader with {params} ...")
    loader = data.DataLoader(dataset, **params)
    logger.info(f"Constructed data loader has len={len(loader)}")
    return loader


def worker_init_fn(worker_id):
    # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    # SUMMARY:
    # Pytorch initializes the random seeds for every worker process on every epoch
    # for pytorch and python but NOT for numpy. We need to do that for numpy.
    # The post suggests using:
    #   np.random.get_state()[1][0] + worker_id
    # but that ends up being the same for every epoch and we don't want that.
    # Numpy only supports 32-bit seeds, so we just use the lower bits.
    seed = data.get_worker_info().seed & 0xFFFFFFFF
    logger.debug(f"Setting worker {worker_id} numpy seed to {seed}")
    np.random.seed(seed)


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


def construct_torchvision_dataloaders(lab, tv_data: TorchvisionData, model_config: ModelConfig,
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


#  __  __           _      _
# |  \/  | ___   __| | ___| |
# | |\/| |/ _ \ / _` |/ _ \ |
# | |  | | (_) | (_| |  __/ |
# |_|  |_|\___/ \__,_|\___|_|

def construct_model(arch_config, num_model_classes):
    """
    Loads/constructs the requested model type.
    :param arch_config: Dictionary describing the architecture.
    :param num_model_classes: The number of model classes.
    :return: The constructed model.
    """

    # Split the module name to module and path
    class_data = arch_config['module'].split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]
    args = arch_config.get('args', {})
    jb_data.check_num_classes(args, num_model_classes)

    return model_loader.invoke_method(module_path=module_path,
                                      class_name=class_str,
                                      method_name="__call__",
                                      method_args=args,
                                      dry_run=False)


def save_model(model_manager: ModelManager, model, input_sample) -> None:
    """
    Saves the model to the specified directory using our naming scheme and format.
    :param model_manager: The model manager controlling the model being saved.
    :param model: The model to save.
    :param input_sample: A single sample from the input data. The dimensions of this sample are used to
    perform the tracing when exporting the ONNX model file.
    """
    # We only want to save the non DDP version of the model, so the model without wrappers.
    # We shouldn't be passed a wrapped model.
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        logger.error("ERROR: Being asked to save a DataParallel, DistributedDataParallel model.")
        traceback.print_stack()
        sys.exit(-1)

    # Save the model in PyTorch format.
    logging.info(f"Saving PyTorch model file to {model_manager.get_pytorch_model_path()}")
    model_path = model_manager.get_pytorch_model_path()
    torch.save(model.state_dict(), model_path)

    # Save the model in ONNX format.
    # LIMITATION: If the model is dynamic, e.g., changes behavior depending on input data, the
    # ONNX export won't be accurate. This is because the ONNX exporter is a trace-based exporter.
    logging.info(f"Saving ONNX model file to {model_manager.get_onnx_model_path()}")
    torch.onnx.export(model, input_sample, model_manager.get_onnx_model_path(), export_params=True)


def load_model(model_path, model) -> None:
    """
    Loads the model weights from the model directory using our model naming scheme and format.
    :param model_path: The model directory.
    :param model: The model file into which to load the model weights.
    """
    model.load_state_dict(torch.load(str(model_path)), strict=False)
    model.eval()


def load_weights_from_model(model_manager, model) -> None:
    """
    Loads the model weights from the model directory using our model naming scheme and format.
    :param model_manager: The model manager responsible for the model containing the desired weights.
    :param model: The model file into which to load the model weights.
    """
    model_path = model_manager.get_pytorch_model_path()
    if Path(model_path).exists():
        model.load_state_dict(torch.load(str(model_path)), strict=False)
    else:
        logger.error(f"Model path {model_path} does not exist! EXITING.")
        sys.exit(-1)


def set_pytorch_seeds(seed: int):
    """
    Sets all the random seeds used by all the various pieces.
    :param seed: A random seed to use. Can not be None.
    """
    jb_utils.set_seeds(seed)
    logger.info(f"Setting PyTorch seed to: {str(seed)}")
    torch.manual_seed(seed)


def make_loss(config: PytorchOptions, model, binary):
    """
    Produces the desired loss function from them configuration file.
    :param config: The config stanza for model.
    :param model: Model that will be passed into the loss __init__ function if 'model' is in the signature.
    :param binary: True if this is a binary model.
    :return: The loss function
    """
    loss_fn = None
    if config is not None:
        function_name = config.loss_fn
        function_args = config.loss_args
        optional_args = {'model': model}

        if function_name is not None:
            logger.info(f"Constructing loss function '{function_name}' with args '{function_args}'")
            loss_fn = jbloader.construct_instance(function_name, function_args, optional_args)

    if loss_fn is None:
        logger.warning("No loss function specified. Defaulting to torch.nn.CrossEntropyLoss with default arguments")
        loss_fn = torch.nn.CrossEntropyLoss()

    # If binary, unpack the labels
    if binary:
        loss_fn = function_wrapper_unsqueeze_1(loss_fn)

    return loss_fn


def make_optimizer(config: PytorchOptions, model):
    """
    Produces an optimizer based on the optimizer configuration.
    :param config: The pytorch config.
    :param model: The model.
    :return: The optimizer function.
    """
    if config is not None:
        opt_fn = config.optimizer_fn
        opt_args = config.optimizer_args if config.optimizer_args is not None else {}

        if opt_fn is not None:
            logger.info(f"Constructing optimizer '{opt_fn}' with args {opt_args}")
            opt_args['params'] = model.parameters()
            return jbloader.construct_instance(opt_fn, opt_args)

    logger.warning("No optimizer specified. Defaulting to torch.optim.SGD with lr=0.01")
    return torch.optim.SGD(model.parameters(), lr=0.01)


def make_lr_scheduler(config: PytorchOptions, optimizer, max_epochs):
    """
    Produces a learning rate scheduler based on the lr_schedule configuration.
    :param config: The pytorch config.
    :param optimizer: The optimizer function that's being used.
    :param max_epochs: The maximum number of epochs.
    :return: A learning rate scheduler.
    """

    if config is None or config.lr_schedule_fn is None:
        return None
    if config.lr_schedule_args is None:
        logger.error(f"No args provided for learning rate scheduler {config.lr_schedule}. EXITING.")
        sys.exit(-1)

    lr_name = config.lr_schedule_fn
    lr_args = config.lr_schedule_args

    logger.info(f"Constructing lr scheduler '{lr_name}' with args '{lr_args}'")

    # For now we only support 4 types of lr_scheduler; there are 7 more.
    # TODO: Deprecate these except for LambdaLR
    try:
        if lr_name == 'MultiStepLR':
            logger.warning("MultiStepLR scheduler used: prefer using torch.optim.lr_scheduler.MultiStepLR")
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_args['milestones'], lr_args['gamma'])

        elif lr_name == 'StepLR':
            logger.warning("StepLR scheduler used: prefer using torch.optim.lr_scheduler.StepLR")
            return torch.optim.lr_scheduler.StepLR(optimizer, lr_args['step_size'], lr_args['gamma'])

        elif lr_name == 'CyclicLR':
            logger.warning("CyclicLR scheduler used: prefer using torch.optim.lr_scheduler.CyclicLR")
            return torch.optim.lr_scheduler.CyclicLR(optimizer, lr_args['base_lr'], lr_args['max_lr'],
                                                     step_size_up=lr_args['step_size_up'])

        elif lr_name == 'LambdaLR':
            args = jbloader.extract_kwargs(lr_args)
            if args is None:
                logger.error(f"Failed to extract args for 'LambdaLR' scheduler. args={args}. Exiting.")
                sys.exit(-1)
            args['kwargs']['epochs'] = max_epochs
            fn_obj = jbloader.construct_instance(**args)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, fn_obj)

        else:
            # Add our optimizer to the lr_args and then any additional ones they want.
            # lr_args['optimizer'] = optimizer
            lr_args['optimizer'] = optimizer
            return jbloader.construct_instance(lr_name, lr_args, {'epochs': max_epochs})

    except KeyError as missing_key:
        logger.error(f"Key named {missing_key} not found in learning rate scheduler args {lr_args}. Exiting.")
        sys.exit(-1)


def make_accuracy(config: PytorchOptions, binary):
    """
    Constructs the accuracy function from the provided config.
    :param config: The configuration that specifies the accuracy function and function arguments.
    :param binary: Set to true for Binary functions.
    :return: The constructed accuracy function.
    """
    if config.accuracy_fn is not None:
        acc_name = config.accuracy_fn
        acc_args = config.accuracy_args if config.accuracy_args is not None else {}
        signature_args = acc_args.copy()
        signature_args['y_pred'] = []
        signature_args['y_true'] = []

        logger.info(f"Constructing accuracy function '{acc_name}' with optional args '{acc_args}'")
        acc_fn = jbloader.load_verify_fqn_function(acc_name, signature_args)
        if acc_fn is None:
            logger.error(f"Failed to load accuracy function '{acc_name}'. See log for details. EXITING!!")
            sys.exit(-1)
    else:
        logger.info("No accuracy function specified. Defaulting to 'sklearn.metrics.balanced_accuracy_score'")
        acc_fn = balanced_accuracy_score
        acc_args = {}

    # Now wrap their accuracy function in our data unpacking/formatting
    return lambda x, y: compute_accuracy(x, y, acc_fn, acc_args, binary)


def function_wrapper_unsqueeze_1(fn):
    """ A simple wrapper for unsqueezing the second argument
    :param fn: The underlying function to call
    :return: The function call that unsqueezes and calls the original function
    """
    return lambda a, b: fn(a.type(torch.DoubleTensor), b.type(torch.DoubleTensor).unsqueeze(1))


def output_summary_file(model, image_shape, summary_file_path) -> None:
    """
    Saves a summary of the model to the specified path assuming the provide input shape.
    :param model: The model to summarize
    :param image_shape: The input shape to use for the model
    :param summary_file_path: The path in which to save the output
    """
    orig = sys.stdout
    sys.stdout = open(summary_file_path, 'w+', encoding="utf-8")
    summary(model, image_shape)
    sys.stdout = orig


def generate_sample_images(data_loader, quantity, img_path: Path):
    """
    This function will save some quantity of images from a data iterable.
    :param data_loader: A dataloader of images; typically training images.
    :param quantity: The maximum number of images to sample. The function
    :param img_path: Path in which to save images
    :return: The shape of the first image encountered.
    """

    # Make sure we can save the images
    if not img_path.exists():
        img_path.mkdir(parents=True)

    # Calculate the max number of batches
    num_batches = len(data_loader)

    # Reset the random seed so we get different images each dry run
    random.seed()

    # Loop through each batch and sample an image
    img_shape = None
    for x in range(min(num_batches, quantity) + 1):

        # Get the next batch of images
        images, labels = next(iter(data_loader))

        if img_shape is None:
            img_shape = images[0].shape

        # Pick an image in the batch at random
        rand_idx = random.randrange(0, len(images))
        img = transforms.ToPILImage()(images[rand_idx])

        # Save the image
        img.save(str(img_path / f"{x}.png"))

    logger.info(f'{min(num_batches, quantity) + 1} sample images saved to {img_path}')
    return img_shape
