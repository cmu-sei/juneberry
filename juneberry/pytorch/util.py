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

from collections import namedtuple
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
import torch.utils.data as torch_data
from torch.utils.data.dataset import T_co
from torchsummary import summary
from torchvision import transforms

from juneberry.config.model import PytorchOptions
import juneberry.data as jb_data
import juneberry.loader as jbloader
import juneberry.loader as model_loader
import juneberry.transform_manager as jbtm

logger = logging.getLogger(__name__)

RandomState = namedtuple("RandomState", "numpy python pytorch")


# ======================================================================================================================
# RANDOM
#  ____                 _
# |  _ \ __ _ _ __   __| | ___  _ __ ___
# | |_) / _` | '_ \ / _` |/ _ \| '_ ` _ \
# |  _ < (_| | | | | (_| | (_) | | | | | |
# |_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_|

def wrap_seed(seed: int):
    """ :return: A numerically wraps the seed if it exceeds the maximum value. """
    # Some system has a maximum value of 32 bits
    return seed & 0xFFFFFFFF


def set_seeds(seed: int):
    """
    Sets all the random seeds used by all the various pieces.
    :param seed: A random seed to use. Can not be None.
    """
    if seed is None:
        logger.error("Request to initialize with a seed of None. Exiting")
        sys.exit(-1)

    if seed > 0xFFFFFFFF:
        logger.error("Seed exceeds maximum seed value of 2**32 - 1 (4294967295). EXITING")
        sys.exit(-1)

    logger.info(f"Setting ALL seeds: {str(seed)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_random_state() -> RandomState:
    """
    :return: A structure containing all the cached random states
    """
    return RandomState(numpy=np.random.get_state(),
                       python=random.getstate(),
                       pytorch=torch.get_rng_state())


def set_random_state(random_state: RandomState) -> None:
    """
    Sets all the various random states form the random state structure.
    :param random_state: The random state structure to set
    :return: None
    """
    np.random.set_state(random_state.numpy)
    random.setstate(random_state.python)
    torch.set_rng_state(random_state.pytorch)


class PyTorchStagedTransform(jbtm.StagedTransformManager):
    def __init__(self, consistent_seed: int, consistent, per_epoch_seed: int, per_epoch):
        super().__init__(consistent_seed, consistent, per_epoch_seed, per_epoch)
        self.random_state: RandomState
        self.random_state = None

    def save_random_state(self):
        self.random_state = get_random_state()

    def restore_random_state(self):
        set_random_state(self.random_state)

    def set_seeds(self, seed):
        set_seeds(seed)


# class StagedTransformManager:
#     """
#     A callable transform manager that manages the random seed state in a predictable fashion based on an initial
#     seed state and the seed index.
#     """
#
#     def __init__(self, consistent_seed: int, consistent, per_epoch_seed: int, per_epoch):
#         self.consistent_transform = consistent
#         self.consistent_seed = consistent_seed
#         self.per_epoch_transform = per_epoch
#         self.per_epoch_seed = per_epoch_seed
#
#     def __call__(self, item, index, epoch):
#         # Capture the random state
#         random_state = get_random_state()
#
#         # Set the random seed based on index only
#         seed = wrap_seed(self.consistent_seed + index)
#         set_seeds(seed)
#         item = self.consistent_transform(item)
#
#         # Now, execute the per epoch transform
#         seed = wrap_seed(self.per_epoch_seed + index + epoch)
#         set_seeds(seed)
#         item = self.per_epoch_transform(item)
#
#         # Restore the state
#         set_random_state(random_state)
#
#         return item


def worker_init_fn(worker_id):
    # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    # SUMMARY:
    # Pytorch initializes the random seeds for every worker process on every epoch
    # for pytorch and python but NOT for numpy. We need to do that for numpy.
    # The post suggests using:
    #   np.random.get_state()[1][0] + worker_id
    # but that ends up being the same for every epoch and we don't want that.
    # Numpy only supports 32-bit seeds, so we just use the lower bits.
    seed = torch_data.get_worker_info().seed & 0xFFFFFFFF
    logger.debug(f"Setting worker {worker_id} numpy seed to {seed}")
    np.random.seed(seed)


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


def save_model(model_manager, model) -> None:
    """
    Saves the model to the specified directory using our naming scheme and format.
    :param model_manager: The model manager controlling the model being saved.
    :param model: The model file.
    """
    # We only want to save the non DDP version of the model, so the model without wrappers.
    # We shouldn't be passed a wrapped model.
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        logger.error("ERROR: Being asked to save a DataParallel, DistributedDataParallel model.")
        traceback.print_stack()
        sys.exit(-1)

    model_path = model_manager.get_pytorch_model_path()
    torch.save(model.state_dict(), model_path)


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


def compute_accuracy(y_pred, y_true, accuracy_function, accuracy_args, binary):
    """
    Computes the accuracy from a set of predictions where the output is rows and the classes are the columns.
    :param y_pred: The output predictions to process.
    :param y_true: The correct labels.
    :param accuracy_function: The actual function that does the computation
    :param accuracy_args: Arguments that should be passed to the accuracy function
    :param binary: True if this a binary function.
    :return: Accuracy as a float.
    """
    with torch.set_grad_enabled(False):
        # The with clause should turn off grad, but for some reason I still get the error:
        # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
        # So I am including detach. :(
        if binary:
            np_y_pred = y_pred.type(torch.DoubleTensor).cpu().detach().numpy()
            np_y_true = y_true.type(torch.DoubleTensor).unsqueeze(1).cpu().numpy()
        else:
            np_y_pred = y_pred.cpu().detach().numpy()
            np_y_true = y_true.cpu().numpy()

        # Convert the continuous predictions to single class predictions
        singular_y_pred = continuous_predictions_to_class(np_y_pred, binary)

        # Now call the function
        return accuracy_function(y_pred=singular_y_pred, y_true=np_y_true, **accuracy_args)


def binary_to_classes(binary_predictions):
    """
    Expands the singular binary predictions to two classes
    :param binary_predictions:
    :return: The predictions broken into two probabilities.
    """
    return [[1.0 - x[0], x[0]] for x in binary_predictions]


def continuous_predictions_to_class(y_pred, binary):
    """
    Convert a set of continuous predictions to numeric class.
    :param y_pred: The float predictions.
    :param binary: True if the data is binary
    :return: The classes
    """
    if binary:
        return np.round(y_pred).astype(int)
    else:
        return np.argmax(y_pred, axis=1)


def predict_classes(data_generator, model, device):
    """
    Generates predictions data for the provided data set via this model.
    :param data_generator: The data generator to provide data.
    :param model: The trained model.
    :param device: The device on which to do the predictions. The model should already be on the device.
    :return: A table of the predictions.
    """
    all_outputs = None
    for local_batch, local_labels in data_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        output = model(local_batch)

        with torch.set_grad_enabled(False):
            if all_outputs is None:
                all_outputs = output.detach().cpu().numpy()
            else:
                all_outputs = np.concatenate((all_outputs, output.detach().cpu().numpy()))

    return all_outputs.tolist()


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


def invoke_evaluator_method(evaluator, module_name: str):
    """
    This function is responsible for invoking methods during evaluation.
    :param evaluator: A Juneberry Evaluator object that is managing the evaluation.
    :param module_name: The module being invoked.
    :return: Nothing.
    """
    split_name = module_name.split(".")
    module_path = ".".join(split_name[:-1])
    class_name = split_name[-1]
    args = {"evaluator": evaluator}

    jbloader.invoke_method(module_path=module_path, class_name=class_name, method_name="__call__", method_args=args)


class EpochDataset(torch_data.Dataset):
    """ Base class for datasets that support an epoch notion. """

    def __init__(self):
        self.epoch = 0

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def set_epoch(self, epoch):
        self.epoch = epoch
