#! /usr/bin/env python3

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

import sys
import torch
import random
import logging
import numpy as np

from pathlib import Path
from torch.utils import data
from torchsummary import summary
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score

import juneberry
import juneberry.loader as jbloader
import juneberry.loader as model_loader
from juneberry.transform_manager import TransformManager
from juneberry.pytorch.image_dataset import ImageDataset
from juneberry.pytorch.tabular_dataset import TabularDataset
from juneberry.config.dataset import DataType, DatasetConfig, TaskType


def make_data_loader(data_set_config: DatasetConfig, data_list, transform_config, batch_size, no_paging,
                     collate_fn=None):
    """
    A convenience method to:
    1) Shuffle the data
    2) Construct a transform manager (if transforms is not None)
    3) Construct the appropriate data set
    4) Wrap the data set in a data loader.
    :param data_set_config: The data set configuration file used to construct the data list.
    :param data_list: The data list to load. Pre-shuffled.
    :param transform_config: A configuration for a TransformationManager.
    :param batch_size: The batch size.
    :param no_paging: Should the loader not page data
    :param collate_fn: Collate function to use when setting up the data loader.
    :return: PyTorch DataLoader
    """

    logging.info(f"...shuffling data...")
    random.shuffle(data_list)

    transform_manager = None
    data_set = None

    if transform_config is not None:
        logging.info(f"...found transforms - attempting construction...")
        transform_manager = TransformManager(transform_config)

    if data_set_config.data_type == DataType.IMAGE:

        # If it's a classification task, the loader will use a dataset of images
        if data_set_config.task_type == TaskType.CLASSIFICATION:
            logging.info(f"...constructing ImageDataset...")
            data_set = ImageDataset(data_list, transform_manager, no_paging)

    elif data_set_config.data_type == DataType.TABULAR:
        logging.info(f"Constructing TabularDataset...")
        data_set = TabularDataset(data_list, transform_manager)

    else:
        logging.error(f"Unsupported DataType - '{data_set_config.data_type}'. EXITING!")
        sys.exit(-1)

    # Parameters
    # NOTE: We shuffle the list ourselves so we don't want the data set doing it for us
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': juneberry.NUM_WORKERS,
              'collate_fn': collate_fn}

    # We want to pass in batches from the input set so we tell the data loader
    # to provide us with batches.
    logging.info(f"...constructing DataLoader...")
    return data.DataLoader(data_set, **params)


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
    if 'num_classes' not in args:
        logging.warning(f"The 'modelArchitecture' 'args' does not contain 'num_classes' for validation. "
                        f"Using '{num_model_classes}' from the data set config.")
        args['num_classes'] = num_model_classes
    else:
        if args['num_classes'] != num_model_classes:
            logging.error(f"Num Classes in the training config: '{args['num_classes']}' "
                          f"does not match that in the data set: '{num_model_classes}'. EXITING")
            sys.exit(-1)

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
    model_path = model_manager.get_pytorch_model_file()
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
    model_path = model_manager.get_pytorch_model_file()
    if Path(model_path).exists():
        model.load_state_dict(torch.load(str(model_path)), strict=False)
    else:
        logging.error(f"Model path {model_path} does not exist! EXITING.")
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


def set_seeds(seed: int):
    """
    Sets all the random seeds used by all the various pieces.
    :param seed: A random seed to use. Can not be None.
    """
    if seed is None:
        logging.error("Request to initialize with a seed of None. Exiting")
        sys.exit(-1)

    logging.info(f"Setting ALL seeds: {str(seed)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_criterion(config, binary):
    """
    Produces the desired loss function from them configuration file.
    :param config: The config stanza for model.
    :param binary: True if this is a binary model.
    :return: The loss function
    """
    loss_fn = None
    if config is not None:
        function_name = config.get('lossFunction', None)
        function_args = config.get('lossArgs', {})

        if function_name is not None:
            logging.info(f"Constructing loss function '{function_name}' with args {function_args}")
            loss_fn = jbloader.construct_instance(function_name, function_args)

    if loss_fn is None:
        logging.warning("No loss function specified. Defaulting to torch.nn.CrossEntropyLoss with default arguments")
        loss_fn = torch.nn.CrossEntropyLoss()

    # If binary, unpack the labels
    if binary:
        loss_fn = function_wrapper_unsqueeze_1(loss_fn)

    return loss_fn


def make_optimizer(config, model):
    """
    Produces an optimizer based on the optimizer configuration.
    :param config: The pytorch config.
    :param model: The model.
    :return: The optimizer function.
    """
    if config is not None:
        opt_fn = config.get('optimizer', None)
        opt_args = config.get('optimizerArgs', {})

        if opt_fn is not None:
            logging.info(f"Constructing optimizer '{opt_fn}' with args {opt_args}")
            opt_args['params'] = model.parameters()
            return jbloader.construct_instance(opt_fn, opt_args)

    logging.warning("No optimizer specified. Defaulting to torch.optim.SGD with lr=0.01")
    return torch.optim.SGD(model.parameters(), lr=0.01)


def make_lr_scheduler(config, optimizer):
    """
    Produces a learning rate scheduler based on the lrSchedule configuration.
    :param config: The pytorch config.
    :param optimizer: The optimizer function that's being used.
    :return: A learning rate scheduler.
    """

    if 'lrSchedule' not in config:
        return None

    if 'lrScheduleArgs' in config:
        lr_args = config['lrScheduleArgs']

    else:
        logging.error(f"No args provided for learning rate scheduler. Exiting.")
        sys.exit(-1)

    logging.info(f"Constructing lr scheduler '{config['lrSchedule']}' with args {lr_args}")

    # For now we only support 3 types of lr_scheduler; there are 7 more.
    try:
        if config['lrSchedule'] == 'MultiStepLR':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_args['milestones'], lr_args['gamma'])

        elif config['lrSchedule'] == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer, lr_args['step_size'], lr_args['gamma'])
            
        elif config['lrSchedule'] == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(optimizer, lr_args['base_lr'], lr_args['max_lr'],
                                                     step_size_up=lr_args['step_size_up'])

        else:
            logging.error("Learning rate scheduler config invalid or scheduler type not supported. Exiting.")
            sys.exit(-1)

    except KeyError as missing_key:
        logging.error(f"Key named {missing_key} not found in learning rate scheduler args. Exiting.")
        sys.exit(-1)


def make_accuracy(config, binary):
    """
    Constructs the accuracy function from the provided config.
    :param config: The configuration that specifies the accuracy function and function arguments.
    :param binary: Set to true for Binary functions.
    :return: The constructed accuracy function.
    """
    if 'accuracyFunction' in config:
        acc_name = config['accuracyFunction']
        acc_args = config.get('accuracyArgs', {})
        signature_args = acc_args.copy()
        signature_args['y_pred'] = []
        signature_args['y_true'] = []

        logging.info(f"Constructing accuracy function {acc_name} with optional args {acc_args}")
        acc_fn = jbloader.load_verify_fqn_function(acc_name, signature_args)
        if acc_fn is None:
            logging.error(f"Failed to load accuracy function '{acc_name}'. See log for details. EXITING!!")
            sys.exit(-1)
    else:
        logging.info("No accuracy function specified. Defaulting to 'sklearn.metrics.balanced_accuracy_score'")
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

    logging.info(f'{min(num_batches, quantity) + 1} sample images saved to {img_path}')
    return img_shape
