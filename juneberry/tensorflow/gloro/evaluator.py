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
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import Model
from juneberry.config.model import ModelConfig
from juneberry.filesystem import ModelManager, EvalDirMgr
import juneberry.tensorflow.data as tf_data
import juneberry.tensorflow.evaluator

logger = logging.getLogger(__name__)


# TODO: move these util functions to workspace
from einops import rearrange
from typing import Optional, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

if not HAS_TORCH and not HAS_TF:
    raise EnvironmentError('Must have either pytorch or tensorflow installation.')

if HAS_TORCH and HAS_TF:
    Tensor = Union[torch.Tensor, tf.Tensor]
else:
    Tensor = torch.Tensor if HAS_TORCH else tf.Tensor

def unsqueeze(
    x        :Tensor, 
    n        :int    =1, 
    prepend  :bool   =False
) -> Tensor:
    """Prepend or append n-many singleton dimensions to a tensor.
    """
    if prepend:
        return rearrange(x, '... ->' + ' 1' * n + ' ...')
    else:
        return rearrange(x, '... -> ...' + ' 1' * n)
        

def unsqueeze_like(
    x        :Tensor, 
    y        :Tensor,
    dim      :Optional[int]  =None,
    prepend  :bool           =False,
) -> Tensor:
    """Prepend and append singleton dimensions to `x` to match the shape of `y`. 
    `y` must have at least as many dimensions as `x`.

    Example usage: 
    ```
    x = torch.randn(3)
    y = torch.randn(3,3,32,32)
    unsqueeze_like(x, y)         # returns shape (3,1,1,1)
    unsqueeze_like(x, y, dim=1)  # returns shape (3,1,1)
    unsqueeze_like(x, y, dim=2)  # error
    ```

    :param x        : Tensor to reshape.
    :param y        : Tensor to match.
    :param dim      : Specifies a dimension of `y` to match. If `None`, then automatically. 
    :param prepend  : `False` only appends singleton dimensions.
                      `True` prepends singleton dimensions as well.
                      NOTE: broadcasting semantics don't require prepend singleton dimensions.
    """
    assert len(x.shape) <= len(y.shape), \
        f'Argument `x` of shape `{x.shape}` has more dimensions than argument `y` of shape `{y.shape}`.'\
        'Did you mean to match the dimensions of `y` to `x` instead?'
    if x.shape == y.shape:
        return x
    if len(x.shape) == 0:
        x = x[None]

    # Try to find the first `dim_x` dimension of `x` that is non-singleton.
    for dim_x in range(len(x.shape)):
        if x.shape[dim_x] != 1:
            break

    # Find the `dim_y` dimension of `y` matching the `dim_x` dimension of `x` so we can calculate the number of 
    # singleton dimensions to prepend and append.
    if dim is not None:
        dim_y = dim
        assert x.shape[dim_x] == y.shape[dim_y], \
            f'Dimensions do not match: `x.shape[{dim_x}] != y.shape[{dim_y}]` '\
            f'with values `{x.shape[dim_x]} != {y.shape[dim_y]}`'
    else:
        matched = False
        for dim_y in range(len(y.shape)):
            if ((x.shape[dim_x] == 1) or              # dimension size 1 can match any via broadcasting
                (y.shape[dim_y] == x.shape[dim_x])):  
                matched = True
                break 
        assert matched, \
            f'Could not find a dimension in `y.shape == {y.shape}` matching `x.shape[{dim_x}] == {x.shape[dim_x]}`'

    # Compute the number of prepend and append singleton dimensions
    n_append = max(0, len(y.shape[dim_y:]) - len(x.shape[dim_x:]))
    n_prepend = max(0, dim_y - dim_x)
    n_extra_dims = len(y.shape) - len(x.shape) - n_prepend - n_append
    assert n_extra_dims == 0, \
        f'Matched dimension `x.shape[{dim_x}] == {x.shape[dim_x]}` to `y.shape[{dim_y}] == {y.shape[dim_y]}`, '\
        f'but this results in {n_prepend} prepend and {n_append} append dimensions, resulting in '\
        f'{n_extra_dims} too many singleton dimensions. The shapes are `x.shape == {x.shape}` and '\
        f'`y.shape == {y.shape}`'

    # broadcasting semantics no longer require singleton dimensions prepended
    # return unsqueeze(unsqueeze(x, n_prepend, prepend=True), n_append)
    x_new = unsqueeze(x, n_append)
    if prepend:
        x_new = unsqueeze(x_new, n_prepend, prepend=True)
    return x_new 


# TODO: move this to a workspace file
from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import einops


def normalize(x, geometry, *, eps=1e-10) -> tf.Tensor:
    """A collection of batched normalization functions.
    """
    # l2 normalization over all axes > 0 flattened. Can also be thought of as Frobenius normalization.
    if geometry == 'l2':
        x_flat = einops.rearrange(x, 'b ... -> b (...)') 
        norm = tf.linalg.norm(x_flat, axis=1)
        return x / (unsqueeze_like(norm, x) + eps)

    # Normalizing wrt to linf norm is equivalent to projecting onto boundary of the hypercube [-1,1]^d.
    # The projection operator is therefore elementwise sign.
    elif geometry == 'linf':
        return tf.sign(x)

    # Normalize so that each tensor in the batch has pixel values in the range [0,1].
    elif geometry == '01':
        x_flat = einops.rearrange(x, 'b ... -> b (...)')
        max_val = tf.reduce_max(x_flat, axis=1)
        min_val = tf.reduce_min(x_flat, axis=1)
        return (x - unsqueeze_like(min_val, x)) / (unsqueeze_like(max_val, x) + eps)

    else:
        raise NotImplementedError(geometry)


def _inversion_sgd(
    model: keras.Model, 
    x: tf.Tensor, 
    target: tf.Tensor,
    *,
    step_size: float = 1e-3,
    num_steps: int = 128,
    geometry: str = 'l2',
    weights_std: Optional[float] = None,
) -> tf.Tensor:
    """Perform `num_steps` many iterations of gradient ascent on the loss function with respect to the input.
    Gradients are normalized according to a specified geometry to improve convergence.
    In the case of `geometry='linf'`, this is equivalent to multiple iterates of FGSM targeting a specific label.

    NOTE: This method is not meant for direct use. Instead, use `iterated_inversion`.
    """
    if weights_std is not None:
        weights_orig = model.get_weights()
        weights_noisy = deepcopy(weights_orig)
        for layer in weights_noisy:
            layer += weights_std * np.random.normal(size=layer.shape)
        model.set_weights(weights_noisy)

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = model(x)
            loss = model.loss(target, prediction)
        
        grad = tape.gradient(loss, x)
        grad = normalize(grad, geometry)
        x += step_size * grad

    if weights_std is not None:
        model.set_weights(weights_orig)

    return x


def iterated_inversion(model, x0, target_sequence, **inversion_kwargs):
    x = tf.identity(x0)
    for target in target_sequence:
        x = _inversion_sgd(model, x, target, **inversion_kwargs)
    return normalize(x, '01')


# HACK - This needs to go into the workspace
class Evaluator(juneberry.tensorflow.evaluator.Evaluator):
    def __init__(self, model_config: ModelConfig, lab, dataset: DatasetConfig, model_manager: ModelManager,
                 eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)
        self.inversion = None

    def obtain_model(self) -> None:
        from gloro import GloroNet 
        model_file = str(           # lib-gloro needs a str type for the file name, not a pathlib object.
            self.model_manager.get_train_root_dir() 
            / "../model.gloronet")  # Need to look for the model in the experiment/ directory, not the experiment/train directory.
        logger.info(f"Loading model {model_file}...")
        self.model = GloroNet.load_model(model_file)

        # lib-gloro seems to save the model in a way that is extricated from metrics, losses, and optimizers.
        # As such, need to re-compile the model with the desired evaluation functions.

        # We only actually need the loss here, so neglect the metrics in the config file.
        from juneberry.loader import construct_instance
        loss = construct_instance(
            self.model_config.tensorflow.loss_fn, 
            self.model_config.tensorflow.loss_args)

        logger.info(f'Compiling model with:'
                    f'\n\tloss: {loss.__class__.__name__}')
        self.model.compile(loss=loss)
        self.model.trainable = False

    def obtain_dataset(self) -> None:
        pass

    def evaluate_data(self) -> None:
        """
        TODO: 
        - Get watermark metadata to generate labels
        - Save labeled inversions.
        """
        logger.info('Computing model inversions...')
        if self.model_config.get('inversions') is None:
            raise KeyError(f'Model config missing required field "inversions"')

        num_samples = self.model_config.inversions.num_samples
        num_iterations = self.model_config.inversions.get('num_iterations', 128)
        step_size = self.model_config.inversions.get('step_size', 1e-3)
        target_label_sequence_length = self.model_config.get('target_label_sequence_length', 1)

        # TODO: input_shape batch size from config
        input_shape = (num_samples, *self.model.input_shape[1:])
        x0 = tf.zeros(input_shape)

        # Stupidly do a forward-pass to get output shape
        output_shape = self.model(x0).shape

        # Target label sequence for each batch entry.
        target_sequence = [
            np.random.randint(0, output_shape[-1], size=(num_samples,))
            for _ in range(target_label_sequence_length)]

        # Perform gradient ascent for num_steps iterations per label in the sequence.
        # The result of num_steps iterations will be passed as the initialization to 
        # the optimization targeting the next label in the sequence.
        inversions = iterated_inversion(
            self.model, x0, target_sequence, 
            step_size=step_size, 
            num_steps=num_iterations)
        self.inversions = inversions.numpy()

    def format_evaluation(self) -> None:
        inversions_path = (
            self.eval_dir_mgr.get_dir() / 
            '..' /  # Stupid hack to avoid the dummy dataset directory that gets created for the dataset eval we don't do.
            'inversions.npy')
        logger.info(f'Saving inversions to {str(inversions_path)}')
        np.save(str(inversions_path), self.inversions)
