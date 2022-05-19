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
from pathlib import Path
import sys

import torch
from torchsummary import summary
import kornia 
import numpy as np

from PIL import Image

from torchvision.transforms import functional as F

import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class LoadModel:
    """
    Loads models from existing files or an URL.
    """

    def __init__(self, *, modelName=None, modelPath=None, modelURL=None, strict=True, loadKeyPath=None,
                 includePatterns=None, excludePatterns=None, renamePatterns=None):
        self.model_name = modelName
        self.model_path = modelPath
        self.model_url = modelURL
        self.strict = strict

        # We accept None (don't do), single argument, or lists of arguments.
        # Internally we like lists of arguments so promote if single argument.
        self.load_key_path = _ensure_list(loadKeyPath)
        self.include_patterns = _ensure_list(includePatterns)
        self.exclude_patterns = _ensure_list(excludePatterns)
        self.rename_patterns = _ensure_list_of_lists(renamePatterns)

        if self.model_name is None and self.model_path is None and self.model_url is None:
            logger.error("The LoadModel model transform requires either a modelName, modelPath, or modelURL. EXITING.")
            sys.exit(-1)

    def __call__(self, model):

        state_dict = {}
        if self.model_name is not None:
            logger.info(f"LoadModel transform loading model: {self.model_name}.")
            # Construct a model manager so we can get the model path.
            model_manager = jbfs.ModelManager(self.model_name)
            model_path = model_manager.get_pytorch_model_path()
            if not Path(model_path).exists():
                logger.error(f"Model path {model_path} does not exist! EXITING.")
                sys.exit(-1)
            state_dict = torch.load(str(model_path))
        elif self.model_path is not None:
            logger.info(f"LoadModel transform loading model from Path: {self.model_path}.")
            if not Path(self.model_path).exists():
                logger.error(f"Model path {self.model_path} does not exist! EXITING.")
                sys.exit(-1)
            state_dict = torch.load(self.model_path)
        elif self.model_url is not None:
            logger.info(f"LoadModel transform loading model from URL: {self.model_url}.")
            state_dict = torch.hub.load_state_dict_from_url(self.model_url, progress=False)

        # The first step is to dig down to the path they want.
        if self.load_key_path is not None:
            for key in self.load_key_path:
                state_dict = state_dict[key]

        # DESIGN NOTE: This design focuses on identifying the keys we actually want,
        # then pulling over just that data to operations
        keep_map = self.filter_keys(state_dict.keys())
        logger.info(f"LoadModel loading keys: {keep_map}")

        # Construct the new state dict from the old one
        state_dict = {keep_map[k]: state_dict[k] for k in keep_map.keys()}

        # Shove whatever we have into the state dict of the model
        model.load_state_dict(state_dict, strict=self.strict)

        return model

    def filter_keys(self, input_keys):
        """
        Filters the input key set based on the patterns that were set during construction.
        We expose this for unit testing.
        :param input_keys: The keys to filter.
        :return: The filtered keys as a map of input key name to kept key name.
        """
        # NOTE: We are order preserving, so we can't use sets.
        keep_keys = [*input_keys]

        # Filter based on a positive include pattern, if we have any
        if self.include_patterns is not None:
            keep_keys = [k for k in keep_keys for p in self.include_patterns if p in k]

        # Move the keys to a new keys if we do NOT exclude
        if self.exclude_patterns is not None:
            tmp_keys = keep_keys
            keep_keys = []
            for key in tmp_keys:
                found_it = False
                for p in self.exclude_patterns:
                    if p in key:
                        found_it = True
                        break
                if not found_it:
                    keep_keys.append(key)

        # Build a key conversion map of names
        keep_map = {k: k for k in keep_keys}
        if self.rename_patterns is not None:
            for k in keep_keys:
                new_name = k
                for p in self.rename_patterns:
                    new_name = new_name.replace(p[0], p[1])

                keep_map[k] = new_name

        return keep_map


class SaveModel:
    """
    A transform for saving the model to a file specified by model name or file path.
    """

    def __init__(self, *, modelName=None, modelPath=None, overwrite=False):
        self.model_name = modelName
        self.model_path = modelPath
        self.overwrite = overwrite

        if self.model_name is None and self.model_path is None:
            logger.error("The SaveModel model transform requires either a modelName or modelPath. EXITING.")
            sys.exit(-1)

    def __call__(self, model):
        if self.model_name is not None:
            logger.info(f"SaveModel transform saving model: {self.model_name}.")
            # Construct a model manager so we can get the model path.
            model_manager = jbfs.ModelManager(self.model_name)
            model_dir = model_manager.get_model_dir()
            if not Path(model_dir).exists():
                logger.error(f"Model directory {model_dir} does not exist! EXITING.")
                sys.exit(-1)

            model_path = model_manager.get_pytorch_model_path()
            if Path(model_path).exists() and not self.overwrite:
                logger.error(
                    f"Model file already exists and 'overwrite' setting is False. Model: {model_path} EXITING.")
                sys.exit(-1)

            torch.save(model.state_dict(), model_path)

        elif self.model_path is not None:
            logger.info(f"LoadModel transform loading model from Path: {self.model_path}.")
            if Path(self.model_path).exists() and not self.overwrite:
                logger.error(
                    f"Model file already exists and 'overwrite' setting is False. Model: {self.model_path} EXITING.")
                sys.exit(-1)

            torch.save(model.state_dict(), self.model_path)
        
        return model


class LogModelSummary:
    """
    Transform used to output the model summary to the console for debugging.
    """

    def __init__(self, image_shape):
        """
        Logs the image summary.
        :param image_shape: Shape in a tuple of C, W, H
        """
        # Can we make up an image shape?
        self.image_shape = image_shape

    def __call__(self, model):
        orig = sys.stdout
        sys.stdout.write = logger.info
        with torch.no_grad():
            summary(model, self.image_shape)
        sys.stdout = orig
        return model


class ReplaceFC:
    """
    A transform for replacing the fully connected layer. Useful for pre-trained models.
    """    

    def __init__(self, num_classes, fc_name='fc', fc_bias=True):
        self.num_classes = num_classes
        self.fc_name = fc_name
        self.fc_bias = fc_bias
    
    def __call__(self, model):
        original_layer = getattr(model, self.fc_name)
        in_features = original_layer.in_features
        new_layer = torch.nn.modules.linear.Linear(in_features=in_features, out_features=self.num_classes,
                                                   bias=self.fc_bias)
        setattr(model, self.fc_name, new_layer)
        
        return model

class Freeze:
    """
    Transform used to freeze a pre-trained model
    """

    def __init__(self):
        pass

    def __call__(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

class PILPatch(torch.nn.Module):
    def __init__(self, shape, mask = 'circle'):
        super().__init__()
        # Init as a gray patch 
        self.patch = torch.nn.Parameter( (torch.ones(shape, dtype=torch.float, requires_grad=True))* 0.5 )
        self.model_path = None

        if mask is None:
            mask = torch.ones(shape) 
        elif mask == 'circle':
            mask = torch.tensor( _circle_mask( shape = [shape[1], shape[2], shape[0]])).permute(2,0,1) 
        else:
            raise RuntimeError(f"Mask type {mask} not yet implemented.")

        # Note, this creates self.mask
        self.register_buffer('mask', mask)

    def clamp_to_valid_image(self):
        self.patch.data.clamp_(min=0, max=1)

    def save_patch(self, model_path, filename):
        self.clamp_to_valid_image()
        self.model_path = model_path
        # Permute to PIL's channel last, order, cast to a numpy uint8 array, and return as a instance of a PIL Image 
        patch_image = Image.fromarray( np.array( (self.patch * self.mask).detach().cpu().permute(1,2,0) * 255, dtype=np.uint8 ) ) 

        # Save the patch
        patch_image.save(f"{model_path}.{filename}")

    def forward(self, x):
        # Clamp the patch
        self.clamp_to_valid_image()
        # Repeat the patch and mask across the batches of x and return
        batched_patch = (self.patch * self.mask).repeat( x.shape[0], 1, 1, 1)
        batched_mask = (self.mask).repeat( x.shape[0], 1, 1, 1)

        return( batched_patch , batched_mask)

class PatchLayer(torch.nn.Module):
    def __init__(self, patch, patch_transforms=None, image_transforms=None):
        super().__init__()
        self.patch = patch

        self.degree_max = 0
        self.degree_min = 0
        if 'degree_range' in patch_transforms:
            self.degree_min = patch_transforms['degree_range'][0] 
            self.degree_max = patch_transforms['degree_range'][1]

        self.scale_min = 1
        self.scale_max = 1
        if 'scale_range' in patch_transforms:
            self.scale_min = patch_transforms['scale_range'][0] 
            self.scale_max = patch_transforms['scale_range'][1]

        self.translate_proportion = 0
        if 'translate_proportion' in patch_transforms:
            self.translate_proportion = patch_transforms['translate_proportion']

        self.forward_counter = 0 
        self.debug_interval = None
        if 'debug' in patch_transforms:
            self.debug_interval = patch_transforms['debug']['interval']

    def cat_four(self,x):
        return( torch.cat( ( torch.cat([ x[0], x[1] ], 1 ),  torch.cat([ x[2], x[3] ], 1 ) ), 2)  )

    def save_debug_images(self, x, path, name="batch "):
        if x.shape[0] >= 16:
            quad0 = self.cat_four( x[0:4] )
            quad1 = self.cat_four( x[4:8] )
            quad2 = self.cat_four( x[8:12] )
            quad3 = self.cat_four( x[12:16] )

            four_quads = self.cat_four( torch.stack( (quad0, quad1, quad2, quad3)) )

            debug_image = Image.fromarray( np.array( four_quads.detach().cpu().permute(1,2,0) * 255, dtype=np.uint8 ) ) 
            debug_image.save(f"{path}.{self.debug_interval}.{name}-four-quads.png")
        else: 
            for b in range(min(x.shape[0],16)):
                print(b)
                debug_image = Image.fromarray( np.array( x[b].detach().cpu().permute(1,2,0) * 255, dtype=np.uint8 ) ) 
                debug_image.save(f"{path}.{name}.{b}.png")


    def forward(self, x):
        self.forward_counter = self.forward_counter + 1

        # Grab the current patch, replicated across the batches of x
        patch, mask = self.patch(x)

        # Define the transforms
        # TODO: Why do these have to be created on the forward pass to work? 
        self.patch_transforms = kornia.augmentation.container.AugmentationSequential( 
            kornia.augmentation.RandomAffine( degrees=(self.degree_min, self.degree_max), scale = (self.scale_min, self.scale_max) ),
            kornia.augmentation.RandomCrop( size=(x.shape[2], x.shape[3]), pad_if_needed=True, cropping_mode='resample'),
            kornia.augmentation.RandomAffine( degrees=0, translate=(self.translate_proportion,self.translate_proportion)),
            data_keys=["input", "mask"]
        )
        self.image_transforms = kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.Normalize( mean=torch.tensor( (0.485, 0.456, 0.406) ), std=torch.tensor( (0.229, 0.224, 0.225)))
        )

        # Transform the patches and mask (each element of a batch should have different transforms)
        patch, mask = self.patch_transforms(patch, mask)

        # Apply the patch to the batch
        x = x * (1 - mask) + patch * mask

        if self.debug_interval is not None and self.forward_counter % (self.debug_interval * 2) == 0 :
            with torch.no_grad():
                self.save_debug_images(x, self.patch.model_path, "patched-x")

        # Tranform the patched images
        x = self.image_transforms(x)

        return( x )

class EvalTail(torch.nn.Module):
    def __init__(self, trainable_model, eval_model):
        super().__init__()
        self.trainable_model = trainable_model
        self.eval_model = eval_model 
    
    def forward(self, x):
        x = self.trainable_model(x)
        self.eval_model.eval()
        x = self.eval_model(x)
        
        return(x)


class ApplyPatchLayer:
    def __init__(self, patch, patch_transforms, image_transforms = None):

        if 'shape' in patch and 'mask' in patch: 
            self.patch = PILPatch(patch['shape'], patch['mask'])
        else :
            raise RuntimeError(f"Shape and mask must be keys of patch. Got patch = {patch}")

        # Note: weird error if AugmentationSequential not created inside nn.Module
        self.patch_layer = PatchLayer(self.patch,patch_transforms,image_transforms)

    def __call__(self, model):
        model = EvalTail( self.patch_layer, model)
        return model

def _circle_mask(shape, sharpness = 40):
  """Return a circular mask of a given shape"""
  assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

  diameter = shape[0]  
  x = np.linspace(-1, 1, diameter)
  y = np.linspace(-1, 1, diameter)
  xx, yy = np.meshgrid(x, y, sparse=True)
  z = (xx**2 + yy**2) ** sharpness

  mask = 1 - np.clip(z, -1, 1)
  mask = np.expand_dims(mask, axis=2)
  mask = np.broadcast_to(mask, shape).astype(np.float32)
  return mask

class EmptyTransform:
    def __init__(self):
        pass

    def __call__(self, model):
        return model


# Utilities
def _ensure_list(args):
    if args is None:
        return None

    if isinstance(args, list):
        return args

    return [args]


def _ensure_list_of_lists(args):
    if args is None:
        return None

    if isinstance(args[0], list):
        return args

    return [args]
