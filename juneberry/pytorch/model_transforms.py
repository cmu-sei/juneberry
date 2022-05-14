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

#class Preprocess(torch.nn.Module):
#    """Module to perform pre-process using Kornia on torch tensors."""
#
#    def forward(self, x):
#        import pdb; pdb.set_trace()
#        x_tmp = np.array([x[,1,:],x[2,:],x[0,:]])  # HxWxC
#        x_out = kornia.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
#        print(f"{x_tmp.shape} {x_out.shape}")
#        
#        return x_out.float() / 255.0

class TensorPatch(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        # Note that torch.rand returns values between 0 and 1.
        self.patch = torch.nn.Parameter( (torch.zeros(shape, dtype=torch.float, requires_grad=True) )) 
        # Try a standard normal as an init
        # self.patch = torch.nn.Parameter( torch.normal( mean=0, std=1, size=shape, requires_grad=True) )
        self.shape = shape

        print(f"Init R Patch min: {self.patch[0,:,:].min()} Patch max: {self.patch[0,:,:].max()}") 
        print(f"Init G Patch min: {self.patch[1,:,:].min()} Patch max: {self.patch[1,:,:].max()}") 
        print(f"Init B Patch min: {self.patch[2,:,:].min()} Patch max: {self.patch[2,:,:].max()}") 
        
    def clamp_to_zero_one_imagenet_norms(self):
        """
        Note the standard ImageNet normalization is 
            "mean": [ 0.485, 0.456, 0.406 ],
            "std": [ 0.229, 0.224, 0.225]
        So a tensor, normalized from 0 to 1 will have bounds for the first channel of 
            ( 0 - 0.485 ) / 0.229 = -2.1179039301310043
            ( 1 - 0.485 ) / 0.229 = 2.2489082969432315
        and so on.
        """
        self.patch.data[0, :, :].clamp_(min=( 0 - 0.485 ) / 0.229, max=( 1 - 0.485 ) / 0.229)
        self.patch.data[1, :, :].clamp_(min=( 0 - 0.456 ) / 0.224, max=( 1 - 0.456 ) / 0.224)
        self.patch.data[2, :, :].clamp_(min=( 0 - 0.406 ) / 0.225, max=( 1 - 0.406 ) / 0.225 )

    def un_normalize_imagenet_norms(self, x):
        x_r = x[0, :, :] * 0.229 + 0.485  
        x_g = x[1, :, :] * 0.224 + 0.456  
        x_b = x[2, :, :] * 0.225 + 0.406 

        return (torch.stack([x_r, x_g, x_b]).clamp_(min=0, max=1))

    def save_patch(self, path):

        # Extra clamp to catch the last optimizer.step()
        # N.B. This doesn't normally get called because of the evaluation forward pass.
        # print(f"Pre-clamp R Patch min: {self.patch[0,:,:].min()} Patch max: {self.patch[0,:,:].max()}") 
        # print(f"Pre-clamp G Patch min: {self.patch[1,:,:].min()} Patch max: {self.patch[1,:,:].max()}") 
        # print(f"Pre-clamp B Patch min: {self.patch[2,:,:].min()} Patch max: {self.patch[2,:,:].max()}") 
        self.clamp_to_zero_one_imagenet_norms()

        # Un-normalize back to [0,1)
        patch_zero_one =  self.un_normalize_imagenet_norms( self.patch.detach().cpu() )
        # print(f"R Patch min: {patch_zero_one[0,:,:].min()} Patch max: {patch_zero_one[0,:,:].max()}") 
        # print(f"G Patch min: {patch_zero_one[1,:,:].min()} Patch max: {patch_zero_one[1,:,:].max()}") 
        # print(f"B Patch min: {patch_zero_one[2,:,:].min()} Patch max: {patch_zero_one[2,:,:].max()}") 

        # Permute to PIL's channel last, order, stretch to [0,255), cast to a numpy uint8 array, and return as an image 
        patch_image = Image.fromarray( np.array(patch_zero_one.permute(1,2,0) * 255  , dtype=np.uint8 ) ) 

        # Save the patch
        patch_image.save(path)

    def forward(self, x):
        self.clamp_to_zero_one_imagenet_norms()
        return(self.patch)

class TopLeftPatch(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.patch = TensorPatch(shape)

    def forward(self, x):
        patched_x = x.clone()    
        rgb_patch = self.patch(x)
        patched_x[:, :, 0:self.patch.shape[1], 0:self.patch.shape[2]] = rgb_patch.repeat(x.shape[0],1,1,1)

        #print(f"batch mean: {x.mean()} batch std: {x.std()}")
        #print(f"patch mean: {self.patch.patch.mean()}, patch std: {self.patch.patch.std()}")

        return patched_x

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


class BrownEtAlPatch(torch.nn.Module):
    def __init__(self, patch_shape, scale_max, scale_min, rotate_max, rotate_min):
        super().__init__()
        self.patch = TensorPatch(patch_shape)
        self.mask = torch.tensor( _circle_mask( shape = [patch_shape[1], patch_shape[2], patch_shape[0]])).permute(2,0,1)
        self.patch_shape = patch_shape
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.rotate_max = rotate_max
        self.rotate_min = rotate_min

    def forward(self, x):
        # You have to copy the batch of images if you use subsetting to apply. Otherwise pytorch can't update the gradients
        # patched_x = x.clone() 

        # Repeat the patch across the batches
        batch_patches = (self.patch(x) * self.mask.to(device=x.device)).repeat(x.shape[0],1,1,1)

        # Pick the scale
        # NOTE: This is the same for all patches
        scale = torch.rand(1, device = x.device) * (self.scale_max - self.scale_min) + self.scale_min
        scale_patch_width = (self.patch_shape[1]*scale).to(int).item()
        scale_patch_height = (self.patch_shape[2]*scale).to(int).item()

        # Pick the random location to apply the patch
        if x.shape[2] - scale_patch_width > 0 and x.shape[3] - scale_patch_height > 0 :
            u_t = torch.randint(low=0, high=x.shape[2] - scale_patch_width, size=(1,), device = x.device)
            v_t = torch.randint(low=0, high=x.shape[3] - scale_patch_height, size=(1,), device = x.device)
        else:
            u_t = 0
            v_t = 0

        # Place the patch in a zero image 
        padded_batch_patches = torch.zeros(x.shape, device = x.device)
        batch_patches = kornia.geometry.transform.resize(batch_patches, (scale_patch_width,scale_patch_height), align_corners=False)
        padded_batch_patches[:, :, u_t:(u_t+scale_patch_width), v_t:(v_t+scale_patch_height)] = batch_patches
        # print(f"{u_t} {v_t}")

        #batch_patches = kornia.geometry.transform.scale(batch_patches, scale )
        #padded_batch_patches[:, :, u_t:(u_t+batch_patches.shape[2]), v_t:(v_t+batch_patches.shape[3])] = batch_patches

        # Generate the mask
        mask = torch.zeros(x.shape, device = x.device)
        mask_border = 3
        small_ones = torch.ones([x.shape[0], x.shape[1], scale_patch_width-2*mask_border, scale_patch_height-2*mask_border], device = x.device)
        mask[:, :, u_t+mask_border:(u_t+scale_patch_width-mask_border), v_t+mask_border:(v_t+scale_patch_height-mask_border)] = small_ones 

        # Scale every patch the same in a given batch
        padded_batch_patches = kornia.geometry.transform.scale(padded_batch_patches, scale )
        mask = kornia.geometry.transform.scale(mask, scale )

        # Pick the angle to rotate, and rotate the patches
        angles = torch.rand(x.shape[0], device = x.device) * (self.rotate_max - self.rotate_min) + self.rotate_min
        padded_batch_patches = kornia.geometry.rotate( padded_batch_patches, angles)
        mask = kornia.geometry.rotate( mask, angles)

        # Apply the patch as an overlay
        patched_x = x * (1 - mask) + padded_batch_patches * (mask)

        # TODO this doesn't look quite right
        #import pdb; pdb.set_trace()
        #for b in range(x.shape[0]):
        #    print(b)
        #    image_zero_one =  self.patch.un_normalize_imagenet_norms( patched_x[b].detach().cpu() )
        #    debug_image = Image.fromarray( np.array(image_zero_one.permute(1,2,0) * 255  , dtype=np.uint8 ) )
        #    debug_image.save(f"{b}.png")

        return patched_x


class ApplyBrownEtAlPatch:
    def __init__(self, patch_shape, scale_max, scale_min, rotate_max, rotate_min):
        self.patch = BrownEtAlPatch(patch_shape, scale_max, scale_min, rotate_max, rotate_min)
    
    def __call__(self, model):
        model = torch.nn.Sequential( self.patch, model)
        return model

class ApplyPatch:
    def __init__(self, shape):
        self.patch = TopLeftPatch(shape)

    def __call__(self, model):
        model = torch.nn.Sequential( self.patch, model)
        return model

class DataAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = torch.nn.Sequential(
            kornia.augmentation.RandomHorizontalFlip(p=0.75),
            kornia.augmentation.RandomChannelShuffle(p=0.75),
            kornia.augmentation.RandomThinPlateSpline(p=0.75),
        )

        self.jitter = kornia.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)

    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out



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
