Dataset Config Specification
==========

# Introduction

This document describes the json format for how to specify datasets
and their configuration for machine learning runs. This format allows 
one to specify image directories associated with a particular label for 
training and validation. The directories are assumed to be in a 
relative path from the same "data root."  The data root is not
specified in this file and assumed to be provided along with this
configuration file as part of the input.

# Schema
```
{
    "data_transforms": {
        "seed": < OPTIONAL int seed for randomization of the data transforms >
        "transforms": [ <array of transforms - see below> ]
    },
    "data_type": <Type of data [image | tabular | torchvision]>,
    "description": <OPTIONAL text description>,
    "format_version": <Linux style version string of the format of this file>,
    "url": Uniform Resource Locator, also known as the "link" to where the data is stored on the internet,
    "image_data": {
        "task_type": <Task to be performed [classification | objectDetection]>,
        "sources": [
            {
                "directory": <glob style path within the dataroot>,l
                "remove_image_ids": <OPTIONAL list of image ids to ignore>
                "label":<integer - the class label associated with the directory>,
                "root": <OPTIONAL: [ dataroot (default) | workspace | relative ]>,
                "sampling_count": <OPTIONAL number of files to use from this source>,
                "sampling_fraction": <OPTIONAL file fraction to use from this source> 
            }
        ]
    },
    "label_names" : {
        "<label int>": "<labelname>"
    }
    "num_model_classes": <int>,
    "sampling": {
        "algorithm": <'none', 'random_fraction', 'random_quantity', 'round_robin'>, 
        "arguments": <custom json structure depending on algorithm - see details>
    },    
    "tabular_data": {
        "sources": [
            {
                "root": <OPTIONAL: [ dataroot (default) | workspace | relative ]>,
                "path": <glob style path within the above defined root >
            }
        ],
        "label_index": <integer of the column of the labels>
    },
    "tensorflow_data": {
        "name": <The name of the tensorflow dataset, e.g. 'mnist'>,
        "load_kwargs": <OPTIONAL: kwargs to pass to the load function.>,
    }
    "timestamp": <OPTIONAL last modified - isoformat() with 0 microseconds>
    "torchvision_data": {
        "fqcn": <fully qualified class name e.g., torchvision.datasets.ImageNet>,
        "root": <branch within the data_root to be passed in as the root argument to the torchvision dataset class>,
        "train_kwargs": { <args to pass into the training instance except 'root', 'transform' and 'target_transform'>},
        "val_kwargs": { <args to pass into the validation instance except 'root', 'transform' and 'target_transform'>}
        "eval_kwargs": { <args to pass into the evaluation instance except 'root', 'transform' and 'target_transform'>}
    }
}
```

# Details
This section provides the details of each of the fields.

## data_transforms
**Optional** structure for specifying transforms to apply to the data.

### seed
**Optional** random seed to use to control the data transforms.

### transforms
An array of transforms that are applied to the data before either the "training_transforms" or
"evaluation_transforms" are applied. See the model configuration specification for a description of how 
to use transforms.

## data_type
The type of data this model is consuming. We support either 'image' or 'tabular'.

## description
**Optional** prose description of this dataset.

## format_version
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 0.2.0

## url
**Optional** If the data is stored on the internet, this is the link referencing where to download it.

## image_data
This section describes sources where image data can be found. 

### task_type
This field describes what task the images in the dataset are intended for. Task types currently 
supported: "classification" or "objectDetection".

### sources
This is an array of stanzas that describe directories that contain image data. That image 
data can be in the form of actual raw images or Juneberry metadata files that describe images.

#### directory
A glob style path relative to the data root containing image files or Juneberry 
metadata files that describe images. For example, the path ```mydataset/**/*.json```
would add all json files contained in the mydataset directory and its subdirectories. 

#### remove_image_ids
A list of image IDs that will be ignored when loading metadata.

#### label
The integer label associated with a directory of input files. Note, the label
does NOT need to be unique in the data list. Thus, two different data stanzas 
could have the same label, and the combined files from each stanza should be 
used for that label. This field will be ignored when the task_type is 
"objectDetection".

#### root
The base directory to use when applying the "directory". This can be one of "dataroot", "workspace", or
"relative".  
* dataroot - As defined in the Juneberry config.
* workspace - As defined in the Juneberry config.
* relative - A path relative to this configuration. Generally this is useful for files beside the config.

#### sampling_count
**Optional** A quantity of this set to be used.  Note, this is only used
when the sampling mechanism "randomQuantity" is used and this values overrides the default
value there.

#### sampling_fraction
**Optional** A fraction of this set to be used.  Note, this is only used
when the sampling mechanism "randomFraction" is used and this values overrides the default
value there.

## label_names
A dictionary of labels (integer as string) and the corresponding human-readable label name.

## num_model_classes
Number of classes present in this model. The dataset may omit some classes that are 
supported by the model. 

## sampling
This stanza explains how the images from that space shall be sampled.  The algorithm
type is specified in the random section, and the arguments stanza is provided
to the algorithm. Note, this selection is performed independently of any validation
strategy.

## tabular_data
Describes how to load the data when it is stored in tabular form. We support CSV files where the first line 
is the name of the variable. 

### sources
The array is a list of data source directories that will be scanned for files.

#### root
The base directory to use when applying the path below. This can be one of "dataroot", "workspace", or
"relative".  
* dataroot - As defined in the Juneberry config.
* workspace - As defined in the Juneberry config.
* relative - A path relative to this configuration. Generally this is useful for files beside the config.

#### path
A glob style path that is applied to the path as defined in "root" above. For example, if the root
is defined as "workspace" and this path is "models/my_model/data_*.csv", then all ".csv" files in the  
model directory "my_model" whose names begin with "data_" will be loaded.

### label_index
The label data is stored in the row as a particular column. This value is the index (0-based) of the
column that contains the labels.  Upon load, the labels are __extracted__ (removed from each row) and 
the subsequent row is passed to the transformer.

### algorithm
This can be:
* random_fraction - A random fraction is used to select a subset of images,
with optional random seed.
* random_quantity - A specific total quantity is pulled randomly from the images,
with optional random seed.
* round_robin - The dataset is split into groups selecting a specific group,
with optional random seed.  The same seed with different groups will produce a reproducible 
randomized split of a dataset.

### arguments
For **random_fraction**:
```
{
    "seed": <optional integer seed for the randomizer>
    "fraction": <decimal fraction, can be overriden by set specific fraction>
}
```

For **round_robin**:
```
{
    "seed": <optional (but recommended) integer seed for the randomizer>
    "groups": <number of groups to split dataset into>
    "position": <group to select from result in range [0 - (groups - 1)]>
}
```

For **random_quantity**:
```
{
    "seed": <optional integer seed for the randomizer>
    "count": <specific count of images to provide, can be overridden by set specific quantity>
}
```

## tensorflow_data

This section is used for TensorFlow based datasets, such as the ones loaded by the tensorflow-data package.
See (https://www.tensorflow.org/datasets) for details.

### name

The string name of the TensorFlow dataset, e.g. "mnist".

### load_kwargs

This is a dictionary of additional keyword args to be passed to TensorFlow's load function when loading the
dataset. ( https://www.tensorflow.org/datasets/api_docs/python/tfds/load )  Juneberry will automatically add 
the "as_supervised=True" to set the dataset to format it properly for the training. 
Some arguments such as `batch_size` and `shuffle` are stripped because Juneberry handles those aspects
with a different mechanism. 

**Split** requires special handling and follows these rules.

For Training:

- If "tensorflow" is specified as the validation algorithm in the *model config* then "split" **must** be
specified in `load_kwargs` as an array of 2 strings which are then used verbatim.
- If "random_fraction" is specified as the validation algorithm in the *model config* then:
    - If "split" is in the `load_kwargs` dictionary and the value is a single string, then it is used as the 
    basis to construct the split databases.
    - If "split" is not in the `load_kwargs` dictionary then it is defaults to "train". 

For Evaluation:

- If neither "use_evaluation_split" or "use_validation_split" are specified to `jb_evaluate` on the command line:
    - If "split" specified, then "split" is used directly.
    - If not specified, then "test" is used as the split value.
- If either "use_evaluation_split" or "use_validation_split" are specified then the the rules for training above
  are followed and the appropriate split is selected.

## timestamp
**Optional** Time stamp (ISO format with 0 microseconds) for when this file was last updated.

## torchvision_data"

This section is used to describe when the user wants to use a dataset that subclasses `torch.utils.data.Dataset`.
The class of the dataset must be importable and is specified in the `fqcn` property. The dataset
can be used in three different ways: during training, during validation, and during evaluation. Different
kwargs can be specified during each use case to configure different qualities of the dataset instance.
Any property can be specified except 'root', 'transform' and 'target_transform' because Juneberry provides
these automatically based on the runtime conditions.

### eval_kwargs
A set of kwargs to be passed into the **evaluation** instance, except for 'root', 'transform', and 'target_transform'.

### fqcn
Fully qualified class name of the torchvision dataset, such as `torchvision.datasets.ImageNet`.

### root
The branch within the data_root to be passed in as the root argument to the torchvision dataset class.

### train_kwargs
A set of kwargs to be passed into the **training** instance, except for 'root', 'transform', and 'target_transform'.

### val_kwargs
A set of kwargs to be passed into the **validation** instance, except for 'root', 'transform', and 'target_transform'.
**NOTE**: This is ONLY used when the model specifies a validation algorithm of "torchvision".

# History:
* 0.3.0 - Added tensorflow support.
* 0.2.0 - Big conversion to snake case in Juneberry 0.4.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
