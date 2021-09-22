Training Output
===============

# Introduction

This document describes the JSON format that is outputted after successfully
training a model. Information on when and how the training was performed is 
present. This format contains a reference to a dataset, a set of parameters
for training the architecture and training metadata. The configuration files
are relative to an externally provide configuration directory.
The model path is assumed to be with the import space. (e.g. relative to cwd or 
PYTHONPATH.)

# Schema
```
{
    "format_version": <linux style version string of the format of this file>,
    "options": {
        "training_dataset_config_path": <path to data set>,
        "validation_dataset_config_path": <optional path to dataset used for validation with 'from_file' option>,
        "model_architecture": <the model_architecture stanza from the model file>
        "colorspace": <range of colors in images, rgb, grayscale>,
        "dimensions": <pixel WidthxHeight of images>,
        "epochs": <int>,
        "batch_size": <int>,
        "seed": <int>,
        "num_training_images": <int>,
        "num_validation_images": <int>,
        "keras": {
            "optimizer": {
                "type": <adam, sgd, etc>,
                "learning_rate": <learning_rate for optimizer>,
                "momentum": <momentum for SGD optimizer>,
                "nesterov": <apply Nesterov momentum for SGD optimizer>
            },
            "loss_function": <Any keras loss function by name. e.g. sparse_categorical_crossentropy>
        }
    },
    "results": {
        "model_name": <name of model trained>,
        "model_hash": <hash of the model file produced>,
        "onnx_model_hash": <hash of the onnx model file produced>,
        "loss": [
            <float>,
        ],
        "accuracy": [
            <float>,
        ],
        "val_loss": [
            <float>,
        ],
        "val_accuracy": [
            <float>,
        ],
        "train_error": [
            <float>,
        ],
        "val_error": [
            <float>,
        ],
        "test_error": [
            <float>,
        ],
        "batch_loss": [
            <float>,
        ]
    }
    "times": {
        "start_time": <Time stamp for when the training began>,
        "end_time": <time stamp when the training ended>,
        "duration": <duration of training, in seconds>,
        "epoch_duration_sec": [
            <float>,
        ]
    },
}
```

# Details
This section provides the details of each of the fields.

## format_version
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 1.2.0

## options
These are the options that describe what and how the model was trained. 
Most of these options are lifted directly from the Data Configuration file, 
others are inferred from the data.

### training_dataset_config_path
The path to the data set configuration file.

### validation_dataset_config_path
**Optional** If the 'from_file' option is used for validation, this is the path to that file. 

### model_architecture
The model_architecture stanza from the model file. 

### colorspace
The type of coloring used in the images. Examples are RGB and Gray Scale.

### Dimensions
The dimensions in pixels of the images. Width x Height. Dimensions are uniform during training.

### epochs
Number of Epochs that were trained.

### batch_size
Number of samples per gradient update.

### Seed
Random seed used when processing this data set. The seed 
value affects the random shuffling of the image order, executing tensorflow, and
numpy. The same seed (this value) is used for all three aspects.

### num_training_images
Number of images used to train the model.

### num_validation_images
Number of images used to validate the model.

### keras
Specific parameters for the Keras model compilation.

#### optimizer
The optimizer function with configuration values used.

##### type
The gradient descent strategy used during the training of the model. 
See training_configuration_specification.md for more details.

##### learning_rate
Learning rate/s used during training. Learning rates will either be a single 
float or a dictionary with the keys representing epochs and the values representing 
the learning rate. See training_configuration_specification.md for more details.

##### momentum
The momentum provided to an SGD optimizer; a float >= 0.  

##### nesterov
A boolean indicating whether to apply Nesterov momentum to the optimizer.

#### loss_function
The loss function used during the training of the model. 
See training_configuration_specification.md for more details.

## results
The data profiling how the model trained.

### model_name
Name of the model.

### model_hash
Hash (sha256) of the model file produced.

### onnx_model_hash
Hash (sha256) of the onnx model file produced.

### loss
An array of floats representing the loss of the model on the training data for each epoch.

### accuracy
An array of floats representing the accuracy of the model on the training data for each epoch.

### val_loss
An array of floats representing the loss of the model on the validation data for each epoch.

### val_accuracy
An array of floats representing the accuracy of the model on the validation data for each epoch.

### train_error
An array of floats representing the error of the model on the training data for each epoch.

### val_error
An array of floats representing the error of the model on the validation data for each epoch.

### test_error
An array of floats representing the error of the model on the test data for each epoch.
When the option is turned off, this field will be an empty array [].

### batch_loss
An array of floats representing the loss of the model on the training data for each batch. 
There are 1 or more batches per epoch, so this array's size will be equal to or greater than the loss array.
When the option is turned off, this field will be an empty array [].

## times
This section provides times stamps that describe the training.

### start_time
The time stamp at the beginning of the training.

### end_time
The time stamp at after the training has finished.

### duration
The time, in seconds, spent on the training. End Time - Start Time.

### epoch_duration_sec
A sequential array the length in time it took to complete each epoch in the training.

# Version History

* 0.2.0 - Big conversion to snake case in Juneberry 0.4.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
 