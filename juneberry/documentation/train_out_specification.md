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
    "trainingTimes": {
        "startTime": <Time stamp for when the training began>,
        "endTime": <time stamp when the training ended>,
        "duration": <duration of training, in seconds>,
        "epochDurationSec": [
            <float>,
        ]
    },
    "trainingOptions": {
        "dataSetConfigPath": <path to data set>,
        "testDataSetConfigPath": <path to test data set>,
        "modelArchitecture": <path to model architecture>,,
        "colorspace": <range of colors in images, rgb, grayscale>,
        "dimensions": <pixel WidthxHeight of images>,
        "epochs": <int>,
        "batchSize": <int>,
        "seed": <int>,
        "numTrainingImages": <int>,
        "numValidationImages": <int>,
        "keras": {
            "optimizer": {
                "type": <adam, sgd, etc>,
                "learningRate": <learning_rate for optimizer>,
                "momentum": <momentum for SGD optimizer>,
                "nesterov": <apply Nesterov momentum for SGD optimizer>

            },
            "lossFunction": <Any keras loss function by name. e.g. sparse_categorical_crossentropy>
        }
    },
    "trainingResults": {
        "modelName": <name of model trained>,
        "modelHash": <hash of the model file produced>  
        "loss": [
            <float>,
        ],
        "accuracy": [
            <float>,
        ],
        "valLoss": [
            <float>,
        ],
        "valAccuracy": [
            <float>,
        ],
        "trainError": [
            <float>,
        ],
        "valError": [
            <float>,
        ],
        "testError": [
            <float>,
        ],
        "batchLoss": [
            <float>,
        ]
    }
    "formatVersion": <linux style version string of the format of this file>
}
```

# Details
This section provides the details of each of the fields.

## trainingTimes
This section provides times stamps that describe the training
### startTime
The time stamp at the beginning of the training
### endTime
The time stamp at after the training has finished
### duration
The time, in seconds, spent on the training. End Time - Start Time
### epochDurationSec
A sequential array the length in time it took to complete each epoch in the training.

## Training Options
These are the options that describe what and how the model was trained. 
Most of these options are lifted directly from the Data Configuration file, 
others are inferred from the data.
### dataSetConfigPath
The path to the data set configuration file.
### testDataSetConfigPath
**Optional** The path to a data set configuration file describing the test images. These 
test images are evaluated at the end of every epoch, and the test error rate 
will be added to train_out.json. 
### modelArchitecture
The path to the code that builds the model to be trained. 
### colorspace
The type of coloring used in the images. Examples are RGB and Gray Scale
### Dimensions
The dimensions in pixels of the images. Width x Height. Dimensions are uniform during training.
### epochs
Number of Epochs that were trained
### batchSize
Number of samples per gradient update
### Seed
Random seed used when processing this data set. The seed 
value affects the random shuffling of the image order, executing tensorflow, and
numpy. The same seed (this value) is used for all three aspects.
### numTrainingImages
Number of images used to train the model
### numValidationImages
Number of images used to validate the model
### keras
Specific parameters for the Keras model compilation
#### optimizer
The optimizer function with configuration values used.
##### type
The gradient descent strategy used during the training of the model. 
See training_configuration_specification.md for more details.
##### learningRate
Learning rate/s used during training. Learning rates will either be a single 
float or a dictionary with the keys representing epochs and the values representing 
the learning rate. See training_configuration_specification.md for more details.
##### momentum
The momentum provided to an SGD optimizer; a float >= 0.  
##### nesterov
A boolean indicating whether to apply Nesterov momentum to the optimizer.
#### lossFunction
The loss function used during the training of the model. 
See training_configuration_specification.md for more details.

## Training Results
The data profiling how the model trained.
### modelName
Name of the model
### modelHash
Hash (sha256) of the model file produced.
### loss
An array of floats representing the loss of the model on the training data for each epoch.
### accuracy
An array of floats representing the accuracy of the model on the training data for each epoch.
### valLoss
An array of floats representing the loss of the model on the validation data for each epoch.
### valAccuracy
An array of floats representing the accuracy of the model on the validation data for each epoch.
### trainError
An array of floats representing the error of the model on the training data for each epoch.
### valError
An array of floats representing the error of the model on the validation data for each epoch.
### testError
An array of floats representing the error of the model on the test data for each epoch.
When the option is turned off, this field will be an empty array [].
### batchLoss
An array of floats representing the loss of the model on the training data for each batch. 
There are 1 or more batches per epoch, so this array's size will be equal to or greater than the loss array.
When the option is turned off, this field will be an empty array [].

## formatVersion
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 1.2.0

* 1.2.0 - modelFile became modelName. Added in model Hash.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
 