Evaluation Output Specification
==========


# Introduction

This document describes the json format for how Juneberry saves the evaluation outputs from a particular model and 
dataset pair.

# Schema
```
{
    "format_version": <linux style version string of the format of this file>,
    "options": {
        "dataset": {
            "classes": <dictionary of the integer-based labeling scheme for the classes>,
            "config": <path to the dataset config file>,
            "histogram": <dictionary of classes and their frequencies>,
        }
        "model": {
            "hash": <string of the model hash value>,
            "name": <path to a sub-directory within 'models' containing a model config file>,
            "num_classes": <integer of the number of classes>
        }
    },
    "results": {
        "classifications": [
            <list of dictionaries containing classification information>
        ]
        "labels": [ 
            <array of actual integer test labels>
        ],
        "metrics": {
            "accuracy": <float of the model accuracy>,
            "balanced_accuracy": <float of the model's balanced accuracy>,
            "bbox": <dictionary of mAP scores>,
            "bbox_per_class": <dictionary of per-class mAP scores>
        },
        "predictions": [
            [
                <confidence of each class, one entry for each class>
            ]
        ]
    },
    "times": {
        "duration": <duration in seconds>,
        "end_time": <iso format time>,
        "start_time": <iso format time>
    },
}    
```

# Detail

## format_version
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 1.1.0

## options
A stanza describing the options that were provided from the test set. This data should match 
that from the test dataset and is simply here for convenience.

### dataset
A stanza containing information about the dataset.
#### classes
The labeling scheme for the dataset. Integer keys are mapped to human-readable string values.
#### config
The path to the dataset config file.
#### histogram
A histogram of classes and their frequencies represented as a dictionary.

### model
A stanza containing information about the model.
#### hash
The model hash value.
#### name
The path to a sub-directory within the 'models' directory that contains the model config.json for
the model that was evaluated.
#### num_classes
The number of classes the model is aware of.

## results
This section contains the true labels of the input data, some metrics, and the predictions for the test data.

### classifications
This is a list of entries containing the following classification information:
* The actual label and label name
* The path to the file that was evaluated
* A list of the top k predicted classes (specified by label and label name) with their corresponding confidence values

### labels
This is an array of the correct labels for the test data.

### metrics
A stanza reporting the metrics from evaluation.
#### accuracy
Float of the accuracy of the model.
#### balanced_accuracy
Float of the balanced accuracy of the model.
#### bbox
A dictionary of mAP scores.
#### bbox_per_class
A dictionary of per-class mAP scores.

### predictions
This is a list of predictions of which class each piece of data belongs to. Every piece of data 
in the test set will have a list of prediction values. The length of the list of predictions 
is equal to the number of possible classes. 

## times
This section describes the timing information about the evaluation.

### duration
Duration of the evaluation, in seconds.
### end_time
End time of the evaluation in iso format.
### start_time
Start time of the evaluation in iso format.

    
# Version History

* 1.1.0 - modelFile became modelName. Added in model Hash.

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.
 
