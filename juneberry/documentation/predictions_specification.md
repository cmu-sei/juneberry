Predictions Specification
==========


# Introduction

This document describes the json format for how we save the predictions results
from a particular model data set pair.

# Schema
```
{
    "testTimes": {
        "imgProcessingStartTime": <iso format time>,
        "imgProcessingEndTime": <iso format time>,
        "imgProcessingDuration": <duration in seconds>,
        "predictionStartTime": <iso format time>,
        "predictionEndTime": <iso format time>,
        "predictionDuration": <duration in seconds>
    },
    "testOptions": {
        "numModelClasses":  <int>,
        "testImages": <path within experiment root to input data set>,
        "colorspace": <colorspace [ 'rgb' | 'gray' ]>,
        "dimensions": <dimensions in "width,height">,
        "modelName": <name of the model read>,
        "modelHash": <hash of the model file read>,
        "mapping": { 
            <label as string>: <class name as string>
        }
    },
    "testResults": {
        "labels": [ 
            <array of actual integer test labels>
        ],
        "predictions": [
            [
                <confidence of each class, one entry for each class>
            ]
        ]
    }
    "formatVersion": <linux style version string of the format of this file>
}    
```

# Detail

## testTimes
This section describes the times taken by processing. If the image processing is done independently
of the evaluation time, then those times are reported separately. If the image loading is done
during evaluation, then only one aggregate is reported as prediction times.

### imgProcessingStartTime
Optional start time of the image processing in iso format, if image processing is done first.

### imgProcessingEndTime
Optional end time of the image processing in iso format, if image processing is done first.

### imgProcessingDuration
Optional duration of image processing in seconds, if image processing is done first.

### predictionStartTime
Start time of the prediction process in iso format.

### predictionEndTime
End time of the prediction process in iso format.

### predictionDuration
Duration of prediction process in seconds.

## testOptions
A stanza describing the options that were provided from the test set. This data should match 
that from the test data set and is simply here for convenience.

## numModelClasses
Number of classes present in this model. Some classes may be omitted for this dataset.

### testImages
Path within experiment root to input data set.

### colorspace
The colorspace of the test images.

### dimensions
The dimensions of the images in "width,height".

### modelName
The name of the trained model used for the evaluation.

### modelHash
The hash of the model file used for the training.

### mapping
A mapping of integer labels to string names.
    
## testResults
This section contains the correct labels and predictions for the test images.

### labels
This is an array of the correct labels for the test images.

### predictions
This is a list of predictions of which class each image belongs to. Every image 
in the test set will have a list of prediction values. The length of this prediction 
list is equal to the number of possible classes. 

## formatVersion
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 1.1.0

* 1.1.0 - modelFile became modelName. Added in model Hash.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
 
