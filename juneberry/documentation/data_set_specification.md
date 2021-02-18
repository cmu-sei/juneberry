Data Set Specification
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
    "dataType": <Type of data [image | tabular]>
    "numModelClasses": <int>,
    "labelNames" : {
        "<label int>": "<labelname>"
    }
    "imageData": {
        "taskType": <Task to be performed [classification | objectDetection]>
        "sources": [
            {
                "directory": <glob style path within the dataroot>,
                "label":<integer - the class label associated with the directory>,
                "samplingCount": <OPTIONAL number of files to use from this source>,
                "samplingFraction": <OPTIONAL file fraction to use from this source> 
            }
        ]
    },
    "tabularData": {
        "sources": [
            {
                "root": <OPTIONAL: [ dataroot (default) | workspace | relative ]>
                "path": <glob style path within the above defined root >
            }
        ]
        "labelIndex": <integer of the column of the labels>
    },
    "sampling": {
        "algorithm": <'none', 'randomFraction', 'randomQuantity', 'roundRobin'> 
        "arguments": <custom json structure depending on algorithm - see details>
    },    
    "description": <OPTIONAL text description>
    "timestamp": <OPTIONAL last modified - isoformat() with 0 microseconds>
    "formatVersion": <Linux style version string of the format of this file>
}
```

# Details
This section provides the details of each of the fields.

## dataType
The type of data this model is consuming. We support either 'image' or 'tabular'.

## numModelClasses
Number of classes present in this model. The dataset may omit some classes that are 
supported by the model. 

## labelNames
A dictionary of labels (integer as string) and the corresponding human-readable label name.

## imageData
This section describes sources where image data can be found. 

### taskType
This field describes what task the images in the dataset are intended for. Task types currently 
supported: "classification".

### sources
This is an array of stanzas that describe directories that contain image data. That image 
data can be in the form of actual raw images or Juneberry metadata files that describe images.

#### directory
A glob style path relative to the data root containing image files or Juneberry 
metadata files that describe images. For example, the path ```mydataset/**/*.json```
would add all json files contained in the mydataset directory and its subdirectories. 

#### label
The integer label associated with a directory of input files. Note, the label
does NOT need to be unique in the data list. Thus, two different data stanzas 
could have the same label, and the combined files from each stanza should be 
used for that label. This field will be ignored when the taskType is 
"objectDetection".

#### samplingCount
**Optional** A quantity of this set to be used.  Note, this is only used
when the sampling mechanism "randomQuantity" is used and this values overrides the default
value there.

#### samplingFraction
**Optional** A fraction of this set to be used.  Note, this is only used
when the sampling mechanism "randomFraction" is used and this values overrides the default
value there.

## tabularData
Describes how to load the data when it is stored in tabular form.

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

### labelIndex
The label data is stored in the row as a particular column. This value is the index (0-based) of the
column that contains the labels.  Upon load, the labels are __extracted__ (removed from each row) and 
the subsequent row is passed to the transformer.

## sampling
This stanza explains how the images from that space shall be sampled.  The algorithm
type is specified in the random section, and the arguments stanza is provided
to the algorithm. Note, this selection is performed independently of any validation
strategy.

### algorithm
This can be:
* none - All images are used.
* randomFraction - A random fraction is used to select a subset of images,
with optional random seed.
* randomQuantity - A specific total quantity is pulled randomly from the images,
with optional random seed.
* roundRobin - The dataset is split into groups selecting a specific group,
with optional random seed.  The same seed with different groups will produce a reproducible 
randomized split of a dataset.

### arguments
For **randomFraction**:
```
{
    "seed": <optional integer seed for the randomizer>
    "fraction": <decimal fraction, can be overriden by set specific fraction>
}
```

For **roundRobin**:
```
{
    "seed": <optional (but recommended) integer seed for the randomizer>
    "groups": <number of groups to split dataset into>
    "position": <group to select from result in range [0 - (groups - 1)]>
}
```

For **randomQuantity**:
```
{
    "seed": <optional integer seed for the randomizer>
    "count": <specific count of images to provide, can be overriden by set specific quantity>
}
```

## description
**Optional** prose description of this data set.

## timestamp
**Optional** Time stamp (ISO format with 0 microseconds) for when this file was last updated.

## formatVersion
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 3.2.0

History:
* 3.2.0 - Added a taskType field for imageData; changed csv type to "tabular"; updated descriptions
          for imageData properties to reflect support for Juneberry metadata files
* 3.1.0 - Pulled label name out to separate stanza, added csvData and removed properties from imageData. 
* 3.0.0 - Added dataType and switched data into imageData

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
