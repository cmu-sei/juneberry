Experiment Configuration Specification
==========


# Introduction

This document describes the schema for the JSON formatted experiment configuration file.

An experiment involves the training of one or more models with training data sets, 
combines this with the generation of predictions (tests) with a different collection of data sets, 
and the generation of some number of reports from aggregates of the test results.

# Schema
```
{
    "description": <Human readable description (purpose) of this experiment.>,
    "filters": [
        {
            "tag": "Tag referenced from the model or test."
            "cmd": [ string args with replacement of 
                     {model_name} {train_output}
                     {eval_predictions}  {dataset_path} ],
            "inputs": [ same as command ]
        },
    ]
    "format_version": <linux style version string of the format of this file>,
    "models": [
         {
            "filters" : [ "tag1", "tag2" ] - OPTIONAL
            "maximum_evaluations": <EXPERIMENTAL: maximum number of similar evaluations to perform at once. 
            Default is 1.>
            "name": <name of model in models directory>,
            "onnx": <boolean indicating if an ONNX version of the model should be saved>,
            "tests": [
                {
                    "classify": <Integer that controls how many of the top predicted classes get recorded>
                    "dataset_path": <A path to a dataset config file>,
                    "filters" : [ "tag1", "tag2" ]
                    "tag": <internal tag that reports references>,
                    "use_train_split": false
                    "use_val_split": false
                }
            ],
            "train": <OPTONAL> - True to train the model, False to just use existing model file. Default is True.
            "version": <OPTIONAL - which version of the model to use>,
        }
    ],
    "reports": [ <array of Report plugins - see below> ],
    "tuning": [
        {
            "model": <name of model in models directory to tune>,
            "tuning_config": <name of tuning config in workspace describing how to tune the model>
        }
    ],
    "timestamp": <optional ISO time stamp for when this was generated>
}
```

# Details

## description
**Optional** prose description of this experiment.

## filters
This section lists the filters that can be applied training output or evaluation output.
Each model or test stanza has a **filters** field specified which filters to invoke after the
training or evaluation as appropriate.
Each entry in here describes a pattern for how the filter but the actual values will be filled
in by the experiment.

### tag
A tag used to identify the filter

### cmd
A list of strings that represent the complete command that one would run on the command line.
The values will be expanded and any tokens in curly braces will be replaced based on current values 
from the current model and test stanzas. For example, if the
value is "{model_name}" then it will be replaced with the "name" from the model stanza.

The supported tokens are:
* model_name - The model name as a Path object
* train_output - A path to the training output as a Path object
* dataset_path - The dataset path from the model config or as passed to jb_evaluate as a Path object.
* eval_predictions - The path to the predictions file for the dataset as a Path object. Only valid
for evaluations.

By "as a Path object" we mean it is a Path object form the python pathlib and supports things
such as "name" and "stem". 

Multiple tokens can be used together.

Example:

If we have a model of "mymodel/myvariant" and a dataset of "data_sets/my_training_set.json" with
the value of "{model_name.name}-{dataset.stem}.txt" the result would be "myvariant-my_training_set.txt"

### inputs

A list of things that are "inputs" to the script as dependencies. These tell the build system
that it can execute the filter. This works identically to the command in terms of token replacement. 
In general practice, these inputs are just a subset of the command listed above.

## format_version
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 1.5.0

## models
An array of entries that describe which model(s) to train (if needed) and which test(s) to run against
the trained model(s).

### maximum_evaluations
**EXPERIMENTAL** **Optional** number specifying how many evaluations to perform **per call** to jb_evaluate
on loading the model. When generating the calls to jb_evaluate, similar evaluations (meaning the 
same classify, use_train_split, use_val_split, etc.) up to this maximum will be performed while only
loading the model once. This only impacts performance and should not affect correctness. This is
mostly useful from a runtime performance perspective when the model is large and the dataset is small.
This value defaults to 1.

### name
The name of the model to train if needed. (The directory in the "models" directory.)

### onnx
This boolean controls if an ONNX version of the model will be saved after training. When 
set to true, the "--onnx" option will be added to the jb_train command for the model during the 
creation of the experiment rules file.

### tests
An array of test sets to be run against the models. 

#### tag
A tag associated with the test to be used for report generation.

#### dataset_path
The name of the data set (file in data_sets folder) to be used for the test.

#### classify
This is an integer value that will control how many of the top-K predicted classes will be recorded 
for each input in the predictions file. This value is used as the "--classify" argument when 
jb_make_predictions is run. Since this field is not optional, set this property to zero if you do not 
want the predictions script to perform classifications. 

### train
By default, all models are trained before evaluation and reports are run. When this flag is set to false,
the model is assumed to have been trained outside of Juneberry, so no experiment rules will be generated 
to train (or clean) the model.

### version
**Optional** string that indicates which version of the model to use.

## reports
**OPTIONAL** An array of one or more Plugins, where each Plugin corresponds to a Juneberry Report 
class. Refer to the [report config specification](report_configuration_specification.md) for more information 
about Juneberry Report Plugins.

## tuning
**OPTIONAL** An array of one or more dictionaries, where each dictionary describes a tuning run. A tuning run 
can be used to optimize the hyperparameters of a model.

### model
This string indicates the name of the model in the workspace to tune. When generating a 'jb_tune' command 
for the experiment, the value of this property will be used for the command's "modelName" argument.

### tuning_config
This string indicates the name of the tuning config file, relative to the workspace, to use when tuning 
the model. When generating a 'jb_tune' command for the experiment, the value of this property will be 
used for the command's "tuningConfig" argument.

## timestamp
**Optional** Time stamp (ISO format with 0 microseconds) for when this file was last updated.

# Version History

* 0.2.0 - Big conversion to snake case in Juneberry 0.4.

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.
