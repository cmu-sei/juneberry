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
            "version": <OPTIONAL - which version of the model to use>,
        }
    ],
    "reports": [
        {
            "description": <human readable description of this report>,
            "type": <report type: [plot_roc | plot_pr | summary],
            "classes": <default classes - used if not specified per test>,
            "tests": [
                {
                    "tag": <tag from test stanza above>,
                    "classes": <comma separated classes to extract: e.g. 0,1,2,3,8,0+1,2+3>
                }
            ],
            "output_name": <filename for output file>,
            "csv_name": <For summary optional tag to write table to csv file>,
            "plot_title": <Title to use for ROC plot figure>
        }        
    ],
    "timestamp": <optional ISO time stamp for when this was generated>
}
```

# Details

## description
**Optional** prose description of this experiment.

## filters
This sections lists the filters that can be applied training output or evaluation output.
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

### name
The name of the model to train if needed. (The directory in the "models" directory.)

### onnx
This boolean controls whether or not an ONNX version of the model will be saved after training. When 
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

### version
**Optional** string that indicates which version of the model to use.

## reports
A list of reports to produce. (See [Report Specifications](#Report Specifications) for details on each.)

### description
A human-readable description of the report.

### type
The report type to generate. We currently support "plot_roc" which invokes "jb_plot_roc", "plot_pr" which 
invokes "jb_plot_pr", and "summary" which invokes "jb_summary".

### classes
The default list of classes to be extracted from the predictions file to be used in the plot.
Each test can specify classes to override this default value.

### plot_title
When the report type is "plot_roc", this string will be used for the title of the Figure for 
the ROC plot.

### tests
A lists of test sets to be aggregated into this report.

#### tag
This refers to the 'tag' field in the lists of tests described above in the tests sections.
This is a simple a convenient way to refer to the above tests. 

#### classes
Optional list of classes to be used from this test set.  This overrides the classes field at the
root if this stanza.

### output
The base name of the output provided to the report generator.

## timestamp
**Optional** Time stamp (ISO format with 0 microseconds) for when this file was last updated.


# Report Specifications

## ROC Plot
```
{
    "description": <human readable description of this report>,
    "type": "plot_roc",
    "classes": <default classes - used if not specified per test>,
    "tests": [
        "tag": <tag from test stanza above>,
        "classes": <comma separated classes to extract: e.g. 0,1,2,3,8,0+1,2+3>
    ]
    "output_name": <filename for output file>,
    "plot_title": <Title to use for ROC plot figure>
}
```

ROC Plot will generate a RoC curve plot for the specified tests and classes based on model test output.


## PR Plot
```
{
    "description": <human readable description of this report>,
    "iou": <IoU threshold to use for determining whether to count a detection as a true positive>,
    "output_dir": <filename for output file>,
    "tests": [
        "tag": <tag from test stanza above>
    ],
    "type": "plot_pr",
}
```

PR Plot will generate PR plots for the specified tests.


## Summary
```
{
    "description": <human readable description of this report>,
    "type": "summary"
    "output_name": <filename for output file>,
    "csv_name": <optional tag to write table to csv file>
} 
```

Summary will generate a simple markdown report that summarizes all model and prediction values and displays or 
links to any training graphs and plots that were produced as part of the experiment process.


# Version History

* 0.2.0 - Big conversion to snake case in Juneberry 0.4.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
