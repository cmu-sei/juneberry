Experiment Outline Specification
==========


# Introduction

This document describes the schema for the JSON formatted experiment outline file.

An experiment outline file can be used to produce an experiment configuration file, along with all the 
model configuration files for the models in the experiment. For details on these types of files, 
refer to the specifications for experiment configurations and model configurations. 

# Schema
```
{
    "baseline_config": <The name of a model in the model directory>,
    "description": <Human readable description (purpose) of this experiment.>,
    "filters": [ < list of experiment filters to include verbatim > ]
    "format_version": <linux style version string of the format of this file.>,
    "model" : {
        "filters": [ <list of filters to add to each model> ]
    }
    "reports": [
        {
            "type": <report type: [plot_roc | plot_pr | summary | all_roc | all_pr]>,
            "description": <A brief description of this report>,
            "test_tag": <REQUIRED, type must be plot_roc - A tag from the tests stanza above>,
            "classes": <OPTIONAL, type must be plot_roc - comma separated classes to plot: e.g. 0,1,2,3,8,0+1,2+3>,
            "iou": <OPTIONAL, type must be plot_pr or all_pr - A float between 0.5 and 1.0>,
            "output_name": <REQUIRED, type must be summary - filename for output file> 
        }
    ],
    "tests": [
        {
            "tag": <An internal tag that will reference this test.>,
            "dataset_path": <A path to a data set config file.>,
            "classify": <Integer that controls how many of the top predicted classes get recorded.>
        }
    ],
    "timestamp": <optional ISO time stamp for when the file was created.>,
    "variables": [
        {
            "nickname": <A short string to describe the variable; used in the derived model name>, 
            "config_field": <A string indicating which ModelConfig parameter to change>,
            "vals": [list of desired options for this variable] | "RANDOM" (if the config_field is a seed)
        }
    ]
}
```

# Details

## baseline_config
A string that corresponds to the name of a model located in the "models" directory of 
the workspace_root. The config.json of the indicated model will serve as the "baseline" for 
all the model variations that will be generated for the experiment. Combinations of variables 
will be applied to the baseline and then saved as unique model configs.

## description
**Optional** prose description of this experiment.

## filters
A list of experiment filters to add directly to the generated experiment configuration.
See the experiment configuration specification for details.

## format_version
Linux style version of **format** of the file. Not the version of 
the data but the version of the semantics of the fields of this file. 
Current: 1.0.0

## model
Specific details to add to each generated model stanza.

### filters
A list of filters to apply to the model stanza verbatim.  See the experiment configuration
specification for details.

## reports
A list of reports to produce when running the experiment.

### type
**REQUIRED** for any report. The report type to generate. We currently support "plot_roc" 
(which invokes "jb_plot_roc"), "summary" (which invokes "jb_summary"), "plot_pr" (which 
invokes "jb_plot_pr"), "all_roc" and "all_pr". The "all_roc" and "all_pr" plot types will 
place every model for a given test on the same plot. 

### description
**REQUIRED** for any report type. A brief, human-readable description of the report.

### test_tag
**REQUIRED** if the report type is "plot_roc". This should be a string that matches one of the 
tags in the 'tests' section. For every model in the experiment, a ROC plot will be produced using 
the test data from the tag matching the tag provided here.

### classes
*OPTIONAL* if the report type is "plot_roc". The list of classes to be extracted from the predictions 
file for use in the ROC plot.

### iou
*OPTIONAL* if the report type is "plot_pr". This float value (expected to be between 0.5 and 1.0) is 
provided to jb_plot_pr as the value for the iou argument.

### output_name
**REQUIRED** if the report type is "summary". This string will be the filename that should 
be used when saving the summary report to the experiment directory.

## tests
A list of test sets to be run against each of the models in the experiment.

### tag
A short string that will distinguish this test from any other tests specified 
in the outline.

### dataset_path
The path to a data set config file to use for the test. The path 
to use for the file should be relative to the workspace_root.

### classify
This is an integer value that will control how many of the top-K predicted classes will be
recorded for each input in the predictions file. Set this property to zero if you do not
want the predictions script to perform classifications.

## timestamp
**Optional** Time stamp (ISO format with 0 microseconds) for when this file was last updated.

## variables
This is a list of parameters to vary in the experiment. Each variable is represented by 
a dictionary with three keys, which are described below.    

### nickname
The variable's nickname should be a short string that describes the variable. The nickname is used 
when deriving the unique model names for the combinations of variables. These strings are not required, 
however they can help make your model names more readable. You can provide an empty string "" to skip 
the use of a nickname for a particular variable.

### config_field
This key is a string that describes which property in the ModelConfig should be varied. It is important 
to understand that fundamentally, a model config JSON file is a tree, with various paths corresponding 
to different configuration parameters. You change the configuration by modifying the value at a particular 
path in the tree. A root level parameter can be varied by providing the name of the root level parameter 
as your config_field.

Dot notation can be used to reference keys that are nested within root level keys. For example, 
if the goal is to vary the validation seed, the key validation.arguments.seed can be used instead of 
having to provide the entire dictionary for the root level key named 'validation'. It is a good idea 
reference the structure of how your baseline_config is constructed when selecting the keys used in your 
"variables" dictionary.

Comma notation can be used to indicate a group of parameters should be applied together, and this 
grouping should not be broken up when calculating the combinations of the variables in the experiment. 
Example: If the goal is to compare different learning rate schedulers, the key 
pytorch.lrSchedule,pytorch.lrScheduleArgs can be used. The comma in the middle separates the two keys we 
want to group together. Suppose the corresponding list of variables contains two elements, one with 
combination lrSchedule A with Args group 1 and one combination with lrSchedule B with Args group 2. Comma 
notation ensures that if lrSchedule A is chosen for the model config, then it will only be used with 
Args group 1 (and never Args group 2). Additionally, if lrSchedule B gets chosen, then Args group 2 must 
be used (and never Args group 1).

### vals
This key should be a list of the vals you would like to assign to the variable. The correct type for 
the elements in this list will depend on the parameter specified in the config_field. Integers, strings, 
and dictionaries are probably the most common types.The list must have a minimum of two elements, 
otherwise the generation script will indicate the parameter is not actually a variable. 

Dictionaries should be used when the config_field involves comma notation. The keys in the dictionary 
would correspond to the different comma separated parameters in config_field. If the config_field involves 
a seed value, you may forgo the list and simply provide the string "RANDOM". This will generate a 
random numpy compatible value (2**32 - 1) any time a model config is generated with that variable.

# Version History

* 0.2.0 - Big conversion to snake case in Juneberry 0.4.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.