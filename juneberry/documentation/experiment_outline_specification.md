Experiment Outline Specification
==========


# Introduction

This document describes the schema for the JSON formatted experiment outline file.

An experiment outline file can be used to produce an experiment configuration file, along with all of the 
training configuration files for the models in the experiment. For details on these types of files, 
refer to the specifications for experiment configurations and training configurations. 

# Schema
```
{
    "description": <Human readable description (purpose) of this experiment.>,
    "formatVersion": <linux style version string of the format of this file.>,
    "timestamp": <optional ISO time stamp for when the file was created.>,
    "tests": [
        {
            "tag": <An internal tag that will reference this test.>,
            "datasetPath": <A path to a data set config file.>
            "classify": <Integer that controls how many of the top predicted classes get recorded.>
        }
    ],
    "reports": [
        {
            "type": <report type: [plotROC | summary | all]>,
            "description": <A brief description of this report>,
            "testTag": <REQUIRED, type must be plotROC - A tag from the tests stanza above>,
            "classes": <OPTIONAL, type must be plotROC - comma separated classes to plot: e.g. 0,1,2,3,8,0+1,2+3>,
            "outputName": <REQUIRED, type must be summary - filename for output file> 
        }
    ]
    "baselineConfig": <The name of a model in the model directory>,
    "variables": [
        {
            "nickname": <A short string to describe the variable; used in the derived model name> 
            "configField": <A string indicating which TrainingConfig parameter to change>
            "values": [list of desired options for this variable] | "RANDOM" (if the configField is a seed)
        }
    ]
}
```

# Details

## description
**Optional** prose description of this experiment.

## timestamp
**Optional** Time stamp (ISO format with 0 microseconds) for when this file was last updated.

## tests
A list of test sets to be run against each of the models in the experiment.

### tag
A short string that will distinguish this test from any other tests specified 
in the outline.

### datasetPath
The path to a data set config file to use for the test. The path 
to use for the file should be relative to the workspace_root.

### classify
This is an integer value that will control how many of the top-K predicted classes will be
recorded for each input in the predictions file. Set this property to zero if you do not
want the predictions script to perform classifications.

## reports
A list of reports to produce when running the experiment.

### type
**REQUIRED** for any report. The type of report to generate. Currently we support "plotROC" 
(which invokes "jb_plot_roc"), "summary" (which invokes "jb_summary"), and "all". The "all" 
plot type will place every model for a given test on the same plot. 

### description
**REQUIRED** for any report type. A brief, human-readable description of the report.

### testTag
**REQUIRED** if the report type is "plotROC". This should be a string that matches one of the 
tags in the 'tests' section. For every model in the experiment, a ROC plot will be produced using 
the test data from the tag matching the tag provided here.

### classes
*OPTIONAL* if the report type is "plotROC". The list of classes to be extracted from the predictions 
file for use in the ROC plot. 

### outputName
**REQUIRED** if the report type is "summary". This string will be the filename that should 
be used when saving the summary report to the experiment directory.

## baselineConfig
A string that corresponds to the name of a model located in the models directory of 
the workspace_root. The config.json of the indicated model will serve as the "baseline" for 
all the model variations that will be generated for the experiment. Combinations of variables 
will be applied to the baseline and then saved as unique model training configs.

## variables
This is a list of training parameters to vary in the experiment. Each variable is represented by 
a dictionary with three keys, which are described below.    

### nickname
The variable's nickname should be a short string that describes the variable. The nickname is used 
when deriving the unique model names for the combinations of variables. These strings are not required, 
however they can help make your model names more readable. You can provide an empty string "" to skip 
the use of a nickname for a particular variable.

### configField
This key is a string that describes which property in the TrainingConfig should be varied. It is important 
to understand that fundamentally, a training config JSON file is a tree, with various paths corresponding 
to different configuration parameters. You change the configuration by modifying the value at a particular 
path in the tree. A root level parameter can be varied by providing the name of the root level parameter 
as your configField.

Dot notation can be used to reference keys that are nested within root level keys. For example, 
if the goal is to vary the validation seed, the key validation.arguments.seed can be used instead of 
having to provide the entire dictionary for the root level key named 'validation'. It is a good idea 
reference the structure of how your baselineConfig is constructed when selecting the keys used in your 
variables dictionary.

Comma notation can be used to indicate a group of parameters should be applied together, and this 
grouping should not be broken up when calculating the combinations of the variables in the experiment. 
Example: If the goal is to compare different learning rate schedulers, the key 
pytorch.lrSchedule,pytorch.lrScheduleArgs can be used. The comma in the middle separates the two keys we 
want to group together. Suppose the corresponding list of variables contains two elements, one with 
combination lrSchedule A with Args group 1 and one combination with lrSchedule B with Args group 2. Comma 
notation ensures that if lrSchedule A is chosen for the training config, then it will only be used with 
Args group 1 (and never Args group 2). Additionally, if lrSchedule B gets chosen, then Args group 2 must 
be used (and never Args group 1).

### values
This key should be a list of the values you would like to assign to the variable. The correct type for 
the elements in this list will depend on the parameter specified in the configField. Integers, strings, 
and dictionaries are probably the most common types.The list must have a minimum of two elements, 
otherwise the generation script will indicate the parameter is not actually a variable. 

Dictionaries should be used when the configField involves comma notation. The keys in the dictionary 
would correspond to the different comma separated parameters in configField. If the configField involves 
a seed value, you may forgo the list and simply provide the string "RANDOM". This will generate a 
random numpy compatible value (2**32 - 1) any time a training config is generated with that variable.

## formatVersion
Linux style version of **format** of the file. Not the version of 
the data but the version of the semantics of the fields of this file. 
Current: 1.0.0

* 1.0.0 - Initial version of experiment outlines

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
