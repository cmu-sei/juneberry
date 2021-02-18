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
    "models": [
         {
            "name": <name of model in models directory>,
            "version": <OPTIONAL - which version of the model to use>,
            "tests": [
                {
                    "tag": <internal tag that reports references>
                    "datasetPath": <A path to a data set config file.>
                    "classify": <Integer that controls how many of the top predicted classes get recorded.>
                }
            ]
        }
    ],
    "reports": [
        {
            "description": <human readable description of this report>,
            "type": <report type: [plotROC | summary]
            "classes": <default classes - used if not specified per test>
            "tests": [
                "tag": <tag from test stanza above>
                "classes": <comma separated classes to extract: e.g. 0,1,2,3,8,0+1,2+3>
            ]
            "outputName": <filename for output file>
            "csvName": <For summary optional tag to write table to csv file>
            "plotTitle": <Title to use for ROC plot figure>
        }        
    ]
    "formatVersion": <linux style version string of the format of this file>,
    "timestamp": <optional ISO time stamp for when this was generated>
}
```

# Details

## description
**Optional** prose description of this experiment.

## models
An array of entries that describe which model(s) to train (if needed) and which test(s) to run against
the trained model(s).

### name
The name of the model to train if needed. (The directory in the models directory.)

### version
**Optional** string that indicates which version of the model to use.

### tests
An array of test sets to be run against the models. 

#### tag
A tag associated with the test to be used for report generation.

#### datasetPath
The name of the data set (file in data_sets folder) to be used for the test.

#### classify
This is an integer value that will control how many of the top-K predicted classes will be recorded 
for each input in the predictions file. This value is used as the "--classify" argument when 
jb_make_predictions is run. Since this field is not optional, set this property to zero if you do not 
want the predictions script to perform classifications. 

## reports
A list of reports to produce. (See [Report Specifications](#Report%20Specifications) for details on each.)

### description
A human-readable description of the report.

### type
The type of report to generate. Currently we support "plotROC" with invokes "jb_plot_roc" and 
"summary" which invokes "jb_summary"

### classes
The default list of classes to be extracted from the predictions file to be used in the plot.
Each test can specify classes to override this default value.

### plotTitle
When the report type is "plotROC", this string will be used for the title of the Figure for 
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

## formatVersion
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 1.5.0

* 1.5.0 - Added support for showing topK classifications
* 1.4.0 - Added support for figure titles in ROC plots
* 1.3.0 - Added support for csv output from summary report
* 1.2.0 - Changed data_set_name to datasetPath
* 1.1.0 - Added support for "summary" report type

# Report Specifications

## ROC Plot
```
{
    "description": <human readable description of this report>,
    "type": plotROC
    "classes": <default classes - used if not specified per test>
    "tests": [
        "tag": <tag from test stanza above>
        "classes": <comma separated classes to extract: e.g. 0,1,2,3,8,0+1,2+3>
    ]
    "outputName": <filename for output file>
    "plotTitle": <Title to use for ROC plot figure>
}
```

ROC Plot will generate a RoC curve plot for the specified tests and classes based on model test output.


## Summary
```
{
    "description": <human readable description of this report>,
    "type": summary
    "outputName": <filename for output file>
    "csvName": <optional tag to write table to csv file>
} 
```

Summary will generate a simple markdown report that summarizes all model and prediction values and displays or 
links to any training graphs and RoC plots that were produced as part of the experiment process.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
