Report Configuration Specification
========== 
<!-- TOC -->
* [Introduction](report_configuration_specification.md#introduction)
* Report Config Structures
  * [Basic Report Config Structure](report_configuration_specification.md#Basic-Report-Config-Structure)
  * [Report Config Structure Extensions](report_configuration_specification.md#Report-Config-Structure-Extensions)
    * [Experiment Outline](report_configuration_specification.md#report-config-structure-extensions---experiment-outline)
    * [Experiments](report_configuration_specification.md#report-config-structure-extensions---experiments)
  * [Report Specific Config Structures](report_configuration_specification.md#Report-Specific-Config-Structures)
    * [ROC Report](report_configuration_specification.md#ROC-Report)
    * [PR Report](report_configuration_specification.md#PR-Report)
    * [Summary Report](report_configuration_specification.md#Summary-Report)
  * [Plugin Structure](report_configuration_specification.md#Plugin-Structure)
  * [Structure Adjustments in an Experiment Config](report_configuration_specification.md#Structure-Adjustments-in-an-Experiment-Config)
    * [ROC Report](report_configuration_specification.md#Experiment-Config-ROC-Report-Stanza-Differences)
    * [PR Report](report_configuration_specification.md#Experiment-Config-PR-Report-Stanza-Differences)
    * [Summary Report](report_configuration_specification.md#Experiment-Config-Summary-Report-Stanza-Differences)
    * [Custom Report](report_configuration_specification.md#Experiment-Config-Custom-Report-Stanza-Differences)
* [Details](report_configuration_specification.md#Details)
  * [Reports](report_configuration_specification.md#reports)
  * [Experiment Outline Extensions](report_configuration_specification.md#details---experiment-outline-extensions)
  * [Experiment Extensions](report_configuration_specification.md#details---experiment-extensions)
  * [ROC Report](report_configuration_specification.md#details---roc-report)
  * [PR Report](report_configuration_specification.md#details---pr-report)
  * [Summary Report](report_configuration_specification.md#details---summary-report)
<!-- TOC -->

# Introduction
This document describes the JSON format used in a Juneberry Report configuration file. Juneberry 
Report configurations have a unique behavior when compared to other Juneberry configuration files, 
in that it may exist within their own configuration file OR be inserted into other configuration files, 
such as model configs or experiment configs.

# Basic Report Config Structure
The following structure represents the foundation of a "reports" stanza in any Juneberry config 
file. You'll typically encounter this structure (without any extensions) in standalone Report config files, 
as well as model config files. Other types of config files may layer additional extension points on top of 
this structure.
```
{
    "reports": [
        {
            "description": <Human readable description of this report>,
            "fqcn": <fully qualified name of a class that extends the juneberry.reporting.report base class>,
            "kwargs": { <OPTIONAL kwargs to be passed (expanded) to __init__ on construction> }
        }
    ]
}
```

## Report Config Structure Extensions
In certain types of config files, such as Experiment Outline configs and Experiment configs, the basic Report Config 
structure was extended to include fields that were designed to be compatible with features that are specific 
to those types of configs. The details of those extension points are outlined in this section.

### Report Config Structure Extensions - Experiment Outline
The following Report structure demonstrates a "reports" stanza in an Experiment Outline config. Each individual 
report element combines the basic report structure, along with two additional fields: "classes" and "test_tag".
```
{
    "reports": [
        {
            "classes": <OPTIONAL - used in plot ROC reports; comma separated classes to plot: e.g. 0,1,2,3,8,0+1,2+3>,
            "description": <Human readable description of this report>,
            "fqcn": <fully qualified name of a class that extends the juneberry.reporting.report base class>,
            "kwargs": { <OPTIONAL kwargs to be passed (expanded) to __init__ on construction> },
            "test_tag": <REQUIRED (for ROC or PR Report types) - A tag from the tests stanza in the experiment outline>,
        }
    ]
}
```

### Report Config Structure Extensions - Experiments
The following Report structure demonstrates a "reports" stanza in an Experiment config. Each individual 
report element combines the basic report structure, along with two additional fields: "classes" and "tests".
```
{
    "reports": [
        {
            "classes": <OPTIONAL - used in plot ROC reports; comma separated classes to plot: e.g. 0,1,2,3,8,0+1,2+3>,
            "description": <Human readable description of this report>,
            "fqcn": <fully qualified name of a class that extends the juneberry.reporting.report base class>,
            "kwargs": { <OPTIONAL kwargs to be passed (expanded) to __init__ on construction> },
            "tests": <REQUIRED (for ROC or PR Report types) - A tag from the tests stanza in the experiment outline>,
        }
    ]
}
```

## Report Specific Config Structures

Juneberry includes a few built-in Report classes. This section describes the structure for each 
of the available Report types.

### ROC Report
The ROC Report type uses model evaluation output data to generate a plot with one or more RoC curves.
```
{
    "description": <Human readable description of this report>,
    "fqcn": <A string, must be equal to "juneberry.reporting.roc.ROCPlot">
    "kwargs": {
        "output_filename": <A string indicating the desired location for the output image>,
        "plot_title": <A string indicating the desired title for the Figure>,
        "legend_scaling": <A float indicating the scale factor to apply to the legend>,
        "legend_font_size": <An integer that controls the font size in the legend>,
        "line_width": <An integer that controls the line width in the figure>,
        "curve_sources": {A dictionary of key:value pairs, where each key represents a 
                          "predictions.json" file produced by a model evaluation, and the 
                          value represents which classes to generate ROC curves for from that 
                          predictions data.}
    }
}
```

### PR Report
The PR Report type will generate Precision-Recall-Confidence plots.
```
{
    "description": <Human readable description of this report>,
    "fqcn": <A string, must be equal to "juneberry.reporting.pr.PRCurve">
    "kwargs": {
        "output_dir": <A string indicating the desired output directory for storing the images 
                       generated by this report>,
        "iou": <A float indicating the iou threshold to use when calculating the curves in this report>,
        "tp_threshold": <A float indicating the true-positive threshold to use when calculating the 
                         curves in this report>,
        "stats_fqcn": <A string indicating which class to use in the Juneberry MetricsManager>,
        "curve_sources": <A list of dictionaries, where each dictionary in the list contains two keys: 
                          "model" and "dataset". A PR curve will be added to each figure for each 
                          dictionary in the list.>
    }
}
```

### Summary Report
The Summary Report type will generate a simple markdown report (and optional CSV) that summarizes 
model training and eval results, and displays or links to any training graphs and plots that were 
produced.
```
{
    "description": <Human readable description of this report>,
    "fqcn": <A string, must be equal to "juneberry.reporting.summary.Summary">
    "kwargs": {
        "md_filename": <A string indicating the desired output filename for the Summary markdown file>,
        "csv_filename": <A string indicating the desired output filename for the Summary CSV file>,
        "metrics_files": <A list of strings, where each string represents a Juneberry metrics.json 
                          file which the Report should pull Summary information from>,
        "plot_files": <A list of strings, where each string represents an image file to include in 
                       the Summary report>
    }
}
```

# Plugin Structure

The Juneberry Report structure follows the same basic pattern as other Juneberry Plugins. A class is 
specified by its Fully Qualified Class Name (FQCN) and a dictionary of keyword arguments can be 
passed in during construction. A Report Class is typically written with an init method and a 
"create_report" method.

The following code depicts a simple example of a Report Class with two arguments in the init method 
and a "create_report" method to produce the report:

```python
from juneberry.reporting.report import Report

class ExampleReport(Report):
    def __init__(self, report_output_dir: str, model_name: str, accuracy: float):
        super().__init__(output_str=report_output_dir)
        self.model_name = model_name
        self.accuracy = accuracy
        
    def create_report(self) -> None:
        print(f"Model name: {self.model_name}\n  Accuracy: {self.accuracy}")
```

The `kwargs` stanza when using this Report should have `model_name` and `accuracy` properties:

```json
{
    "kwargs": {
        "report_output_dir": "/juneberry/report_output_dir",
        "model_name": "ResNet",
        "accuracy": 0.5
    }
}
```

If the code for the Example report was stored in a file named `juneberry/reporting/example.py`, then the 
full report stanza would look something like this:

```json
{
    "description": "Report stanza illustrating how to use ExampleReport",
    "fqcn": "juneberry.reporting.example.ExampleReport",
    "kwargs": {
        "report_output_dir": "/juneberry/report_output_dir",
        "model_name": "ResNet",
        "accuracy": 0.5
    }
}
```

The previous stanza could then be inserted into the "reports" list of a report config file, a 
model config file, or an experiment config file.

# Structure Adjustments in an Experiment Config
When a "reports" stanza exists inside an experiment config, the structure and behavior of a particular 
Report may vary slight from the structures described above. This section will describe these differences 
for each Report Type.

There's a good chance that some filenames required by a particular report's structure may not be 
known in advance of running the experiment. In these situations, a report stanza may omit certain 
fields from the experiment config and rely on Juneberry to fill in the correct value(s) for the 
field once those values are determined. This is especially common for the "curve_sources" field in 
the ROC and PR Reports. When generating the rules.json file for the experiment, Juneberry will fill 
in these fields based on the output filenames that were determined during rule generation.

Although Juneberry will fill in some fields for a report stanza in an experiment config, these values 
will never be written to experiment config itself, in order to preserve to original state of that config. 
Instead, a sub-directory named 'report_json_files' will appear inside the experiment directory. That 
directory will contain one or more JSON files, where each file contains an individual report stanza 
from the "reports" stanza in the experiment config, augmented with any fields that were determined 
during the creation of the pydoit rules for that Report type. Refer to the information in the next 
four subsections for more information on possible augmentations for each Report type.

Another unique Report structure behavior involving experiments relates to any fields where an output filename or 
directory is requested. Since Juneberry has a strong desire to organize any experiment files inside 
the appropriate experiment directory, any values for output filenames or directories are checked to verify 
that they are inside the experiment's experiment directory. When the provided string indicates a file 
or directory that is not inside the experiment's directory, then Juneberry will adjust the value such 
that the file or directory will be located inside the experiment's directory. Like before, these adjustments 
will not be reflected inside the experiment config, but you would be able to see them inside the 
individual report JSON files that appear in the 'report_json_files' sub-directory of the experiment.

## Experiment Config ROC Report Stanza Differences
An ROC Report stanza in an experiment config may have two types of differences: one for "tests" and 
another for "classes". Both a related to the construction of the "curve_sources" field.

"tests" are a way to indicate which model and eval dataset combination from the experiment's "models" 
stanza should be included in the report. According to the 
[experiment configuration specification](experiment_configuration_specification.md), a "test" identifies 
which dataset to use when evaluating a model, and each test is identified by a unique filter. The "tests" 
stanza in an experiment's ROC Report stanza is used to add particular model and eval dataset combinations 
to the ROC report. The "tests" field is inserted as a peer to the 'fqcn' and 'kwargs' fields. Consider 
the following example:

```json
{
    "reports": {
        "description": "",
        "fqcn": "juneberry.reporting.roc.ROCPlot",
        "kwargs": {
            "output_filename": ""
        },
        "tests": [
            {
                "tag": "Tag 1"
            },
            {
                "tag": "Tag 2"
            }
        ]
    }
}
```

This ROC report stanza will add two curve sources to the curve_sources field in the individual report 
JSON that will be created for this experiment report: one for the model/eval dataset combination matching 
"Tag 1" and one for the model/eval dataset combination matching "Tag 2".

However, which model/eval dataset combination to use is not the only piece of information required for a 
proper entry in the "curve_sources" field. A curve_source must also indicate which classes to use when 
plotting ROC curves. This need is resolved by the introduction of a "classes" field in the stanza. The 
"classes" field may appear as a peer to the "tests" field, or as a field *inside* a test. If the "classes" 
field appears in both locations, the "classes" field inside the test is chosen. Consider the following 
example:

```json
{
    "reports": {
        "classes": "all",
        "description": "",
        "fqcn": "juneberry.reporting.roc.ROCPlot",
        "kwargs": {
            "output_filename": ""
        },
        "tests": [
            {
                "classes": "0,1,2",
                "tag": "Tag 1"
            },
            {
                "classes": "0,1,2",
                "tag": "Tag 2"
            }
        ]
    }
}
```

Here you can see the "classes" field has been added to the Report in both possible locations. However, 
the classes "0,1,2" will be chosen for the value in curve_sources because that location is given 
higher priority.

## Experiment Config PR Report Stanza Differences
A PR report stanza in an experiment config only has one potential new field in the stanza, and it's the 
"tests" field described in the [previous section](#Experiment Config ROC Report Stanza Differences). The 
purpose and functionality of the field is the same: it's to populate the 'curve_sources' field for the 
PR Report with the correct model/eval dataset combinations. 

## Experiment Config Summary Report Stanza Differences
The Summary Report stanza differences in an experiment config don't include any new fields. Instead, 
they're described best by the fields that might not be there: the 'metrics_files' and 'plot_files' 
kwargs. The 'metrics_files' indicates which evaluation metrics to include in the Summary report, while 
the 'plot_files' indicate any plots or figures to include in the summary. It's likely that both of these 
categories will contain filenames that won't be known until experiment runtime. 

As a result, the experiment rules generation process aims to append the metrics and plot files generated 
during the experiment into the appropriate list. If any files appear in either list beforehand, the rule 
generation process will augment the existing list(s) with the metrics and plot files produced during the 
rules process. 

For the additions to the 'metrics_files' list, the process will loop through the unique tags in the 
experiment and add the metrics file for the model/eval_dataset combination to the list of metrics files.
Additions to the 'plot_files' list are identified by looping through the lists of reports in the "reports" 
stanza, identifying and reports that produce PR or ROC images, and then appending those results images into 
the 'plot_files' list.

## Experiment Config Custom Report Stanza Differences
When designing your own custom reports, the alternate stanza behaviors described in the previous sections 
will not apply to your report type without significant modifications to 
[jb_experiment_to_rules](../../bin/jb_experiment_to_rules). Without making those modifications, it's best 
to treat the inclusion of a custom report into your experiment as a strict follower of the
[basic report config structure](#Basic Report Config Structure), and not expect the experiment rules generation 
process to auto-fill any missing fields for the custom report.

# Details
This section provides more information about each of the fields in the basic Report structure.

## reports
A list of reports to produce, where each Report is represented by a single Report plugin as 
described in the [Plugin Structure](#Plugin Structure) section.

### description
A human-readable description of the report.

### fqcn
The fully qualified class name (fqcn) of the Report to build, e.g. 
"**juneberry.reporting.summary.Summary**".

## Details - Experiment Outline Extensions
This section provides more information about each of the extension fields to the basic 
Report structure that are specific to "reports" stanzas found in experiment outline configs.

### classes
A string used in a ROC Report indicating which classes to produce ROC curves 
for on the ROC plot.

### test_tag
A string matching one of the tags in the 'tests' section of the experiment outline. A Report of 
the type indicated by the 'fqcn' will be generated for every model/eval dataset combination that 
matches this test tag. 

## Details - Experiment Extensions
This section provides more information about each of the extension fields to the basic 
Report structure that are specific to "reports" stanzas found in experiment outline configs.

### classes
A string used in a ROC Report indicating which classes to produce ROC curves 
for on the ROC plot.

### tests
A string which should match one of the tags in the experiment. The data from the indicated test will 
be used to generate the report.

### kwargs
A dictionary containing all the arguments to pass into the `__init__` method of the 
Report class indicated in the 'fqcn' field.

## Details - ROC Report
This section provides more information about each of the fields in a ROC Report stanza.

### description
A human-readable description of the report.

### fqcn
The fully qualified class name (fqcn) of the Report to build, which for a Juneberry 
ROC Report should be equal to "**juneberry.reporting.roc.ROCPlot**".

### kwargs
A dictionary containing all the arguments to pass into the `__init__` method of 
**juneberry.reporting.roc.ROCPlot**.

#### output_filename
A string indicating the desired name for the output file. The output file is a figure 
contain ROC curves, so typically this field will be a PNG file. When this field is not 
provided, the ROC image will be placed in the current working directory using the 
filename "ROC_curves.png". When this field is set to a directory, the ROC image will 
be placed inside that directory using the filename "ROC_curves.png". 

#### plot_title
This string will be used for the title of the Figure in the ROC plot. When this field 
is not provided, the Report uses "ROC Curve(s)" for the figure title.

#### legend_scaling
A float that can be used to adjust the scale factor for the legend. When this field 
is not provided, the Report uses 1.0 as the default value.

#### legend_font_size
This integer is used to control the fontsize of the text in the legend. When this field 
is not provided, the Report uses 10 as the default value.

#### line_width
This integer controls the width of ROC curve lines in the figure. When this field is 
not provided, the Report uses 2 as the default value.

#### curve_sources
This dictionary contains key:value pairs that indicate which evaluation data and classes 
should be used to generate ROC curves for the figure. 

A key should be a string indicating 
a Juneberry Evaluation Output file (see the [Evaluation Output Specification](eval_output_specification.md)). 

The corresponding value should be a string of classes in the evaluation data to produce ROC 
curves for. The following examples illustrate various ways to construct the class string:
 - 'all' - Plots an ROC curve for each class in the evaluation data.
 - '0,1,2' - Plots three different ROC curves, one for class 0, one for class 1, and one for class 2.
 - '0,dog,1' - Plots three different ROC curves, one for class 0, one for the 'dog' class (the Report 
will look up the class string and convert to the correct integer class), and one for class 2.
 - '0,1+2,3' - Plots three different ROC curves, one for class 0, one for class 1 and 2 data combined 
into a single curve, and one for class 3.
 - 'cat, 3+dog, 4' - Various combinations of the previous techniques are supported. The Report will 
simply convert any strings to the corresponding integers and perform class combinations as requested.

Therefore, a correctly constructed 'curve_sources' property would look something like this:
```json
{
    "curve_sources": {
        "models/example_model/eval/example_dataset/predictions.json": "0,1,2",
        "models/example_model2/eval/example_dataset/predictions.json": "0,dog+cat,2"
    }
}
```

This set of curve sources would produce a Figure containing 6 ROC curves from 2 different models which 
were evaluated using the same evaluation dataset. Each model provides the data for 3 curves. The 
second curve from example_model2 will be a combined curve for the dog and cat classes.

## Details - PR Report
This section provides more information about each of the fields in a PR Report stanza.

### description
A human-readable description of the report.

### fqcn
The fully qualified class name (fqcn) of the Report to build, which for a Juneberry 
PR Report should be equal to "**juneberry.reporting.pr.PRCurve**".

### kwargs
A dictionary containing all the arguments to pass into the `__init__` method of 
**juneberry.reporting.pr.PRCurve**.

#### output_dir
A string indicating the desired output directory for the three images produced by this 
Report type. When this field is not provided, the three images will be placed in the 
current working directory. When this field is set to a filename, the output directory 
will be set to the parent directory of that file. The filenames of the three images 
that will appear in the output directory are "pr_curve.png", "pc_curve.png", and 
"rc_curve.png".

#### iou
This float controls the iou threshold to use when generating the data for the curves. 
When not provided, the default value for this field is 0.5.

#### tp_threshold
This float controls the true positive threshold to use when generating the data for 
the curves. When not provided, the default value for this field is 0.8.

#### stats_fqcn
This string provides an opportunity to use a custom class to calculate the metrics. The 
default value for this class is "juneberry.metrics.metrics.Stats".

#### curve_sources
This array contains one or more dictionaries describing which combinations of models and evaluation 
datasets should be used to add curves to each Figure. Each dictionary should contain a "model" key 
indicating the model to use for the curve source and a "dataset" key indicating which evaluation 
dataset to use along with the model to produce data for the curve.

For example, a properly constructed 'curve_sources' field would look something like this:
```json
{
    "curve_sources": [
        {
            "model": "model_1",
            "dataset": "eval_dataset_1"
        },
        {
            "model": "model_1",
            "dataset": "eval_dataset_2"
        },
        {
            "model": "model_2",
            "dataset": "eval_dataset_1"
        }
    ]
}
```

In this example, each of the three PR Figures would contain 3 curves. Two curves would come from the 
same model but different eval datasets. The third curve would come from a different model evaluated 
using one of the previous datasets.

## Details - Summary Report
This section provides more information about each of the fields in a Summary Report stanza.

### description
A human-readable description of the report.

### fqcn
The fully qualified class name (fqcn) of the Report to build, which for a Juneberry 
Summary Report should be equal to "**juneberry.reporting.summary.Summary**".

### kwargs
A dictionary containing all the arguments to pass into the `__init__` method of 
**juneberry.reporting.summary.Summary**.

#### md_filename
This string indicates the desired name for the output markdown file for the Summary Report. When this 
string is not provided, the output markdown file will be placed in the current working directory 
with the filename "summary.md". When a directory is provided in this field, the output markdown file 
will be placed in that directory with the filename "summary.md". When this field is set to a filename, 
then the Summary report will be saved to that file.

#### csv_filename
This string indicates that a CSV version of the Summary file is desired and what the corresponding 
filename of that CSV file should be. If this key is omitted from the kwargs, then no CSV will be 
generated. If this field is set to an empty string, then the Summary CSV will be placed in the 
current working directory with the filename "summary.csv". If this field is set to a directory, 
the output CSV file will be placed in that directory with the filename "summary.csv". When this 
field is set to a filename, then the Summary CSV will be saved to that file.

#### metrics_files
This array of strings indicates which model metrics files should contribute data to the Summary 
Report. A metrics file is an [evaluation output](eval_output_specification.md) file that does not 
contain predictions data. Each string in this array should correspond to a metrics file to pull 
summary data from. One row will be added to the Summary table for every metrics file in this array. 
When a Summary CSV has been requested, each metrics file will contribute one line to the CSV.

#### plot_files
This array of strings indicates any plot files or figures that should be included in the Summary 
Report markdown file. Any plot files included in this array will appear in the "Experiment Plots" 
section which appears below the "Experiment summary" table in the Summary markdown file. The plot 
files in this list do not contribute any information to the Summary CSV file.
