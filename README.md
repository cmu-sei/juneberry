README
==========


# Introduction

Juneberry is a framework for running machine learning experiments. Think of it as an "automated grad student."

This README describes how to use the Juneberry framework to execute machine learning tasks.
Juneberry follows a (mostly) declarative programming model composed of sets of config files 
(dataset, model, and experiment configuration) and python plugins for things such
as model construction and transformation.

This README contains:
* Steps for running a simple test of your Juneberry installation
* A tutorial on running a simple experiment by hand
* Details about building your own models

# TL;DR

To see if the repo is installed and working properly
1) Clone the Juneberry repository
1) `cd juneberry` (or whatever directory you cloned the repository to)
1) `pipenv install -e . --dev`
1) `pipenv run jb_train -w . -d . tabular_multiclass_sample`

When the training is complete, you should see a training accuracy of 97.67% in the final epoch.

NOTE: When running the environment, all the Juneberry tools (scripts starting with jb_) are automatically placed
in your PATH.

## Other Environments

Juneberry is a fully-featured python package and can be installed into any environment such as venv. Once python has
been installed, simply install Juneberry using `pip install -e .` from within the juneberry directory. This will
set up the package, including its dependencies, and add the bin directory to your path.

# Experiment Overview

Juneberry uses a workspace directory structure to hold models and experiments. Most
of the configuration is done via files.  

```
<workspace root>/
    data_sets/ - JSON config files describing datasets are listed here.
        <data-set-name>.json
    models/ - Files related to models, including JSON config files describing models, are listed here.
        <model-name>
            config.json
            ... experiment specific artifacts (logs, visualizations, etc.) ARE PLACED here ...
    experiments/ - experiments are sets of models with their tests
        <experiment-name>
            config.json
            ... output such as logs and trained models ARE PLACED here ...
    <workspace package> - [OPTIONAL] Python code goes here and should be available for import
        <subpackage>
        
        
<DATA_ROOT>/
    datagroup_x/
        version_x/
    cache/
``` 

Juneberry scripts need to know the path to the workspace and data directories described above. These paths can 
be provided on a per-script basis using the `-w` and `-d` switches respectively. Normally it is much more
convenient to specify these (and other settings) once via a `juneberry.ini` file, described below in 
"Configuration via juneberry.ini". If you wish to use a `juneberry.ini` instead of providing the workspace and 
data root directories via command line switches, you are expected to create your own `juneberry.ini` file in an 
appropriate location. Refer to the "Search Order" section of "Configuration via juneberry.ini" for the relevant
details.

# WORKSPACE Directory 

The workspace path can be set via the `-w` command line switch or more generally via `workspace_root` in your
`juneberry.ini`.

### Models (model directory)

A model directory (user specified name in models) should contain a config file that describes 
an architecture to build, configuration changes to that architecture, and the datasets
used to train the model. It will also contain the trained model, log files, plots, and other
files specifically associated with this model.

One of the core concepts of Juneberry is the model. Our definition of model is the combination of an architecture
configured with hyperparameters, trained using a particular dataset (modified by transformers as needed),
a loss function, an optimizer, a training process (e.g. adversarial, validation splitting) which results
in a trained model, and resulting output files. Together, these components form what we refer to as the model. 
The central part of the model is the "training config" (config.json) which describes the components of the model. 
Constituent parts such as architecture code or dataset descriptions are stored elsewhere, 
but the model config specifies which of these parts go in to the model.

### Datasets (data_set directory)

Dataset configs are JSON files that describe where data elements come from, and how that data should be labeled.
Data configs allow you to define subsets of source directories, how they are ordered, and how the data should be 
split or transformed (such as image resizing). Together these options allow you to construct a unique and traceable
data input set. Each JSON file is a different dataset configuration. All data paths in the JSON file should be 
relative to a single data root. This root is provided to Juneberry via the `juneberry.ini` file or the `-d` 
command line switch.



### Experiments (experiments directory)

Experiments group multiple models together for the purpose of comparing their outputs, providing
the computational basis for scientific experiments. Each experiment should contain a JSON config file describing 
the models in the experiment, the datasets to test against those models, and the reports to produce using those 
trained models.

## DATA_ROOT Directory Path

The data root directory path can be set via the `-d` command line switch or via the `data_root` field in your
`juneberry.ini` file.

The DATA_ROOT directory is where input data files live. For more details on the exact DATA_ROOT structure,
refer to [data_file_structure.md](./juneberry/documentation/data_file_structure.md) 
in the Juneberry documentation directory.

### datagroup_x (optional)
A group of data files grouped together under some common attributes.

### version_x (optional)
Datagroups can be versioned. If they are, you will see version directories within the datagroup directory.

## Configuration via juneberry.ini

### Content

The juneberry ini file can contain the following fields:

```
[DEFAULT]
workspace_root = /path/to/workspace
data_root = /path/to/data
tensorboard_root = /path/to/tensorboard/data
```

The Juneberry framework is designed to work with "workspaces" or "data roots"
that are external to the Juneberry code itself. This allows the workspace (which may be small and versioned) 
to be independent. 

The workspace (specified by workspace_root) is the data associated with the actual experiments. It 
contains references to datasets, transforms for the input data, python files for constructing models, 
hyperparameters to those models, references to loss functions or optimizers, and references to 
visualizations or other reporting analytics.

The data root (specified by data_root) is the root level directory containing your input data. Paths in 
the dataset config file will be treated as relative to this "data root" directory.

The tensorboard root (specified by tensorboard_root) controls where tensorboard summary data will be 
written during training. If a tensorboard root is not provided, then the training data will not be 
logged for tensorboard.

If an ini file is not specified, the path to the "workspace root" and "data root" can be passed to the 
scripts via the -w and -d switches respectively.

### Search Order

Juneberry will inspect several locations for a workspace and data root location until a value is found.  Only the 
first value found will be used; others will be ignored on a per value basis.  For example, one could specify 
the data_root in a juneberry.ini in the current working directory, with a workspace root in the juneberry.ini in 
the home directory.  The command line switches -w and -d would override either of these values taken from the 
ini files. The hierarchy:

1) Command line switches `-w` or `-d`.
1) ./juneberry.ini - The _current_ working directory.
1) ~/juneberry.ini - The home directory.
1) File path (not directory) specified in $JUNEBERRY_CONFIG environment variable

## Example

Juneberry provides a sample model used for a small system test. The model file is located in
[`models/imagenet_224x224_rgb_unit_test_pyt_resnet50`](./models/imagenet_224x224_rgb_unit_test_pyt_resnet50)
and uses the dataset
[`data_sets/imagenet_unit_train.json`](data_sets/imagenet_unit_train.json)
 to 'train' the model and 
[`data_sets/imagenet_224x224_rgb_unit_test.json`](./data_sets/imagenet_unit_test.json)
 to test the model.  The output is stored in 
[`models/imagenet_224x224_rgb_unit_test_pyt_resnet50`](./models/imagenet_224x224_rgb_unit_test_pyt_resnet50). 
The [`pytorchSystemTest`](./experiments/pytorchSystemTest) experiment will 
use this model, along with the data sets, to generate two different ROC curves.

# Tutorial

## Step 1 - Create a dataset specification (data spec) file
Example file: [data_sets/imagenet_unit_train.json](data_sets/imagenet_unit_train.json)

Dataset specifications follow the format described in 
[documentation/data_set_specification.md](./juneberry/documentation/data_set_specification.md) 
in the repository. These files are used to describe a dataset 
which is composed of image directories, labels, sampling criteria
and **desired** image properties. Remember that the paths are relative
to an externally specified 'data root.'

The script [jb_preview_filelist](./bin/jb_preview_filelist) can be used to preview the files that
will be selected by a specific dataset specification. Provide the dataset and config
to the script and it will output a csv ('file_list_preview.csv' by default) listing the files.

```jb_preview_filelist <data_root> data_sets/imagenet_224x224_rgb_unit_retrain.json```

## Step 2 - Create a training configuration file.
Example training config file: 
[models/imagenet_224x224_rgb_unit_test_pyt_resnet50/config.json](./models/imagenet_224x224_rgb_unit_test_pyt_resnet50/config.json)

Training specifications follow the format described in 
[documentation/training_configuration_specification.md](./juneberry/documentation/training_configuration_specification.md); 
they must be named "config.json" and must be placed in model directory inside the "models" directory.
These config files specify details such as model architecture, hyperparameters, datasets,
transforms, etc. See the specification for details on each property of the 
training config.

## Step 3 - Train

This step will demonstrate how to use the training script to train a model. The most commonly
used training script is [jb_train](./bin/jb_train), which requires a
model name (i.e. the name of a directory in the models directory containing a valid config.json 
file) as input. The output of the training process is a trained model and training metrics in an 
output JSON file.

**NOTE:** To train with TensorBoard activated, either set a system environment variable 
```TENSORBOARD_DIR="path/to/dir"``` or configure the tensorboard root in a juneberry.ini.

The training script requires the configured workspace root and data root to be set
via the ini file or via command line. The following command demonstrates how to use 
the training script to train a model, assuming the workspace and data roots are set via 
a juneberry.ini file:

```jb_train. imagenet_224x224_rgb_unit_test_pyt_resnet50```

The script provides ongoing status to the console (which can be silenced) 
as well as the model, log files and visual summaries. Some output files you may see:

* model.pt - The trained model.
* train_out.json - A JSON file containing all the output training data.
* train_out.png - Visualization of the accuracy and loss during training.
* log_train.txt - The log file from training.

Dry run mode offers an opportunity to observe what actions the training script would 
perform without actually performing the training. You can initiate dry run mode by 
adding the `--dryrun` argument to the training command. In dry run mode, Juneberry 
will log messages about what actions it would have taken using a `<< DRY_RUN >>` prefix. 
The training output will be saved to a logfile named `log_train_dryrun.txt`.

### Notes on Dataset Selection (Images)

During training, images are loaded based on the specified dataset configuration. 
The dataset config describes the sources of data to pull from, their labels, 
and how the data should be sampled. The training configuration specifies
how the validation process should occur.  The order in which these data items are 
processed is impacted by any seed values specified in the configs. These seed values 
are used when setting up randomization.

The dataset construction process works as follows:

1) The directory specified in each `data` stanza is scanned for files.
1) Each `data` file list is individually sampled based on the sampling stanza in the dataset config. 
If two or more stanzas have the same label, the sampling is per stanza file list not per the
aggregate label. 
Thus, if the two stanzas are for label 0 with a shuffling of "randomQuantity" with size 10, then 20 
   images with label 0 in total will be selected.
If the sampling involves random operations then a seed may be specified for the sampling process.
1) The file list in each data section is then split into training and validation sets 
based on the validation split stanza in the training configuration.
If the validation selection involves random operations then a seed may be specified 
for the process.
1) The training and validation sets from each entry are then merged into one list for
training and one for validation, respectively.
1) The file lists are provided to other parts of the system for cache checks, shuffling, 
   loading, and transformation.


## Step 4 - Test
At this point we want to evaluate a test set against the model and see
how well the model predicts classes. To perform a test, we need a trained
model (like the one from step 3) and a dataset config describing the data to test 
with. As with the other commands, the testing script requires workspace and data
roots to be set either via juneberry.ini or the -w/-d switches.

This example command shows how to conduct testing using the model from
the previous step and a holdout dataset. The workspace and dataroot were defined in a 
juneberry.ini, so they will not be shown in this command:

```jb_make_predictions imagenet_224x224_rgb_unit_test_pyt_resnet50 imagenet_224x224_rgb_unit_test.json```

This script will output a file called `predictions_imagenet_unit_test.json` that 
contains the labels for the test images along with the predictions the model made 
for each test image.

## Step 5 - Format Report
The data from a predictions file can be used to construct ROC plots. The example command below demonstrates 
how to plot the ROC data for particular labels in a predictions file. 
```
jb_plot_roc 
   -f models/imagenet_224x224_rgb_unit_test_pyt_resnet50/predictions_imagenet_224x224_rgb_unit_test.json 
   -p "45,128,225,319,372,402,629,778,816,959" 
   myplot.png
```
The `-f` switch is used to indicate which predictions file the data should come from. The `-p` switch indicates 
integer labels of which classes from the predictions file should be included on the plot. The `myplot.png` 
indicates the filename to use when saving the plot. It is possible to plot data from multiple predictions files 
on the same plot. To do so, you would simply add another `-f` switch and the desired file, along with another 
`-p` switch with the desired class integers.

## Step 6 - Building experiments

Example data config file: [experiments/pytorchSystemTest/config.json](./experiments/pytorchSystemTest/config.json)

As described above, subdirectories in the "experiments" directory describe separate experiments to run. 
Each experiment directory contains a config.json that describes the experiment. The config files follows 
the format specified in 
[documentation/experiment_configuration_specification.md](./juneberry/documentation/experiment_configuration_specification.md). 

There are two primary sections. The first is the "models" section which lists each model to be built and the
set of datasets to be tested against each model. Each dataset that is tested is given a tag to be used to 
refer to it in the reports section. The reports section lists a series of reports to be generated where each
report has a type (e.g. ROC curve) and a list of tags from the predictions above. Optionally, a set of 
classes can be specified for the report for when large datasets (e.g. imagenet) are used in the experiment.

Once an experiment config is created, it can be executed with the [jb_run_experiment](./bin/jb_run_experiment) 
command. As with other scripts the workspace root (-w) can be provided via juneberry.ini or via command line.  
However, the data root **is ignored** because each model may use a different data root. The experiment runner will
switch to each model directory before training or testing the model, allowing for the train or test scripts
to find juneberry configs in those directories.

Because experiments can be extremely time-consuming to run, the default mode of the script is to run in dry run
mode, which describes what actions the experiment will perform. To see what the experiment would look like 
for **pytorchSystemTest** run the following:

```jb_run_experiment pytorchSystemTest```

The output should describe the current state of the models, the predictions, and what reports it would generate.
From the logs one can look at the left most position of the log lines to see the mode of the experiment 
runner. To actually perform the action, the `--commit` option should be specified.

```jb_run_experiment pytorchSystemTest --commit```

Before a full experiment run, sometimes it is useful to clean all the artifacts. To see what would be
removed, specify the `--clean` option.

```jb_run_experiment pytorchSystemTest --clean```

By default, the `--clean` option only describes what it would do. As with running the experiment, you must  
specify the `--commit` option to perform the actual cleaning.

```jb_run_experiment pytorchSystemTest --clean --commit```

If you want to see what each model would do in detail before running them, you can run the `--dryrun`
command on each model by specifying `--dryrun` to the experiment runner.  All the appropriate
dry run logs, summaries, and sample images will be in each model. This mode is incompatible with
`--commit` and `--clean`.

```jb_run_experiment pytorchSystemTest --dryrun```



# Special Configuration Variables

## JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD

If this environment variable is set, then during training the `torch.cude.memory_summary()` shall be output after 
model load and every number of epochs based on the above period starting at epoch one. 
Thus, if JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD is set to 10, then the memory summary will be emitted 
after model load and again after epochs 1, 11, 21, 31, etc.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
