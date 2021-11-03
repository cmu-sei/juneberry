README
==========


# Introduction

Juneberry improves the experience of machine learning experimentation by providing a framework for automating 
the training, evaluation, and comparison of multiple models against multiple datasets, thereby reducing errors and 
improving reproducibility.

This README describes how to use the Juneberry framework to execute machine learning tasks. Juneberry follows a (mostly)
declarative programming model composed of sets of config files (dataset, model, and experiment configurations) and
Python plugins for features such as model construction and transformation.

This README contains:
* Steps for running a simple test of your Juneberry installation
* A tutorial on running a simple experiment by hand
* Details about building your own models and experiments

As an alternative to this README, there are some vignettes located in the `juneberry/documentation/vignettes` directory 
which offer more structured walkthroughs of some common Juneberry operations. A good start is 
[Replicating a Classic Machine Learning Result with Juneberry](./juneberry/documentation/vignettes/Replicating_a_Classic_Machine_Learning_Result_with_Juneberry.md).

# TL;DR

## How to Install Juneberry

To get started with Juneberry, run these steps from your terminal:
* `git clone https://github.com/cmu-sei/Juneberry.git`
* `cd juneberry`

Next, ensure you have Docker installed by following the installation steps
[here - (external link)](https://docs.docker.com/get-docker/).

Now you will need to build your Docker image. You can build the image using either
`docker build -f cudadev.Dockerfile -t juneberry/cudadev:dev .` or
`docker build -f cpudev.Dockerfile -t juneberry/cpudev:dev .`, depending on if you are working in a CPU or CUDA
environment.  There is a sample script you can use for building if you need to pass in proxy variables.  
See `docker/build.sh` for details.

After building your Docker image, either use the sample project directory layout or edit the mount points listed in the 
[`enter_juneberry_container`](./docker/enter_juneberry_container) script. Once the mount points are updated, 
you can run `./docker/enter_juneberry_container` from within the `juneberry` directory to launch a Docker
container.

If you have successfully entered the container, you will see a "CPU Development" or "CUDA Development" banner
followed by the current version. If an error occurs, double-check the paths you specified for the mount points 
in your`enter_juneberry_container` file and try again.

From inside the container, the final step is to run `pip install -e .`. 
**NOTE:** Because `enter_juneberry_container` creates a temporary container, you must perform this step every time 
you use `enter_juneberry_container`.

You can reference the [Docker README file](./docker/README.md) for more information.

### Other environments

Juneberry is a fully-featured Python package and can be installed into any environment, such as venv. Once Python has
been installed, simply install Juneberry using `pip install -e .` from within the Juneberry directory. This will
set up the package, including its dependencies, and add the bin directory to your path. NOTE: This requires that you
have manually followed installation instructions for MMDetection and Detectron2.

### Using bash completion
Bash completion is available for the `jb_run_experiment`, `jb_train`, and `jb_evaluate` scripts. You can source 
bash completion using `source scripts/juneberry_completion.sh`, or source the script into your .bashrc or Dockerfile.

## How to test your installation

Run the following command from within your pipenv shell or docker container:
```jb_train -w . -d . tabular_multiclass_sample```.

This should train a basic tabular dataset, and the final epoch should report a training accuracy of 97.58%.

# Experiment Overview

Juneberry uses a workspace directory structure to hold models and experiments. Most
of the configuration is done via files.

```
<WORKSPACE_ROOT>/
    data_sets/ - datasets pair data with labeling schemes
        <data-set-name>.json
    models/ - models combine architectures and hyperparameters
        <model-name>
            config.json
            ... experiment specific artifacts (logs, visualizations, etc.) ARE PLACED here ...
    experiments/ - experiments specify models and tests to evaluate and compare
        <experiment-name>
            config.json
            ... generated PyDoit files and outputs such as logs or trained models ARE PLACED here ...
    <workspace package> - [OPTIONAL] Python code goes here and should be available for import
        <subpackage>
        
        
<DATA_ROOT>/
    ... store your data in this directory ...

<TENSORBOARD_ROOT>/ - [OPTIONAL]
    <time-stamp>
    ... a timestamped directory will (optionally) be generated during each execution and filled with summary data ... 
``` 

Juneberry scripts need to know the path to the workspace, data, and (optional) tensorboard directories described above.
These paths can be provided on a per-script basis using the `-w`, `-d`, and `-t` switches respectively.

Normally, it is much more convenient to specify these (and other settings) once via a `juneberry.ini` file, described
below in "Configuration via juneberry.ini". If you wish to use a `juneberry.ini` file instead of providing directories
via command line switches, you are expected to create your own `juneberry.ini` file in an appropriate location. Refer to
the "Search Order" section of "Configuration via juneberry.ini" for the relevant details.

## WORKSPACE_ROOT Directory

The workspace path can be set via the `-w` command line switch or more generally via the `workspace_root` field in your
`juneberry.ini` file.

### Models (`models` Directory)

"Models" combine an architecture with hyperparameters to define model training using a
particular dataset (modified by transformers as needed), loss function, optimizer, and training process (e.g.
adversarial, validation splitting). Each model configuration is defined by a JSON file describing
the architecture to build, any configuration changes to the architecture, and the dataset for training the
model. 

To work with your own model, you will need to create a <model-name> sub-directory within the `models` directory and a
corresponding model config file. This sub-directory will be populated by model-specific outputs, such as the
trained model, log files, and plots generated throughout the training and/or evaluation process. This sub-directory 
can be more than one level deep under `models`. Whenever you see something refer to either "model name" or a model by
"name", it means that sub-directory path under `models`.  For example, the config file for the "unit test" for Detectron2 is in 
`models/text_detect/dt2/ut/config.json`, so the model "name" is `text_detect/dt2/ut`. 

For more details on the structure of model configs, refer to
[model_configuration_specification.md](./juneberry/documentation/model_configuration_specification.md) in the Juneberry
documentation directory.

### Datasets (`data_set` Directory)

"Datasets" describe where data elements come from and how the data should be labeled. Each dataset configuration is
defined by a JSON file. These configuration files allow you to specify subsets of source directories,
how they are ordered, and how the data should be split or transformed (such as image resizing). Together, these options
construct a unique and traceable data input set which can be referenced by your model and experiment configs.

All data paths in the JSON file should be relative to a single data root. This root is provided to Juneberry via the
`juneberry.ini` file or the `-d` command line switch.

For more details on the structure of dataset configs, refer to
[dataset_configuration_specification.md](juneberry/documentation/dataset_configuration_specification.md) 
in the Juneberry documentation directory.

### Experiments (`experiments` Directory)

"Experiments" group multiple models together for the purpose of comparing their outputs. Experiments provide the 
computational basis for scientific experiments. An experiment config JSON file defines the experiment, describing the 
models involved in the experiment, the datasets to test against those models, and the comparison reports to produce 
using those trained models.

To work with your own experiment, you will need to create an <experiment-name> sub-directory within the `experiments` 
directory and a corresponding experiment config file. This sub-directory will be populated by the experiment-specific
outputs such as log files and plots that were generated throughout the experiment process.

For more details on the structure of experiment configs, refer to
[experiment_configuration_specification.md](./juneberry/documentation/experiment_configuration_specification.md) in the
Juneberry documentation directory.

## DATA_ROOT Directory

The data root directory path can be set via the `-d` command line switch or via the `data_root` field in your
`juneberry.ini` file.

## TENSORBOARD_ROOT Directory (optional)

The tensorboard directory path can be set via the `-t` command line switch or more generally via the `tensorboard_root`
field in your `juneberry.ini` file.

## Configuration via juneberry.ini

### Content

The `juneberry.ini` file can contain the following fields:

```
[DEFAULT]
workspace_root = /path/to/workspace
data_root = /path/to/data
tensorboard_root = /path/to/tensorboard/data
```

The Juneberry framework is designed to work with a "workspace root", "data root", and/or "tensorboard root"
that is external to the Juneberry code itself. This allows the various directories (which may be small and versioned) to
be independent. 

### workspace_root
The workspace root directory (specified by `workspace_root`) is the root-level directory storing your data_sets,
models, and experiments directories. It contains references to datasets, transforms for the input data, python files for
constructing models, hyperparameters to those models, references to loss functions or optimizers, and references to 
visualizations or other reporting analytics.

### data_root
The data root directory (specified by `data_root`) is the root-level directory containing your input data. Paths in 
your dataset config files will be treated as relative to this "data root" directory.

### tensorboard_root
The tensorboard root directory (specified by `tensorboard_root`) is where your tensorboard summary data will be 
written during training. If a tensorboard root is not provided, then the training data will not be 
logged for tensorboard.

### Root switches
When an ini file is not found, the path to the "workspace root", "data root", or "tensorboard root" can be passed to
Juneberry scripts via the `-w`, `-d`, and `-t` switches respectively.

### Search Order

Juneberry will inspect several locations for a workspace and data root location until a value is found.  Only the 
first value found will be used; others will be ignored on a per-value basis.  For example, one could specify 
the data root in a `juneberry.ini` file in the current working directory, with a workspace root in the `juneberry.ini`
file in the home directory.  The command line switches `-w` and `-d` would override either of these values taken from 
the ini files. The hierarchy:

1) Command line switches `-w` or `-d`.
1) `./juneberry.ini` - The _current_ working directory.
1) `~/juneberry.ini` - The home directory.
1) File path (not a directory) specified in $JUNEBERRY_CONFIG environment variable

## Example

Juneberry provides a sample model to use for a small system test. This classification model config file is located in
[`models/imagenette_160x160_rgb_unit_test_pyt_resnet18`](./models/imagenette_160x160_rgb_unit_test_pyt_resnet18)
and uses the dataset config files
[`data_sets/imagenette_unit_train.json`](./data_sets/imagenette_unit_train.json) to 'train' the model and 
[`data_sets/imagenette_unit_test.json`](./data_sets/imagenette_unit_test.json) to test the model.  The output is stored
in [`models/imagenette_160x160_rgb_unit_test_pyt_resnet18`](./models/imagenette_160x160_rgb_unit_test_pyt_resnet18). 
The experiment config file located in [`experiments/smokeTests/classify`](./experiments/smokeTests/classify) will use 
this model, along with the datasets, to generate several ROC curves.

There are also sample object detection models that demonstrate the use of Detectron2 and MMDetection. They can 
be found in the [`text_detect/dt2`](./models/text_detect/dt2) and [`text_detect/mmd`](./models/text_detect/mmd)
directories under the Juneberry models directory.  Each has a "unit test" (`ut`) version and a "full" (`all`) version.
The "unit test" version has too few images and epochs to provide a useful model output but exercises the infrastructure
quickly, so you can evaluate if things are working properly.

# Experiment Tutorial

## Step 1 - Create a dataset config file
Example dataset config file: [data_sets/imagenette_unit_train.json](./data_sets/imagenette_unit_train.json)

For this example you will need the Imagenette data, which you can obtain by following the 
steps [here](./models/imagenette_160x160_rgb_unit_test_pyt_resnet18/README.md).

Dataset configs follow the format described in 
[dataset_configuration_specification.md](./juneberry/documentation/dataset_configuration_specification.md) in the
Juneberry documentation directory. These config files are used to describe a dataset 
which is composed of image directories, labels, sampling criteria
and **desired** image properties. Remember that the paths are relative
to an externally specified `data_root`.

## Step 2 - Create a model config file
Example model config file: 
[models/imagenette_160x160_rgb_unit_test_pyt_resnet18/config.json](./models/imagenette_160x160_rgb_unit_test_pyt_resnet18/config.json)

Model configs follow the format described in 
[model_configuration_specification.md](./juneberry/documentation/model_configuration_specification.md) in
the Juneberry documentation directory. Model configs must be named "config.json" and placed in a model sub-directory 
inside the "models" directory. These config files specify details such as model architecture, hyperparameters, datasets,
transforms, etc. See the documentation for details on each property of the model config.

## Step 3 - Train
This step demonstrates how to use the training script to train a model. The most commonly
used training script is [jb_train](./bin/jb_train), which requires a
model name (i.e. the name of a sub-directory in the "models" directory containing a valid config.json 
file) as input. The output of the training process is a trained model and training metrics in an 
output JSON file.

**NOTE:** To train with TensorBoard activated, either set a system environment variable 
```TENSORBOARD_DIR="path/to/dir"``` or configure the tensorboard root in a `juneberry.ini`.

The training script requires the configured `workspace_root` and `data_root` to be set
via the ini file or via command line. The following command demonstrates how to use 
the training script to train a model, assuming the `workspace_root` and `data_root` are set via 
a `juneberry.ini` file:

```jb_train imagenette_160x160_rgb_unit_test_pyt_resnet18```

The script provides ongoing status to the console (which can be silenced with the `-s` flag) 
as well as the model, log files, and visual summaries. Some output files you may see in the
`imagenette_160x160_rgb_unit_test_pyt_resnet18` model folder include:

* 📝**model.pt:** Your trained model with all parameters.
* 📁`train` A new folder containing training data
    * 📝**log.txt:** A text file with the command line output which you saw during training.
    * 📝**output.json:** The training data organized by type for all epochs in json format.
    * 🖼️**output.png:** An image showcasing the training and validation loss & accuracy of the most recent run. 

dryrun mode offers an opportunity to observe what actions the training script would 
perform without actually performing the training. You can initiate dryrun mode by 
adding the `--dryrun` argument to the training command. In dryrun mode, Juneberry 
logs messages about the actions it would have taken using a `<<DRY_RUN>>` prefix. 
The dryrun log output will be saved to a logfile named `log_dryrun.txt`.

### Notes on Dataset Selection (Images)

During training, images are loaded based on the specified dataset config file. The dataset config describes the sources
of data to pull from, their labels, and how the data should be sampled. The model config also defines the composition of
the validation dataset.  The order in which these data items are processed is impacted by the `seed` value specified in
the config. This seed value is used when setting up randomization.

The dataset construction process works as follows:

1) The directory specified in each `data` stanza is scanned for files.
1) Each `data` file list is individually sampled based on the sampling stanza in the dataset config. 
If two or more stanzas have the same label, the sampling is per stanza file list not per the
aggregate label. 
Thus, if the two stanzas are for label 0 with a shuffling of "random_quantity" with size 10, then 20 
   images with label 0 in total will be selected.
If the sampling involves random operations, then a seed may be specified for the sampling process.
1) The file list in each data section is then split into training and validation sets 
based on the validation split stanza in the model config.
If the validation selection involves random operations then a seed may be specified 
for the process.
1) The training and validation sets from each entry are then merged into one list for
training and one for validation, respectively.
1) The file lists are provided to other parts of the system for cache checks, shuffling, 
   loading, and transformation.


## Step 4 - Test
At this point we want to evaluate a test set against the trained model and see
how well the model performs. To perform a test, we need a trained
model, like the one from step 3, and a dataset config describing the data to test 
with. As with the other commands, the testing script requires workspace and data
roots to be set either via `juneberry.ini` or the `-w`/`-d` switches.

This example command demonstrates how to test the model trained in the previous step on a 
new dataset it has not seen before. The workspace root and data root were defined in a 
`juneberry.ini`, so they will not be shown in this command:

```jb_evaluate imagenette_160x160_rgb_unit_test_pyt_resnet18 data_sets/imagenette_unit_test.json```

This script produces the following files in the `imagenette_160x160_rgb_unit_test_pyt_resnet18` model folder:

* 📁`eval` A new folder containing eval data
    * 📁`imagenette_unit_test` A sub-directory indicating which dataset performed the eval 
        * 📝`log.txt` : A text file with the command line output which you saw during evaluation
        * 📝`metrics.json` : The evaluation metrics in JSON format
        * 📝`predictions.json` : The evaluation predictions in JSON format
    
Evaluation output files follow the format described in 
[eval_output_specification.md](./juneberry/documentation/eval_output_specification.md) in
the Juneberry documentation directory.

## Step 5 - Format Report
The data from a predictions file can be used to construct ROC plots. The example command below demonstrates 
how to plot the ROC data for particular labels in a predictions file. 

```
jb_plot_roc 
   -f models/imagenette_160x160_rgb_unit_test_pyt_resnet18/predictions_imagenette_unit_test.json 
   -p "0,217,482,491,497,566,569,571,574,701" 
   models/imagenette_160x160_rgb_unit_test_pyt_resnet18/myplot.png
```
This example command adheres to the following structure:
```
jb_plot_roc -f [prediction file path] -p [classes] [output location]
```

The `-f` switch indicates which predictions file the data should come from. The `-p` switch indicates 
integer labels of the classes to include on the plot. The `myplot.png` indicates the filename to use when 
saving the plot. It is possible to plot data from multiple predictions files 
on the same plot. To do so, simply add another `-f` switch and the desired file, along with another 
`-p` switch with the desired class integers.

## Step 6 - Building experiments

### Using jb_run_experiment:
Example data config file: [experiments/classificationSmokeTest/config.json](experiments/smokeTests/classify/config.json)

As described above, sub-directories in the "experiments" directory describe separate experiments to run. 
Each experiment directory contains a `config.json` defining the actions in the experiment, such as which models to 
train, how those models should be evaluated, and which reports should be generated. The experiment config file follows 
the format specified in 
[experiment_configuration_specification.md](./juneberry/documentation/experiment_configuration_specification.md). 

### Experiment configs:
An experiment config file consists of two primary sections:

1. The  "models" section: lists each model to be trained and the set of datasets to be tested against each 
   model. Each evaluation receives a tag for identification purposes for reference in the reports section.

2. The "reports" section: lists a series of reports to be generated where each report has a type (e.g. ROC
   curve), and a list of tags from the models section. Optionally, a set of classes can be specified for the report
   for when large datasets (e.g. imagenet) are used in the experiment.

Once an experiment config is created, it can be executed with the [jb_run_experiment](./bin/jb_run_experiment) 
command. As with other scripts, the workspace root `-w` can be provided via `juneberry.ini` or via the command line.  
However, the data root **is ignored** because each model may use a different data root. The experiment runner 
switches to each model directory before training or testing the model, allowing for the train or test scripts
to locate juneberry.ini files in those directories.

### Doit backend
Behind the scenes, `jb_run_experiment` uses the [Doit]("https://pydoit.org/) task runner. `jb_run_experiment`
generates two Doit files based on the experiment config file:
1) `main_dodo.py`: this file executes the workflow specified in the experiment config file 
2) `dryrun_dodo.py`: this file describes in-depth the actions that the `main` experiment would perform without actually
   executing them.
   
### Execution flags
* The default behavior of `jb_run_experiment` is to run in preview mode. This mode lists the tasks that would be
  executed by the workflow, along with the corresponding status messages.
  
* To actually execute these tasks and save your changes, use the `--commit` or `-X` flag. Note: Only tasks identified 
  as not "up-to-date" will be executed.

* To use dryrun (as described above in "Step 3") on the models, use the `--dryrun` or `-D` flag.

* To clean the files associated with either the main or dryrun tasks, use the `--clean` or `-C` flag.

* The `--processes` or `-N` flag specifies the number of GPUs on which you would like to run parallel model
  trainings. The default behavior runs one training process at a time across all available GPUs.
  
* To regenerate experiment config files from an experiment outline, use the `--regen` or `-R` flag.

* The abbreviated, single-character versions of these flags can be appended to form one flag, e.g. `--regen --commit
  --clean` can be specified using `-RXC`.
  
### Sample commands
The example commands below demonstrate how to use the various flags to run and clean an experiment. The
classificationSmokeTest trains three models and generates three reports plus one summary report.

Running the `dryrun` of the workflow:
* Run this command to run the classificationSmokeTest experiment in `dryrun`:
```jb_run_experiment classificationSmokeTest --commit --dryrun```
* Run this command to preview which tasks from dryrun can be cleaned:
```jb_run_experiment classificationSmokeTest --dryrun --clean```
* Run this command to actually clean the dryrun tasks:
```jb_run_experiment classificationSmokeTest -XDC```
  
Running the `main` workflow:
* Run this command to preview which Doit tasks the main workflow would execute:
```jb_run_experiment classificationSmokeTest```
* Run this command to execute the main workflow:
```jb_run_experiment classificationSmokeTest --commit```
* Run this command to clean the main tasks:
```jb_run_experiment classificationSmokeTest --commit --clean```

# Special Configuration Variables

## JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD

When this environment variable is set, the `torch.cuda.memory_summary()` will appear during training after 
the model is loaded and again after the specified period of epochs starting at epoch one.  
In other words, if JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD is set to 10, the memory summary will be emitted 
after model load and again after epochs 1, 11, 21, 31, etc.

## Further Reading

As a reminder, the `juneberry/documentation/vignettes` directory contains more detailed walkthroughs of various 
Juneberry tasks. The vignettes provide helpful examples of how to construct various Juneberry configuration files, 
including datasets, models, and experiments. A good start is 
[Replicating a Classic Machine Learning Result with Juneberry](./juneberry/documentation/vignettes/Replicating_a_Classic_Machine_Learning_Result_with_Juneberry.md).


# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
