Workspace and Experiment Overview
==========

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

# WORKSPACE_ROOT Directory

The workspace path can be set via the `-w` command line switch or more generally via the `workspace_root` field in your
`juneberry.ini` file.

## Models (`models` Directory)

"Models" combine an architecture with hyperparameters to define model training using a
particular dataset (modified by transformers as needed), loss function, optimizer, and training process (e.g.
adversarial, validation splitting). Each model configuration is defined by a JSON file describing
the architecture to build, any configuration changes to the architecture, and the dataset for training the
model. 

To work with your own model, you will need to create a <model-name> sub-directory within the `models` directory and a
corresponding model config file. This sub-directory will be populated by model-specific outputs, such as the
trained model, log files, and plots generated throughout the training and/or evaluation process. This sub-directory 
can be more than one level deep under `models`. Whenever you see something refer to either "model name" or a model by
"name", it means that sub-directory path under `models`.  For example, the config file for the "unit test" for 
Detectron2 is in `models/text_detect/dt2/ut/config.json`, so the model "name" is `text_detect/dt2/ut`. 

For more details on the structure of model configs, refer to
[model_configuration_specification.md](specs/model_configuration_specification.md) in the Juneberry
documentation directory.

## Datasets (`data_set` Directory)

"Datasets" describe where data elements come from and how the data should be labeled. Each dataset configuration is
defined by a JSON file. These configuration files allow you to specify subsets of source directories,
how they are ordered, and how the data should be split or transformed (such as image resizing). Together, these options
construct a unique and traceable data input set which can be referenced by your model and experiment configs.

All data paths in the JSON file should be relative to a single data root. This root is provided to Juneberry via the
`juneberry.ini` file or the `-d` command line switch.

For more details on the structure of dataset configs, refer to
[dataset_configuration_specification.md](specs/dataset_configuration_specification.md) 
in the Juneberry documentation directory.

## Experiments (`experiments` Directory)

"Experiments" group multiple models together for the purpose of comparing their outputs. Experiments provide the 
computational basis for scientific experiments. An experiment config JSON file defines the experiment, describing the 
models involved in the experiment, the datasets to test against those models, and the comparison reports to produce 
using those trained models.

To work with your own experiment, you will need to create an <experiment-name> sub-directory within the `experiments` 
directory and a corresponding experiment config file. This sub-directory will be populated by the experiment-specific
outputs, such as log files and plots, that were generated throughout the experiment process.

For more details on the structure of experiment configs, refer to
[experiment_configuration_specification.md](specs/experiment_configuration_specification.md) in the
Juneberry documentation directory.

# DATA_ROOT Directory

The data root directory path can be set via the `-d` command line switch or via the `data_root` field in your
`juneberry.ini` file.

# TENSORBOARD_ROOT Directory (optional)

The tensorboard directory path can be set via the `-t` command line switch or more generally via the `tensorboard_root`
field in your `juneberry.ini` file.

# Configuration via juneberry.ini

## Content

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

## workspace_root
The workspace root directory (specified by `workspace_root`) is the root-level directory storing your data_sets,
models, and experiments directories. It contains references to datasets, transforms for the input data, python files for
constructing models, hyperparameters to those models, references to loss functions or optimizers, and references to 
visualizations or other reporting analytics.

## data_root
The data root directory (specified by `data_root`) is the root-level directory containing your input data. Paths in 
your dataset config files will be treated as relative to this "data root" directory.

## tensorboard_root
The tensorboard root directory (specified by `tensorboard_root`) is where your tensorboard summary data will be 
written during training. If a tensorboard root is not provided, then the training data will not be 
logged for tensorboard.

## Root switches
When an ini file is not found, the path to the "workspace root", "data root", or "tensorboard root" can be passed to
Juneberry scripts via the `-w`, `-d`, and `-t` switches respectively.

## Search Order

Juneberry will inspect several locations for a workspace and data root location until a value is found.  Only the 
first value found will be used; others will be ignored on a per-value basis.  For example, one could specify 
the data root in a `juneberry.ini` file in the current working directory, with a workspace root in the `juneberry.ini`
file in the home directory.  The command line switches `-w` and `-d` would override either of these values taken from 
the ini files. The hierarchy:

1) Command line switches `-w` or `-d`.
1) `./juneberry.ini` - The _current_ working directory.
1) `~/juneberry.ini` - The home directory.
1) File path (not a directory) specified in $JUNEBERRY_CONFIG environment variable

# Example

Juneberry provides a sample model to use for a small system test. This classification model config file is located in
[`models/imagenette_160x160_rgb_unit_test_pyt_resnet18`](../models/imagenette_160x160_rgb_unit_test_pyt_resnet18)
and uses the dataset config files
[`data_sets/imagenette_unit_train.json`](../data_sets/imagenette_unit_train.json) to 'train' the model and 
[`data_sets/imagenette_unit_test.json`](../data_sets/imagenette_unit_test.json) to evaluate the model.  The output is 
stored in 
[`models/imagenette_160x160_rgb_unit_test_pyt_resnet18`](../models/imagenette_160x160_rgb_unit_test_pyt_resnet18). The 
experiment config file located in [`experiments/smokeTests/classify`](../experiments/smokeTests/classify) will use 
this model, along with the datasets, to generate several ROC curves.

There are also sample object detection models that demonstrate the use of Detectron2 and MMDetection. They can 
be found in the [`text_detect/dt2`](../models/text_detect/dt2) and [`text_detect/mmd`](../models/text_detect/mmd)
directories under the Juneberry models directory.  Each has a "unit test" (`ut`) version and a "full" (`all`) version.
The "unit test" version has too few images and epochs to provide a useful model output but exercises the infrastructure
quickly, so you can evaluate if things are working properly.
