Workspace and Experiment Overview
==========

# Lab Structure (Common)

One of the core purposes of Juneberry is to manage all the inputs and output files associated
with a set of machine learning experiments. Similar to other experimental labs, we organize
files into a "lab structure" that has data and workspaces. A common lab structure looks 
like this:

```
<lab>
    juneberry/ - The cloned repo for tools, docs, and support scripts
    dataroot/ - Where all the data assets are stored
    tensorboard/ - OPTIONAL - Where tensorboard outputs are stored
    cache/ - A place where we store platform cached data for PyTorch, TensorFlow, etc.
    *workspace-directories*/ - e.g. juneberry-example-workspace
```

This structure is preferred but not required. Many of the support tools are designed, by default, to work with
this structure. However, especially in larger deployments, circumstances may require you to store 
files in locations on different storage devices. If a different lab structure is chosen, then the directories 
must be provided to the enter_juneberry_container script or directly to the Juneberry tools (e.g. jb_train) via the 
various configuration variables. Refer to the [Getting Started](getting_started.md) guide for details.

# Workspace (Required)

Although Juneberry only recommends a common structure for its lab layout, a specific structure is **required** 
for managing the files in a workspace. The git repository at 
[https://github.com/cmu-sei/juneberry-example-workspace](https://github.com/cmu-sei/juneberry-example-workspace)
is an example workspace with working models, data_specs, and experiments. Even though 
you can use this example workspace to test your Juneberry installation, we recommend creating a new  
workspace to manage your own models and experiments. Refer to the 
[Getting Started - Custom Workspace](getting_started.md#creating-workspaces) section for directions on how to create 
your own workspace.

While a workspace directory can have any name, it should be at the root of the lab and
its contents **MUST** adhere to the following layout:

```
<WORKSPACE_ROOT>/
    config.json - (Optional) A file that describes various machine-specific configurations used by the workspace
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
    <workspace package> - (Optional) Python code goes here and should be available for import
        <subpackage>
    setup.py - (Optional) Needed if ther
        
<DATA_ROOT>/
    ... store your input data for models in this directory ...
``` 

In most cases, each workspace will have some python code for defining the model, and perhaps transforms
or other plugins. In those situations, the workspace should have a package and corresponding setup.py file.

Most Juneberry commands will assume the current working directory is the workspace. However, the workspace 
directory path can be set via the `-w` command line switch or the `JUNEBERRY_WORKSPACE` environment
variable. See 
[Getting Started - Specifying the structure manually](getting_started.md#Specifying the structure manually) for
details.

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

## Data specs (`data_sets` Directory)

"Datasets" describe where data elements come from within the data root and how that data should be labeled.
Each dataset configuration is defined by a JSON file. These configuration files allow you to specify subsets 
of source directories, how they are ordered, and how the data should be split or transformed 
(such as image resizing). Together, these options construct a unique and traceable input dataset 
which can be referenced by your model and experiment configs.

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

# Data Root (`dataroot`) Directory

The data root directory is where all the input data is stored. Juneberry interprets any paths in the dataset configs 
as relative to this data root.

By default, the dataroot directory is assumed to be a peer of the workspace.
The data root directory path can be set via the `-d` command line switch or `JUNEBERRY_DATAROOT` environment
variable. Refer to  
[Getting Started - Specifying the structure manually](getting_started.md#Specifying the structure manually) for details.

# TensorBoard (`tensorboard`) Directory (optional)

The tensorboard directory (when specified and if it exists) will be used to store outputs to be used by TensorBoard.

By default, the tensorboard directory is assumed to be a peer of the workspace.
The tensorboard directory path can be set via the `-t` command line switch or via the `JUNEBERRY_TENSORBOARD` 
environment variable. Refer to  
[Getting Started - Specifying the structure manually](getting_started.md#Specifying the structure manually) for details.

# Example

Juneberry provides a sample model to use for a small system test in the repository 'juneberry-example-workspace'.
This classification model config file is located in
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
