Juneberry Configuraton Guide
==========

# Introduction

Juneberry is primarily a configuration driven system. Models, data specifications, experiments, and the like all 
have different configuration files. These files are discussed in the section on 
[Configuration File Specifications](configuring.md#configuration-file-specifications).

Most utilities accept a common set of switches or configuration variables. This is described in the section on 
[Common Configuration Variables](configuring.md#common-configuration-variables)

There are also configuration variables that *should not* necessarily impact the correctness or quality of the
output model, but may affect training time or performance. These parameters (e.g. number of gpus, number of
worker threads, etc.) are usually associated with a particular host or model, but generally don't need to be
stored with the model. This is discussed below in the 
[Workspace Config and Lap Profiles](configuring.md#lab-profiles-and-the-workspace-config) section.
We say that these *should not* impact correctness, but distributed computing is always a challenge, and some
backends have challenges with reproducibility when there are changes to the number of gpus or even when
different gpus (memory sizes) are used.

# Configuration File Specifications

The specifications for these configuration files can be found in the `docs/specs` directory. For example:

* [Model Specification](specs/model_configuration_specification.md)
* [Data Set Specification](specs/dataset_configuration_specification.md)
* [Experiment Configuration Specification](specs/experiment_configuration_specification.md)
* [Experiment Outline Specifications](specs/experiment_outline_specification.md)

See the directory `specs/docs` directory for a complete list of specifications. 

Most of the configuration files have associated schema files that can be used to ease the development
and editing of these files when using a schema-aware editor. See the directory `juneberry/schemas` for
a list of the schemas.

# Common Configuration Variables

Most of the commands (e.g. `jb_train`, `jb_evaluate`, etc.) can have the paths to the data root, workspace root,
and tensorboard root configured via environment variables or command line switches. 
See [Configuring Juneberry Paths](getting_started.md#configuring-juneberry-paths) and 
[Specifying the structure manually](getting_started.md#specifying-the-structure-manually) sections in the
[Getting Started](getting_started.md) guide.

By default, the tools keep a history of older log files with timestamps.  To disable this feature 
set `JUNEBERRY_REMOVE_OLD_LOGS` to the value `1`. 

Each tool also supports a set of command line switches that can be displayed with the `-h` or `--help` option
such as using the command `jb_train -h`.  Along with the switches described above most scripts also often support
the following:

* `-l`, `--logDir` - A directory in which to place the log file for the command.
* `-p`, `--profileName` - The name of a lab profile to use.  See the section 
    [Lab Profiles and the Workspace Config](#lab-profiles-and-the-workspace-config) below for details.
* `-s`, `--silent` - Silent flag to silence output to console.
* `-v`, `--verbose` - Verbose flag that will log DEBUG messages.

# Lab Profiles and the Workspace Config

There is often a need to be able to configure various resources such as gpus, process, threads and memory
on the host to improve performance. This sort of configuration is *incidental* to a model definition 
and is related more to the host upon which it is run. This configuration is referred to as the *workspace config*
and is stored in a special configuration file named `config.json` located at the root of a workspace.
The file contains a set of *LabProfiles* which describe the various resources for 
different hosts or host groups within the lab. LabProfiles can "include" other host profiles allowing for 
a robust system to support the management of experiments across a variety of hosts. Each profile can also 
be specified per model, to support model specific overrides, such as the number of worker threads when loading data.

## Lab Profile

A LabProfile is an extensible set of configuration values that can be accessed off the lab object inside Juneberry.
Some common properties are:
* max_gpus - If num_gpus is undefined and this value is set then the number of gpus used will be capped at
  this value.
* num_gpus - The specific (required) number of gpus to use. If this value is specified and that number of gpus cannot
  be acquired then the command will fail with an error. If None or undefined, Juneberry will try to use all available
  gpus. If set to 0, then the system will run in cpu mode. 
* num_workers - The number of worker threads to be used by data loaders or other cpu processes. (If supported). 
  The default value is 4.
* no_paging - By default Juneberry will load data within the epoch only as needed and "page" through the data. Setting
  this value to `true` will instruct Juneberry to cache the data. This can provide significant performance boost
  when trying small datasets that can fit completely within memory.

Example: A sample default LabProfile stanza that would be used inside a workspace config:

```json
{
  "max_gpus": null,
  "no_paging": false,
  "num_gpus": null,
  "num_workers": 4
}
```

**NOTE**: Other values can be added by any part of the system, so this is just a sample of common ones. Other values
will be described as needed elsewhere as needed.

## Workspace Config

The workspace config is a JSON file and is organized first by profile name, then inside that by model name.
If unspecified, the "default" profile and model are loaded.

Example 1: A sample workspace config file that defines num_gpus and num_workers for all profiles (outer `default`) 
and all models (inner `default`) is:
```json
{
  "default" : {
    "default" : {
      "num_gpus": 2,
      "num_workers": 4
    }
  }
}
```

Example 2: If one wanted to define a profile for a set of hosts named "three-gpu" for all models 
(i.e. `defualt` model) the workspace config would *contain*:

```json
{
  "three-gpu" : {
    "default" : {
      "num_gpus": 3
    }
  }
}
```

Example 3: If one wanted to define the no-paging value for a tabular model run on any profile the workspace config 
would *contain*:

```json
{
  "default" : {
    "my-tabular-model" : {
      "no_paging": true
    }
  }
}
```

### Layering of values

When a profile name is specified the LabProfile is accumulated by layering the values of more specific configurations
on top of more general configurations.  The values in increasing priority are: 

1. default:default
1. profile-name:default
1. default:model-name
1. profile-name:model-name

Where the profile-name is the one specified by `JUNEBERRY_PROFILE_NAME` or `-p`/`--profileName` switch and
model-name is the name of the current model being used.

Thus, looking back at our examples above the aggregated LabProfile Juneberry would use internally
for the "my-tabular-model" when the profile-name is specified as "three-gpu" would be:

```python
profile = {
    "no_paging": True,
    "num_gpus": 3,
    "num_workers": 4
    }
```

### Includes

As described above in the section on layering values, the various values are built up from default values. 
Instead of layering up the values from just defaults, the final profile-name and model names stanza can be applied on
top of a specific profile pair. To base a profile on top of an existing one use the `include` key in the stanza.
The `include` key takes a profile-name and model-name pair separated by a colon such as `three-gpu:my-tabular-model`.
For example, one could have a LabProfile for "tabular_models" then build a specific tabular model on top of that.

```json
{
  "default" : {
    "tabular-base" : {
      "no_paging": true
    },
    "my-tabular-model" : {
      "include": "default:tabular-base",
      "num_workers": 8
    }
  }
}
```

### Wildcard Names

The names of models can contain regular expression wildcards to allow them match a vareity of similarly
named models. For example if one had two models such as "my-tabular-1" and "my-tabular-2"
the following stanza would be used by both models.

```json
{
  "default" : {
    "my-tabular-.*" : {
      "num_workers": 8
    }
  }
}
```

# Special Configuration Variables

There are occasional instances where a few special features are controlled by special configuration variables.

## PyTorch

### JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD

When this environment variable is set, the `torch.cuda.memory_summary()` will appear during training after 
the model is loaded and again after the specified period of epochs starting at epoch one.  
In other words, if JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD is set to 10, the memory summary will be emitted 
after model load and again after epochs 1, 11, 21, 31, etc.

