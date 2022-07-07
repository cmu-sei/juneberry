Tuning Configuration Specification
==========


# Introduction
This document describes a JSON format that controls the configuration options used when performing 
hyperparameter tuning on a machine learning model in Juneberry. Since Juneberry relies on 
[Ray Tune - external link](https://docs.ray.io/en/latest/tune/index.html) to perform hyperparameter 
tuning, the fields in this config are heavily influenced by terminology used by Ray Tune. This 
[key concepts page - external link](https://docs.ray.io/en/latest/tune/key-concepts.html) offers a 
quick, high-level explanation of several tuning components that can be adjusted using values in 
the tuning config.

# Schema

```
{
    "description": <OPTIONAL text description of the tuning performed by this file>,
    "format_version": <Linux style version string of the format of this file>,
    "sample_quantity": <Integer indicating how many trials to perform during the tuning run>,
    "scheduler": {
        "fqcn": <fully qualified name of the ray.tune scheduler class to use for scheduling>,
        "kwargs": { <OPTIONAL> kwargs to be passed to the scheduler class during its construction }
    },
    "search_algorithm": {
        "fqcn": <fully qualified name of the ray.tune search algorithm class to use for hyperparameter selection>,
        "kwargs": { <OPTIONAL> kwargs to be passed to the search algorithm class during its construction }
    },
    "search_space": {
        "model_config_attribute_name": {
            "fqcn": <fully qualified name of the ray.tune sampling function to use for assigning values to the 
            model config attribute>,
            "kwargs": { <OPTIONAL> kwargs to be passed to the sampling function during its construction }
        }
    },
    "timestamp": <OPTIONAL ISO timestamp for when this file was last updated>,
    "trial_resources": {
        "cpu": <Integer indicating how many CPUs to allocate to a tuning trial>,
        "gpu": <Integer indicating how many GPUs to allocate to a tuning trial>
    },
    "tuning_parameters": {
        "checkpoint_interval": <Integer indicating how often to save model checkpoints>,
        "metric": <String indicating which metric the tuning process should optimize>,
        "mode": <String indicating how the tuning process should optimize the target metric.
                 Supported modes: ["min", "max"] >,
        "scope": <String indicating which steps of the metric to consider while optimizing.
                  Supported scopes: ["all", "last", "avg", "last-5-avg", "last-10-avg"],>
    }
}
```

# Details
This section provides more information about each of the fields in the schema.

## description
**Optional:** Prose description of the tuning performed by the config.

## format_version
Linux style version of the **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
The current version: 0.1.0

## sample_quantity
This integer indicates how many times to sample from the hyperparameter space. Each "sample" 
corresponds to a selection of hyperparameter values from the search space. The model is then 
trained for a "trial" using those hyperparameters.

## scheduler
The scheduler is responsible for running trials in the Tuner. A scheduler can cause the 
early termination of bad trials, pause trials, clone trials, and even potentially alter 
hyperparameters of a running trial. Refer to this 
[external link](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html) for more 
information about schedulers, including what schedulers are supported in Ray Tune.

If this field is not provided, Ray Tune will use the 'ray.tune.schedulers.FIFOScheduler' 
by default.

### fqcn
The fully qualified name of the scheduler class to use.

### kwargs
This stanza contains all the arguments to pass in to the scheduler during its construction.

## search_algorithm
The search algorithm is responsible for making hyperparameter selections out of the search space. 
Refer to this [external link](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html) for more 
information about search algorithms, including what search algorithms are supported in Ray Tune.

If this field is not provided, Ray Tune will use the 'ray.tune.suggest.basic_variant.BasicVariantGenerator' 
by default.

### fqcn
The fully qualified name of the search algorithm class to use.

### kwargs
This stanza contains all the arguments to pass in to the search algorithm during its construction.

## search_space
The search space defines which hyperparameters in the model config will be varied during tuning. The 
search space is represented by a dictionary, where each key name corresponds to the name of a property 
in a Juneberry Model Config. The value for the key describes which function to use for generating 
values for the indicated Model Config property. Refer to this 
[external link](https://docs.ray.io/en/latest/tune/api_docs/search_space.html) for more information 
about which functions can be used to generate values in the search space.

### model_config_attribute_name
The name of this field can be a little misleading. It is not literally 'model_config_attribute_name'. 
Instead, the key here should correspond to the string name of a property in a Juneberry Model Config 
that you would like to vary prior to training the model. Examples of such properties include (but are 
not limited to) 'batch_size', 'epochs', and 'model_architecture'. 

#### fqcn
The fully qualified name of the sampling function to use when generating values to substitute in 
for the target model config attribute.

#### kwargs
This stanza contains all the arguments to pass in to the sampling function during its construction.

## timestamp
**Optional:** Timestamp (ISO format) indicating when the config was last modified.

## trial_resources
The trial resources are used to indicate what machine resources should be allocated per 'trial'. A 
'trial' represents the state corresponding to one run of model training. The default behavior is to 
allocate 1 CPU and 0 GPUs to each trial.

### cpu
This integer indicates how many CPUs should be allocated to the trial.

### gpu
This integer indicates how many GPUs should be allocated to the trial.

## tuning_parameters
Tuning parameters control what aspects the Tuner will attempt to optimize during model training and 
how it should perform that optimization. By default, the Tuner will attempt to minimize the last 
reported training loss value.

### metric
This string indicates which training metric the Tuner should attempt to optimize. Example values for 
this field include 'loss', 'accuracy', 'val_loss', and 'val_accuracy'.

### mode
This string indicates how the Tuner should optimize the indicated metric. The only accepted values 
for this field are 'min' or 'max'.

### scope
This string indicates which steps of the indicated metric should be examined when optimizing the 
target metric. The only accepted values for this field are 'all', 'last', 'avg', 'last-5-avg', 
and 'last-10-avg'.

# Copyright

Copyright 2022 Carnegie Mellon University. See LICENSE.txt file for license terms.
