Property Inference Attack Configuration Specification
==========


# Introduction

This document describes the schema for the JSON formatted attack configuration file that was 
designed to carry out a property inference attack.

The property inference attack sequence consists of three phases and two universes. The universes 
describe the full scope of dataset properties, with some properties belonging to the "superset" 
universe and the remaining properties belonging to the "disjoint" universe. 

The first phase involves training two "private" models. One private model is trained using a 
dataset augmented with a property from the "superset" universe and the training watermarks. 
The other private model is trained using a dataset with the training watermarks and a property 
from the "disjoint" universe. Each private model is also evaluated using a query dataset from its 
respective universe. The evaluation results help construct the "private" in-out datasets, 
which are used in the "meta" model phase.

The next phase involves the training and evaluation of multiple "shadow" models. The  
number of shadow models involved in this phase is user-defined; it is controlled by two 
quantity properties in the attack config. Shadow models belong to either the "superset" 
or "disjoint" universe, and there is a quantity field to control the amount of shadow 
models in each universe. Regardless of how many shadow models are involved, the general 
process for each of them follows the same pattern. First the shadow model trains using 
one of the training datasets from its respective universe. Next, the shadow model is 
evaluated using a query dataset composed of the query watermarks and the same property 
from the same universe that was used to train that particular shadow model. Once all the 
shadow models have been trained in both universes, the evaluation results are used 
by the in-out builder to construct additional in-out datasets for the "meta" phase.

The final phase of the property inference attack is the "meta" phase, which involves the 
training and evaluation of two final models, one in each universe. A meta-model gets 
trained using each of the two in-out datasets created during the "private" phase, leading 
to a "superset" meta-model and a "disjoint" meta-model. Once both models are trained, 
each meta-model is evaluated four times, using each of the in-out test datasets. There 
is a "superset" in-out test dataset from the private phase, a "superset" in-out 
test dataset from the shadow phase, a "disjoint" in-out test dataset from the 
private phase, and a "disjoint" in-out test dataset from the shadow phase. The property 
inference attack is finished once all meta-model evaluations are complete.

# Schema
```
{
    "format_version": <Linux style version string of the format of this file>,
    "models": {
        "private": <Name of model in the models directory that will be the foundatation of the private model>,
        "shadow": <Name of model in the models directory that will be the foundation of the shadow model(s)>,
        "shadow_disjoint_quantity": <The number of shadow models to create in the disjoint universe>,
        "shadow_superset_quantity": <The number of shadow models to create in the superset universe>,
        "meta": <Name of model in the models directory that will be the foundation of the meta model>
    },
    "data_configs": {
        "training_data": <The path to the dataset config file to use for the foundation of the training datasets>,
        "query_data": <The path to the dataset config file to use for the foundation of the query datasets>,
        "in_out_builder": {
            "fqcn": <Fully qualified name of a class that will build the in-out datasets>,
            "kwargs": { <OPTIONAL kwargs to be passed (expanded) to the class> }
        }
    },
    "watermarks": {
        "training_watermarks": {
                "fqcn": <Fully qualified name of a class that will watermark the training dataset>,
                "kwargs": { <OPTIONAL kwargs to be passed (expanded) to the class> }
        },
        "query_watermarks": {
                "fqcn": <Fully qualified name of a class that will watermark the query dataset>,
                "kwargs": { <OPTIONAL kwargs to be passed (expanded) to the class> }
        },
        "disjoint_args": [ { <Dictionary describing the properties belonging to the disjoint universe> } ],
        "superset_args": [ { <Dictionary describing the properties belonging to the superset universe> } ],
        "private_disjoint_args": { <A single selection from the disjoint args that will be combined with the private 
                                     model config to create the private disjoint model> },
        "private_superset_args": { <A single selection from the private args that will be combined with the private 
                                     model config to create the private superset model> }
    }
}
```

# Details

## format_version
Linux-style version of **format** of the file. Not the version of 
the data but the version of the semantics of the fields of this file. 

## models
This section describes the Juneberry model config files required to 
execute the various phases of a property inference attack.

### private
The name of a model in the model directory whose model config will serve as the 
baseline config for training the private models associated with the attack. The base 
private model config will be combined with the "private_disjoint_args" to produce 
the model config for the private disjoint model. Similarly, the base model config 
will also be combined with the "private_superset_args" to produce a model config for 
the private superset model.

### shadow
The name of a model in the model directory whose model config will serve as the baseline 
config for training the shadow models associated with the attack. This baseline config 
will be combined with various selections from the "disjoint_args" and "superset_args" 
universes to produce model configs for the shadow models.

### shadow_disjoint_quantity
Integer indicating the number of disjoint shadow models to generate during the 
property inference attack.

### shadow_superset_quantity
Integer indicating the number of superset shadow models to generate during the 
property inference attack.

### meta
The name of a model in the model directory whose model config will serve as the baseline 
config for training the meta-models associated with the property inference attack. There 
will be a meta-model config for the "disjoint" universe and another for the "superset" 
universe.

## data_configs
This section describes the dataset config files that will be required to train and evaluate 
the models during the various phases of the property inference attack.

### training_data
The name of a dataset config file that will serve as the baseline dataset config for all the 
training datasets associated with the attack. This baseline training config will be combined 
with the training watermarks plus a property from the disjoint or superset universe until each 
property in both universes is represented by its own dataset config. These dataset configs are 
used to train both the private and shadow models.

### query_data
The name of a dataset config file that will serve as the baseline dataset config for all the 
query datasets associated with the attack. This baseline training config will be combined 
with the query watermarks plus a property from either the disjoint or superset universe. A random 
seed value will also be chosen for each query dataset, so that query datasets with the same 
disjoint or superset property selection still vary to some degree. The total number of query 
datasets created will be equal to 2 (one for each private model) plus N, where N is the total 
number of shadow models requested for the property inference attack. Each query dataset is 
stored inside the model directory of the model that will use that dataset for evaluation.

### in_out_builder
The in-out builder uses model evaluation results to construct in-out datasets. The in-out 
datasets are used to train and evaluate the meta-models. In-out datasets consist of three configs, 
one for training data, one for validation data, and a third for test data. During the property 
inference attack, in-out datasets will be generated four times: once using the private superset 
eval results, once using the private disjoint eval results, once using superset shadow model results, 
and once using the disjoint shadow model results. 

#### fqcn
This is the fully qualified name of the class that will generate the in-out datasets.

#### kwargs
This stanza contains any arguments that need to be passed into the class generating the 
in-out datasets.

## watermarks
This section describes the "watermarks" to apply to the datasets involved in the property 
inference attack. A "watermark" is similar to the idea of a transform of the dataset. 

### training_watermarks
The training "watermarks" describe the watermark(s) that should be applied to the training 
datasets.

#### fqcn
This is the fully qualified name of the class responsible for applying the desired watermark(s) 
to the training datasets.

#### kwargs
This stanza contains any arguments that need to be passed into the class applying the 
watermark(s) to the training datasets.

### query_watermarks
The query "watermarks" describe the watermark(s) that should be applied to the query 
datasets.

#### fqcn
This is the fully qualified name of the class responsible for applying the desired watermark(s) 
to the query datasets.

#### kwargs
This stanza contains any arguments that need to be passed into the class applying the 
watermark(s) to the query datasets.

### disjoint_args
This list of dictionaries contains all the properties that belong to the disjoint universe 
of the property inference attack. For each dictionary in this list, a new dataset config 
will be created which combines the baseline training dataset config with a single dictionary 
from this list. Therefore, if there are M elements in the disjoint args universe, then M 
training configs will be created.

### superset_args
This list of dictionaries contains all the properties that belong to the superset universe 
of the property inference attack. For each dictionary in this list, a new dataset config will 
be created which combines the baseline training dataset config with a single dictionary from 
this list. Therefore, if there are N elements in the superset args universe, then N training 
configs will be created. If there were M training configs created from the disjoint universe, 
then the total number of training configs created will be M+N.

### private_disjoint_args
This field contains a single element from the "disjoint_args" universe. This selection defines 
which dataset config from the disjoint universe to use when training the private disjoint model.

### private_superset_args
This field contains a single element from the "superset_args" universe. This selection defines 
which dataset config from the superset universe to use when training the private superset model.

# Version History

* 0.1.0 - Initial commit

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.