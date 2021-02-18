Training Configuration Specification
==========


# Introduction

This document describes the JSON format that controls the configuration 
options for training the machine learning model using the multi-class classifier 
script. This format contains a reference to a dataset, a set of parameters
for training the architecture and training metadata. The configuration files
are an absolute path or relative to an externally provide configuration directory.
The model path is assumed to be with the import space. (e.g. relative to cwd or 
PYTHONPATH.)

# Schema
```
{
    "task": <OPTIONAL type of training to perform [classification]>,
    "batchSize":<number of samples per gradient update>,
    "dataSetConfigPath": <path to data set>,
    "testDataSetConfigPath": <path to test data set>,
    "description":<optional text description>,
    "epochs":<maximum number of epochs>,
    "stopping": {
        "threshold": <OPTIONAL minimum value reached>
        "plateau_count": <OPTIONAL integer of number of epochs with no change in value> 
        "plateau_abs_tol": <OPTIONAL floating point tolerance (psilon) value for plateau comparison>
    }
    "formatVersion":<linux style version string of the format of this file>,
    "pytorch": {
        "lossFunction": <Name of loss function in pytorch. e.g. torch.nn.CrossEntropyLoss>
        "lossArgs": <kwargs to pass to the loss function>
        "optimizer": <Pytorch optimizer : e.g. torch.optim.SGD>
        "optimizerArgs": <kwargs to pass to the optimizer>
        "lrSchedule": <A pytorch lr_scheduler : e.g. MultiStepLR>
        "lrScheduleArgs": <kwargs to pass to the lr_scheduler>
        "accuracyFunction": <OPTIONAL fully qualified accuracy function>
        "accuracyArgs": <OPTIONAL arguments to be passed to the accuracy function>
    }
    'platform": <ML Platform. [ pytorch ] with a default of pytorch>
    "modelArchitecture": {
        "module": <fully qualified class name to the model architecture class>,
        "args": <data structure passed to the model construction using keyword expansion>
        "previousModel": <OPTIONAL: name of model directory from which to load weights before training.>
        "previousModelVersion": <OPTIONAL: version of the model directory from which to load weights.>
    },
    "seed":<optional random seed>,
    "timestamp":<optional ISO time stamp for when this was generated generated>,
    "validation": {
        "algorithm": "randomFraction",
        "arguments": {
            "seed": <none or int>,
            "fraction": <float>
        }
    }
    "trainingTransforms": [
        {
            "fullyQualifiedClass": <fully qualified name of transformer class that supports __call__(image_or_tensor)>,
            "kwargs": { <kwargs to be passed (expanded) to __init__ on construction> }
        }
    ],
    "predictionTransforms": [
        {
            "fullyQualifiedClass": <fully qualified name of transformer class that supports __call__(image_or_tensor)>,
            "kwargs": { <kwargs to be passed (expanded) to __init__ on construction> }
        }
    ],
    "hints": {
        "numWorkers": <integer number of workers the system should prefer>
    }
}
```

# Details
This section provides the details of each of the fields.

## task
**Optional** This string indicates the type of training this configuration file will perform. 
Supported values for this field are: "classification". When this field is not provided, Juneberry 
will assume that the task is "classification".

## batchSize
Number of samples per gradient update.

## dataSetConfigPath
The path to the data set configuration file.

## testDataSetConfigPath
**Optional** The path to a data set configuration file describing the test images. These 
test images are evaluated at the end of every epoch, and the test error rate 
will be added to train_out.json. 

## description
**Optional** prose description of this data set.

## epochs
Number of epochs to train.

## stopping
In the basic configuration the training stops after some number of epochs. Alternatively,
the training can be stopped after a particular threshold is reached, so some value
(such as the loss) plateaus over some number of epochs.

### threshold
**Optional** validation loss threshold. If specified training will stop when this 
threshold is reached or if the maximum number of epochs occurs.

### plateau_count
**Optional** count of epochs in which the loss does not appreciably improve before stopping.
Thus, when this count is set to 5, the training will terminate early if there are 5 consecutive 
epochs with no change in the loss.

### plateau_abs_tol
**Optional** absolute loss tolerance (epsilon) for considering loss to be different between two
epochs. Two epochs are considered different if `abs(loss_a - loss_b) > plateau_abs_tol`. This value
defaults to `0.0001`.

## pytorch
Specific parameters for the PyTorch model compilation.

### deterministic
This boolean controls two options that need to be set in order for deterministic 
operation to occur when running PyTorch on a GPU. Setting this to `true` will help with 
GPU reproducibility. There may be a performance impact when enabling this option.

### lossFunction
The name of a pytorch loss function such as 'torch.nn.CrossEntropyLoss.' Note, these
are not dynamically found, but identified by string name comparison. 

## lossArgs
Keyword args to be provided to the loss function.

### optimizer
The name of a pytorch optimizer such as 'torch.optim.SGD.' Note, these
are not dynamically found, but identified by string name comparison.

### optimizerArgs
Keyword args to be provided to the optimizer function.

### lrSchedule
A string that indicates which type of PyTorch lr_scheduler to use. Juneberry currently supports 
the CyclicLR, StepLR, or MultiStepLR types.

### lrScheduleArgs
A dictionary of arguments used by the lr_scheduler. The number and type of arguments 
required will vary based on the type of scheduler. 

### accuracyFunction
OPTIONAL fully qualified accuracy function. When not specified, sklearn.metrics.accuracy_score is used.
Any provided accuracy score must take the parameters "y_pred" (array of predicted classes) and 
"y_true" an array of true classes.

### accuracyArgs
A dictionary of optional arguments to be passed to the accuracy function.

## platform
This describes the ML Platform used to perform the training. Currently, we support 
'keras' and 'pytorch'. If this value is unspecified (as in format versions prior to 2.3) 
the platform is assumed to be 'keras.'

## modelArchitecture
The trainer will instantiate a model via code. This specifies the python class
(model factory) to be instantiated and invoked (via `__call__`) to generate a model.
The `__call__` method should take the following arguments:
* img_width : width in pixels
* img_height : height in pixels
* channels : integer number of channels image. e.g. RGB is 3.
* num_classes : The number of output classes of the model

```
class <MyClass>
    def __call__(self, img_width, img_height, channels, num_classes):

        ... make the model here ...
        return model
```

### module
The fully qualified path to the class.

### args
An optional set of args as a dictionary to be passed directly into the `__call__` method
via keyword expansion.  Thus, if the `args` dictionary contains a property
'size' then the `__call__` method must have a corresponding `size` parameter. For example:

```
class <MyClass>
    def __call__(self, img_width, img_height, channels, num_classes, size):

        ... make the model here ...
        return model
```

### previousModel
This **optional** property lists another model directory from which to load weights into this model
when constructed. No checks are done for model compatibility.

### previousModelVersion
This **optional** property is used along with the previousModel property. This property controls the 
version of the previous model that will be used.

## seed
Random seed used when processing this data set. The seed 
value affects the random shuffling of the image order, executing tensorflow, and
numpy. The same seed (this value) is used for all three aspects.

## timestamp
**Optional** time stamp (ISO format) for when this config was last modified.

## validation
This stanza explains the validation strategy to be employed.

### algorithm
This can be:
* randomFraction - A random fraction is used to select a subset of images,
with optional random seed.

### arguments
For **randomFraction**:
```
{
    "seed": <optional integer seed for the randomizer>
    "fraction": <decimal fraction>
}
```

## trainingTransforms

This optional section includes a **chain** of transforms that should be applied to each 
image (training and validation) on load. Each transformer is an instance of a class
that has an `__call__` method that accepts an image or tensor and an optional 
`__init__` method that accepts an optional set of arguments.  The values to the
`__init__` method come from an optional kwargs stanza. The class should follow the
following structure:

```
class <MyTransformerClass>:
    def __init__(self, <config expanded from kwargs>):
        ... initialization code ...

    def __call__(self, image_or_tensor):
        ... transformation ...
        return image_or_tensor
```

An example can be found in Juneberry under **juneberry.transforms.debugging_transformer**.

Native pytorch transformers can be used directly with no modification such as 
**torchvision.transforms.CenterCrop**.

The transforms are called in the order they are specified in the chain, and the output
of one transformer is passed to the next as-is. Thus, if a transformer is used that returns
a `torch.Tensor` instead of a `PIL.Image` then the next transformer should accept a 
`torch.Tensor`.

At the end of the chain, all the input images are converted into `torch.Tensors` if they
have not already been converted.

Each class is constructed **once** at Juneberry start up time. The class is initialized
with the values specified in the `kwargs` parameter using "keyword" expansion. In keyword 
expansion the value of each key is passed in as that named argument, therefore all the 
values in the init call (except for self) **must** be in the `kwargs` structure, and the kwargs
structure must contain no additional arguments.

For example, if the `__init__` call looks like this:

```
class <MyTransformerClass>:
    def __init__(self, width, height):
            ... initialization code ...
```

Then the `kwargs` stanza should have `width` and `height`.  Such as:

```
{
    "kwargs": { 
        "width": 224,
        "height": 224
    }
}
```

### fullyQualifiedClass
This is the fully qualified name to the class to be loaded.
**juneberry.transforms.debugging_transformer.NoOpTensorTransformer**.

## kwargs

This stanza contains all the arguments that are to be passed into the `__init__` method of
the class.

## predictionTransforms

This section contains a **chain** of transforms to be applied to images used for validation
and testing. This section behaves identically to the 'trainingTransforms' list above, except 
for when the transforms are applied.

## hints
A section for different values that may improve that performance or general computation. These should 
not impact the output in any substantial way and may be ignored by the system based on other requirements.

### numWorkers
A hint indicating how many worker threads that is best for training this model.

## formatVersion
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
The current version: 3.3.0

* Version 3.3.0 Added the "task" field.
* Version 3.2.0 Added optional hints section.
* Version 3.1.0 Added PyTorch option "deterministic"
* Version 2.8.0 Added optional property previousModel.
* Version 2.7.0 Reworked transforms
* Version 2.6.0 added lrSchedule and lrScheduler args for PyTorch
* Version 2.5.0 added optional stanza for transformers.
* Version 2.4.0 added the platform keyword.
* Version ? added RMSProp
* Version 2.3.0 marked testDataSetConfigPath as optional
* Version 2.1 added the 'keras' section.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
