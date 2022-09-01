Model Configuration Specification
==========


# Introduction
This document describes a JSON format that controls the configuration 
options used when constructing machine learning models in Juneberry. This 
format supports a wide variety of options, including references that govern 
model construction, model training, and model evaluation. Depending on the 
desired task, you may only need to provide a subset of the fields described 
in this schema in order to produce a model configuration file that would be 
valid for the desired task.

The configuration files are an absolute path or relative to an externally 
provided configuration directory. The model path is assumed to be within the 
import space. (e.g., relative to cwd or PYTHONPATH.)

# Schema
```
{
    "batch_size": <The number of data samples to use per update when training or evaluating the model.>,
    "description": <OPTIONAL text description of the model in this file>,
    "detectron2": {
        "enable_val_loss": <OPTIONAL; bool; set to true to enable experimental validation loss computation>,
        "metric_interval": <OPTIONAL; integer; logs the training metrics to console every X iterations>,
        "overrides": [ <array of VALUES (alternating name and value) to add to the config using merge_from_list()> ]
        "supplements": [ <array of FILES to add to the config using merge_from_file()> ]
    },
    "epochs": <The maximum number of epochs to train>,
    "evaluation_metrics" : [ <array of plugins - see below> ],
    "evaluation_metrics_formatter" : <OPTIONAL; plugin to format metrics output>,
    "evaluation_transforms": [ <array of transforms - see below> ],
    "evaluation_target_transforms": [ <array of transforms - see below> ],
    "evaluator": {
        "fqcn": <fully qualified name of class that extends the juneberry.evaluation.evaluator base class>,
        "kwargs": { <OPTIONAL kwargs to be passed (expanded) to __init__ on construction> }
    }
    "format_version": <Linux style version string of the format of this file>,
    "lab_profile": <Specific host settings or limits relevant to this model.>
    "label_mapping": <OPTIONAL: A dictionary or filepath to a JSON file containing a dictionary which translates
                     the integer class numbers the model is aware of into human-readable strings.>,
    "mmdetection": {
        "load_from": <url or file to load weights from>,
        "overrides": {
            <overrides to apply to the config.  Such as "optimizer_fn.lr" or "model.roi_head.bbox_head.num_classes".>
        },
        "train_pipeline_stages": [
            {
                "name": < Name of existing stage. >,
                "stage": { <mmdetection stanza with "type" keyword> },
                "mode": <OPTIONAL: [ before (default) | replace | after | delete ] How we insert relative to "name">,
                "tupleize": <OPTIONAL: True to convert list values in stage to tuples before adding. Default is False.>
            }
        ],
        "test_pipeline_stages": [
            <same as train_pipeline_stages>
        ]
    }
    "model_architecture": {
        "fqcn": <Fully qualified class name to the model architecture class>,
        "kwargs": <Data structure passed to the model construction using keyword expansion>,
        "previous_model": <OPTIONAL: name of model directory from which to load weights before training>,
    },  
    "model_transforms" : [ <array of plugins - see below> ],
    "preprocessors": [ <array of plugins - see below> ],
    "pytorch": {
        "accuracy_args": <OPTIONAL kwargs to be passed when calling the accuracy_fn>,
        "accuracy_fn": <OPTIONAL accuracy function: e.g. sklearn.metrics.balanced_accuracy_score>,
        "loss_args": <OPTIONAL kwargs to pass when constructing the loss_fn>,
        "loss_fn": <FQCN of a loss function: e.g. torch.nn.CrossEntropyLoss>,
        "lr_schedule_args": <OPTIONAL kwargs to pass when constructing lr_scheduler_fn>,
        "lr_schedule_fn": <FQCN of a learning rate scheduler: e.g. torch.optim.lr_scheduler.MultiStepLR>,
        "lr_step_frequency": <OPTIONAL string value of "epoch" (default) or "batch" for when to 'step()' the optimizer_fn.>,
        "optimizer_args": <OPTIONAL kwargs to pass when constructing the optimizer_fn>,
        "optimizer_fn": <FQCN of an optimizer: e.g. torch.optim.SGD>,
        "strict": <OPTIONAL When loading a model, should we use strict flag. Default is true.>
    },
    "reports": [ OPTIONAL <array of Report plugins - see below ],
    "seed": <OPTIONAL integer seed value for controlling randomization>,
    "stopping_criteria": {
        "direction": <OPTIONAL direction of comparison.  Should be 'le' (default) or 'ge'.>,
        "history_key": <OPTONAL field to use for the comparison. The default is 'val_loss'.>,
        "plateau_count": <OPTIONAL integer of number of epochs with no change in value>,
        "abs_tol": <OPTIONAL floating point tolerance (epsilon) value for plateau comparison>,
        "threshold": <OPTIONAL minimum value reached>,
    },
    "summary_info": { <OPTIONAL set of descriptive properties to use when making summary outputs.> },
    "tensorflow": {
        "callbacks": [ <array of callbacks - see below>],
        "loss_args": <OPTIONAL kwargs to pass when constructing the loss_fn>,
        "loss_fn": <FQCN of a loss function: e.g. tensorflow.keras.losses.SparseCategoricalCrossentropy>,
        "lr_schedule_args": <OPTIONAL kwargs to pass when constructing lr_scheduler_fn>,
        "lr_schedule_fn": <OPTIONAL FQCN of a learning rate scheduler: e.g. tensorflow.keras.optimizers.schedules.ExponentialDecay>,
        "metrics": [ <OPTIONAL array of string names of metrics plugins. Default is "accuracy".> ],
        "optimizer_args": <OPTIONAL kwargs to pass when constructing the optimizer_fn>,
        "optimizer_fn": <FQCN of an optimizer: e.g. tensorflow.keras.optimizers.SGD>,
    },
    "timestamp": <OPTIONAL ISO timestamp for when this file was last updated>,
    "trainer": {
        "fqcn": <fully qualified name of class that extends the juneberry.trainer base class>,
        "kwargs": { <OPTIONAL kwargs to be passed (expanded) to __init__ on construction> }
    }
    "training_dataset_config_path": <The path to a dataset configuration file describing the dataset to use for training.>,
    "training_transforms": [ <array of plugins - see below> ], 
    "training_target_transforms": [ <array of plugins - see below> ],
    "validation": {
        "algorithm": <The type of algorithm to use when sampling images from the dataset to construct a validation set
                      Currently supported algorithms: ["from_file", "none", "random_fraction", "tensorflow", "torchvision"]>,
        "arguments": {
            "seed": <OPTIONAL integer seed value to use for controlling randomization during the creation of 
                     the validation portion of the dataset>,
            "fraction": <float indicating what percentage of the dataset should be set aside for the validation set>,
            "file_path": <Used with the "fromFile" validation algorithm to indicate which dataset config file to use 
                         for constructing the validation dataset>
        }
    },
}
```

## Plugin Structure

There are a variety of places in Juneberry that need to import, construct, and execute a custom
python extension.  Examples: when transforming data, construct a loss function, or training callbacks.
All these uses have a similar pattern where a class is specified with a Fully 
Qualified Class Name (FQCN) and a dictionary of keyword arguments to be passed
in during construction.  The plugin must have a `__call__(self)` method, which is invoked 
for each data input. Different transforms may have different parameters to `__call__`, depending on 
their specific use case. Refer to the details or arguments when invoking each transform property.

The general schema for a plugin is:

```
{
    "fqcn": <fully qualified name of class that supports __call__(self, *args)>,
    "kwargs": { <OPTIONAL kwargs to be passed (expanded) to __init__ on construction> }
}
```

An example of a transform with an init method that takes width and height arguments and a call method 
with an image:

```python
class MyTransformerClass:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        # Transform code here
        return image
```

The `kwargs` stanza when using this transform should have `width` and `height` properties:

```json
{
    "kwargs": { 
        "width": 224,
        "height": 224
    }
}
```

# Details
This section provides more information about each of the fields in the schema.

## batch_size
This field describes the number of samples to use during each update while training or 
evaluating the model.

## description
**Optional:** Prose description of the model.

## detectron2
Specific parameters for detectron2.  This is only used when the platform is detectron2.

### enable_val_loss
**Optional**: Set to true to enable experimental validation loss computation. Beware: The validation loss computation 
may interact with some layers during training.

### metric_interval
**Optional:** This field is an integer which controls how often detectron2 training will log the training 
metrics to console. When this argument is not provided, the trainer will log the training metrics after 
every iteration.

### overrides
**Optional:** This field is a flat array (not array of pairs) of keys and values where each key is a 
dotted name of path to the config variable, and the value is the desired value.  For example:
```
"overrides": [ "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.05, "MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE", 256 ]
```
Overrides are applied **after** any supplements (described below) or Juneberry customizations.

### supplements
**Optional:** This field is an array of files within the **workspace** that should be added to the config
using Detectron2's `merge_from_file()` functionality. Supplements are applied immediately after the
model is loaded, but before Juneberry adjusts for batch size, solver, etc. and before any `overrides` are applied.

## epochs
The number of epochs to train.

## evaluation_metrics
This section contains a list of Plugins that compute metrics on the evaluation results. Each plugin follows the general
schema for a Plugin as described above. An example of a Metrics plugin list is:

```json
{
    "evaluation_metrics": [
        {
            "fqcn": "juneberry.metrics.metrics.Coco",
            "kwargs": {
                "iou_threshold": 0.5,
                "max_det": 100,
                "tqdm": false
            }
        },
        {
            "fqcn": "juneberry.metrics.metrics.Tide",
            "kwargs": {
                "pos_thresh": 0.5,
                "bg_thresh": 0.5,
                "max_det": 100,
                "area_range_min": 0,
                "area_range_max": 100000,
                "tqdm": false
            }
        }
    ]
}
```

If no formatter is specified (see below), the result of the Metrics plugin calls is of the type Dict[Dict].
The keys of the result are the fully-qualified class names of the Metrics plugins. The value for each key of the result
is a Python Dict containing the metrics produced by that plugin. For example:

```
{
    "juneberry.metrics.metrics.Coco": {
        "mAP": 0.375,
        "mAP_small: 0.526
    },
    "juneberry.metrics.metrics.Tide": {
        "mdAP_localisation: 0.755",
        "mdAP_fp: 0.082
    }
}
```

# evaluation_metrics_formatter

This optional section contains a plugin that formats the output from the evaluation metrics. By specifying a formatter,
the default output can be modified. For example, Juneberry's default COCO formatter, `juneberry.metrics.format.DefaultCocoFormatter`,
will output something like this:

```json
{
  "bbox": {
    "mAP": 4.573400411159715e-05,
    "mAP_50": 0.00031547888650699726,
    "mAP_75": 0.0,
    "mAP_l": 0.00018100842046770595,
    "mAP_m": 6.19280358799161e-05,
    "mAP_s": 9.01350639265607e-06
  },
  "bbox_per_class": {
    "mAP_ENGLISH": 0.0,
    "mAP_HINDI": 0.0,
    "mAP_OTHER": 0.0
  }
}
```

Generally, the results of the metrics calls and formatting will be output to a JSON file (e.g. `metrics.json`) by Juneberry. 

## evaluation_transforms
This section contains a **chain** of transforms to be applied to data during validation
or testing.

*Pytorch Transform Format*

Each transform is an instance of a class that has a `__call__` method which accepts an 
image or tensor (with optional args) and an optional `__init__` method that accepts an 
optional set of arguments. 
The values to the `__call__` method are those that change on a per data element basis.
The values to the `__init__` method come from an optional kwargs stanza in a config.
The plugin should follow the following structure:

```
class <MyTransformerClass>:
    def __init__(self, <config expanded from kwargs>):
        ... initialization code ...

    def __call__(self, image_or_tensor):
        ... transformation ...
        return image_or_tensor
```

An example can be found in Juneberry under **juneberry.transforms.debugging_transformer**.

Each class is constructed **once** at Juneberry start up time. The class is initialized
with the values specified in the `kwargs` parameter using "keyword" expansion. In keyword 
expansion the value of each key is passed in as that named argument, therefore all the 
values in the init call (except for self) **must** be in the `kwargs` structure, and the kwargs
structure must contain no additional arguments.

If the transform `__init__` method has `path_label_list` as a parameter, it will be passed
a copy of the entire list of paths and labels of the input dataset.  Note, this does NOT
guarantee that this instance will be passed every entry in the list due to multi-threading
not can modify the list.  This simply allow the transform to make informed decisions about
how to transform the data based on the number and distribution of data.

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

The `__call__` method of the transform should take as its first argument the item to be transformed.
Additionally, if desired the signature can contain the `label`, item `index`, or `epoch` number. 
If found, Juneberry will pass in those values. The order of the optional arguments isn't important,
but the names must be exact.

The `__call__` method must return either `item` or an `item, label` tuple. Juneberry will accept 
either for data transforms.

For example, a `__call__` definition will all three optional arguments and returns the item and label.

```
class <MyTransformerClass>:
    def __call__(self, image, *, epoch, index, label):
        ... image transformation ...
        ... label change ...
        return image, label
```

The transforms are called in the order they are specified in the chain, and the output
of one transformer is passed to the next as-is. Thus, if a transformer is used that returns
a `torch.Tensor` instead of a `PIL.Image` then the next transformer should accept a 
`torch.Tensor`.

At the end of the chain, all the input images are converted into `torch.Tensors` if they
have not already been converted.

Native pytorch transformers can be used directly with no modification such as 
**torchvision.transforms.CenterCrop**.

### fqcn
This is the fully qualified name to the class to be loaded.
**juneberry.transforms.debugging_transformer.NoOpTensorTransformer**.

### kwargs
This stanza contains all the arguments that are to be passed into the `__init__` method of
the class.

*Detectron2 Transform Format*

When using Detectron2 transforms *can be* Detectron2 Augmentations, Transforms, or some object that looks
like a detectron2 transform by provided the appropriate methods: 

```
def apply_image(self, img: np.ndarray):
def apply_box(self, box: np.ndarray) -> np.ndarray:
def apply_coords(self, coords: np.ndarray):
def apply_polygons(self, polygons: list) -> list:
def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
```

A convenient sample transform to copy/paste can be found here: **juneberry.detectron2.transforms.DT2NoOp**. Not all 
methods are required, so you are free to remove those you don't need.

## evaluation_target_transforms
**Optional:** If the dataset refers to a torchvision dataset (one that subclasses `torch.utils.data.Dataset` and
takes 'root', 'transform' and 'target_transform' arguments), then this specifies a series of transforms to be 
applied to the **target** via the target_transforms parameter during evaluation.

## evaluator
The fully qualified class name (fqcn) and optional kwargs to a class that extends `juneberry.evaluator.EvaluatorBase`.
Juneberry has a variety of built-in evaluators for the following platforms. The built-in evaluators require
no additional kwargs.

The basic evaluators for the platforms are:
* Detectron2 - juneberry.detectron2.evaluator.Evaluator
* PyTorch - juneberry.pytorch.evaluation.evaluator.Evaluator
* MMDetection - juneberry.mmdetection.evaluator.Evaluator
* TensorFlow - juneberry.tensorflow.evaluation.evaluator.Evaluator

The following Evaluators can be used to evaluate ONNX models:
* An ONNX model trained in Juneberry using PyTorch - juneberry.onnx.pytorch.Evaluator
* An ONNX model trained in Juneberry using Tensorflow - juneberry.onnx.tensorflow.Evaluator 
* Custom Evaluator classes are also supported, e.g. juneberry.onnx.onnx_model_zoo.faster_rcnn.Evaluator

Every evaluator classes supports kwargs that allow you to specify custom classes for the various 
evaluation components. The following kwargs are recognized by the base Evaluator class:
"loader": <fully qualified name of class that describes how to construct a data loader for the evaluation>
"output": <fully qualified name of class that describes how to format the evaluation output data>
"procedure": <fully qualified name of class that describes how to conduct the evaluation>

You can construct your own classes to act as a Juneberry evaluator by assembling classes for the data loader, 
evaluation procedure, and evaluation output formatting. For an example of how to build a custom evaluator, 
refer to juneberry.onnx.onnx_model_zoo.faster_rcnn.Evaluator.

## format_version
Linux style version of the **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
The current version: 0.2.0

## lab_profile
Specific limits for this model. Refer to the workspace config section in 
[configuring.md](../configuring.md#workspace-config) for details.

## label_mapping
**Optional:** This field contains the integer class labels that the model is aware of and how those integers 
can be interpreted using human-readable strings. There are two ways to provide this information.
The first (and most direct) way is to provide a dictionary where the keys are all the integer 
classes the model is aware of, represented as strings. The value of each key should be a 
human-readable string describing the class that integer belongs to. Example:

```
{
    "0": "airplane",
    "1": "automobile",
    "2": "bird"
}
```

The second approach is to provide a path to a JSON file containing a dictionary 
named "labelNames". The "labelNames" dictionary is expected to adhere to the format 
described above. Example: 

`"label_mapping": "data_sets/label_mappings/imagenet.json"`

## mmdetection
Configuration options specific to MMDetection.

### load_from
This is an url or file path to a file (pkl or pt/pth) that contains weights for the model.  Usually this is used
to download a pre-trained model.

### overrides
A set of overrides to be applied to the config. The property names use dotted notation. For example, to change
the learning rate use "optimizer_fn.lr".  

### train_pipeline_stages
This is a list of modifications to make to the MMDetection training pipeline "data.train.pipeline".  
Each entry in this list takes the name of an existing stage (the "type" in the existing config), the new content 
to use, and a mode of how to modify the pipeline.

#### name
The name of an existing stage such as "RandomFlip" or "Normalize".

#### stage
The content to add to the pipeline. Such as `dict(type='Resize', keep_ratio=True)`. Note, this is ignored
for the "delete" mode.

#### insertMode
**Optional:** 
This mode is used to determine how to modify the pipeline relative to the stage indicated by "name".

* before - Insert before the found stage.
* replace - Replace the found stage with the provided contents
* update - Modify the existing fields with the provided ones leaving the others intact
* after - Insert after the found stage.
* delete - Delete the found stage.  Note, this does not require a "stage" entry in this stanza.

#### tupleize
**Optional:**
True to convert list values in new stage to tuples before adding. Default is False. Some fields in MMDetection
require arguments to be tuples instead of lists.

### test_pipeline_stages
The same idea as the train_pipeline_stages but instead applied to the test pipelines of "data.val.pipeline" and 
"data.test.pipeline."

## model_architecture
The trainer will instantiate a model via code. This specifies the python class
(model factory) to be instantiated and invoked (via `__call__`) to generate a model.
NOTE: all arguments are passed by name, not position, so the names must match.
The `__call__` method for image datasets should take the following arguments:

* img_width : width in pixels
* img_height : height in pixels
* channels : integer number of channels in the image.
* num_classes : The number of output classes of the model.

```
class <MyClass>
    def __call__(self, img_width, img_height, channels, num_classes):

        ... make the model here ...
        return model
```

For tabular data the `__call__` method should take:

* num_classes : The number of output classes of the model.

```
class <MyClass>
    def __call__(self, num_classes):

        ... make the model here ...
        return model
```

### module
The fully qualified path to the class/file that will construct the model.

** Detectron2 Note **
For Detectron2 the module is a path to a file either in the workspace or the Detectron2 [ModelZoo]
(https://detectron2.readthedocs.io/en/latest/modules/model_zoo.html).
First, Juneberry checks the workspace for an existing file and if found, the file is loaded using 
Detectron2's `merge_from_file()` functionality. If the file does not exist, the value is passed to the 
ModelZoo API `get_config_file()` call. 

### args
An optional set of args as a dictionary to be passed directly into the `__call__` method
via keyword expansion.  Thus, if the `args` dictionary contains a property "size" then the
`__call__` method must have a corresponding `size` parameter. For example:

```
class <MyClass>
    def __call__(self, img_width, img_height, channels, num_classes, size):

        ... make the model here ...
        return model
```

### previous_model
**Optional:** This property lists another model directory from which to load weights into this model
when constructed. No checks are done for model compatibility.

## model_transforms
**Optional:** This section includes a **chain** of transforms that will be applied to the model
after it has been constructed.  The `__call__` method will be passed the model object in the 
format of the platform (e.g., Pytorch) that was used to construct the model architecture.

Each transform is an instance of a class with a `__call__` method which accepts a model 
that was constructed or previously transformed. There may also be an optional `__init__` method 
which accepts an optional set of arguments.  The values to the `__init__` method come from an 
optional kwargs stanza. The class should have the following structure:

```
class <MyModelTransformClass>:
    def __init__(self, <config expanded from kwargs>):
        ... initialization code ...

    def __call__(self, model):
        ... model transformation ...
        return model
```

### fqcn
This is the fully qualified name of the class to be constructed.

### kwargs
This stanza contains all the arguments that are to be passed into the `__init__` method of
the transform upon construction.

## preprocessors

**Optional:** This section only applies to image datasets.

This section contains a **chain** of transforms to be applied to the filename (for classification) 
or metadata (object detection) **before** it is sampled, split, and handed to the data loader
for batching and processing.  

Each transform is an instance of a class that has a `__call__` method which accepts a 
filepath or metadata (conforming to the metadata specification) and an optional `__init__` method 
that accepts an optional set of arguments. 

The values to the `__init__` method come from an optional kwargs stanza. 
The class must return the same type as was passed in or None to skip the item.  The path or
metadata can be modified before passing back.
The class should follow this structure:

```
class <MyTransformerClass>:
    def __init__(self, <config expanded from kwargs>):
        ... initialization code ...

    def __call__(self, path_or_metadata):
        ... preprocessing ...
        return image_or_tensor
```

See **evaluation_transforms** for a discussion of how the class is loaded and kwargs are applied.

## pytorch
Specific parameters for the PyTorch model compilation.

### accuracy_args
A dictionary of optional arguments to be passed to the accuracy function.

### accuracy_fn
**Optional:** A fully qualified accuracy function. When not specified, sklearn.metrics.accuracy_score is used.
Any provided accuracy score must take the parameters "y_pred" (array of predicted classes) and 
"y_true" (array of true classes).

### deterministic
This boolean controls two options that need to be set in order for deterministic 
operation to occur when running PyTorch on a GPU. Setting this to `true` will help with 
GPU reproducibility. There may be a performance impact when enabling this option.

### loss_args
Keyword args to be provided to the loss function.

### loss_fn
The name of a PyTorch loss function such as "torch.nn.CrossEntropyLoss." These are
dynamically found so any loss function can be supplied.  However, no adapting is performed,
so the loss function must be appropriate for the model. See PyTorch documentation
for details on selecting loss functions.

### optimizer_args
Keyword args to be provided to the optimizer_fn function.

### optimizer_fn
The name of a pytorch optimizer_fn such as "torch.optim.SGD." Note, these
are not dynamically found, but identified by string name comparison.

### lr_schedule_args
A dictionary of arguments used by the lr_scheduler. The number and type of arguments 
required will vary based on the type of scheduler.  NOTE: The learning rate schedules are 
factors applied to the base learning rate of the optimizer_fn.

If the type is **LambdaLR** then the lr_schedule_args should be:

```
{
    "fqcn": <fully qualified name of class that takes an __init__(of epochs and kwargs) and __call__(epoch)>,
    "kwargs": { <kwargs to be passed (expanded) to __init__ on construction> }
}
```

So the class should look like:

```
class <MyLearningRateFunctionObject>:
    def __init__(self, epochs, <config expanded from kwargs>):
        ... initialization code ...

    def __call__(self, epoch):
        return desired_learning_rate
```

For subclasses of torch.optim.lr_scheduler._LRScheduler, the first two arguments to __init__
(the optimizer_fn and epochs) will be provided by Juneberry.

### lr_schedule_fn
A string that indicates which type of PyTorch lr_scheduler to use. Juneberry currently supports 
the CyclicLR, StepLR, MultiStepLR, LambdaLR, or a fully qualified name of a class that
extends the pytorch LRScheduler, such as torch.optim.lr_scheduler.CyclicLR. Any class specified
is automatically provided the "optimizer_fn" and, if desired, the "epochs" (with those parameter names) 
during `__init__`.

### lr_step_frequency
**OPTIONAL** This indicates how often we update (step) the learning rate scheduler. By default `step()` 
is called every epoch. To update the frequency every batch, set this property to "batch".

### strict
**OPTIONAL** Boolean that indicates whether models should be loaded as strict. Defaults to True.

## reports
**OPTIONAL** An array of one or more Report Plugins, where each Plugin corresponds to a Juneberry 
Report class, or a custom Report subclass based on the Juneberry Report base class. Refer to the 
[report config specification](report_configuration_specification.md) for more information about 
Juneberry Report Plugins.

## seed
Seed value to use when conducting Juneberry operations with this model. This seed 
value will affect random operations, including numpy.random, and it will also be 
used as the torch.manual_seed.

## stopping_criteria
In a basic configuration, the training will stop after the specified number of epochs. 
Alternatively, the training can be forced to stop when it reaches a particular threshold, 
for example when some value (such as the loss) plateaus over some number of epochs.

### direction
**Optional:** This is the "direction" of the comparison with regard to the particular value. The options
are 'le' for less than or equal to or 'ge' for greater than or equal to.  The default is 'le'.

### history_key
**Optional:** This specifies the name of the specific value to use for comparison from the
training history. By default, it is the 'val_loss'.  Supported values are 'loss', 'accuracy',
'val_loss', and 'val_accuracy'.

### plateau_count
**Optional:** count of epochs in which the loss does not appreciably improve before stopping.
Thus, when this count is set to 5, the training will terminate early if there are 5 consecutive 
epochs with no change in the loss.

### abs_tol
**Optional:** absolute loss tolerance (epsilon) for considering loss to be different between two
epochs. Two epochs are considered different if `abs(loss_a - loss_b) > abs_tol`. This value
defaults to `0.0001`.

### threshold
**Optional:** validation loss threshold. If specified, training will stop when this 
threshold is reached or when the maximum number of epochs occurs, whichever occurs first.

## summary_info
**Optional:** a dictionary produced during experiment generation that identifies which variables 
were chosen for a particular model configuration. This dictionary is primarily used to identify 
the unique properties of the model when summarizing all the models belonging to an experiment.

## tensorflow
Specific parameters for TensorFlow usage. NOTE: Fully qualified paths for tensorflow must start with
`tensorflow` and not `tf`.

### callbacks
A list of callbacks to be added to the callbacks list. Like plugins in Juneberry, these entries
consist of a FQCN and optional kwargs to be used at construction. The callbacks should subclass
`tensorflow.keras.callbacks.Callback` and are called using those APIs instead of `__call__(self)`.

### loss_fn
The fully qualified name of a TensorFlow loss function, such as "tensorflow.keras.losses.SparseCategoricalCrossentropy."

### loss_args
Keyword args to be provided to the loss function during construction.

### lr_schedule_fn
A string indicating which type of learning rate schedule function to use, such as 
"tensorflow.keras.optimizers.schedules.ExponentialDecay".

### lr_schedule_args
Keyword args to be provided to the lr_schedule_fn function during construction.

### metrics
A list of either metric names (e.g. "accuracy") or plugins that have a FQCN and optional kwargs.

### optimizer_fn
The name of an optimizer function, such as "tensorflow.keras.optimizers.SGD." If a learning rate scheduler is
specified via (lr_schedule_fn), it will be constructed and supplied to the optimizer as `learning_rate` during 
optimizer construction.

### optimizer_args
Keyword args to be provided to the optimizer_fn function during construction.

## timestamp
**Optional:** Timestamp (ISO format) indicating when the config was last modified.

## trainer
The fully qualified class name (fqcn) and optional kwargs to a class that extends `juneberry.trainer.Trainer`.
Juneberry has a variety of built-in trainers for the following platforms.  The built-in trainers require
no additional kwargs.

The basic trainers for the platforms are:
* Detectron2 - juneberry.detectron2.trainer.Trainer
* PyTorch - juneberry.pytorch.trainer.Trainer
* MMDetection - juneberry.mmdetection.trainer.Trainer
* Tensorflow - juneberry.tensorflow.trainer.Trainer

## training_dataset_config_path
The path to the dataset configuration file that describe the data to use for training the model.

## training_transforms
**Optional:** This section includes a **chain** of transforms that should be applied to input data
during training. The training transforms will be applied to the input data prior to every training epoch. 
This section behaves identically to the "evaluation_transforms" listed above, except for when 
(during training) the transforms are applied to the data.

## training_target_transforms
**Optional:** If the dataset refers to a torchvision dataset (one that subclasses `torch.utils.data.Dataset` and
takes 'root', 'transform' and 'target_transform' arguments), then this specifies a series of transforms to be 
applied to the **target** via the target_transforms parameter during evaluation.

## validation
This stanza explains the validation strategy to be employed. A validation strategy is the method 
by which a dataset will be divided into a portion for training, and a portion for validation.

### algorithm
This can be:
* from_file - The validation dataset will be constructed using a Juneberry dataset configuration file.
* none - Don't do validation.
* random_fraction - A random fraction is used to select a subset of images, with optional random seed.
* tensorflow - Only valid for datasets of type "tensorflow". In this case the validation split uses
the default "train" and "test" splits from tensorflow, unless `split` is explicitly specified in the
kwargs for construction, in which case it will use the specified split strings. 
* torchvision - Only valid for datasets of type "torchvision". In this case, the validation dataset
will be constructed using the "val_kwargs" stanza from the "torchvision_data" configuration.  The arguments
are ignored for this.

### arguments
For **random_fraction**:
```
{
    "seed": <optional integer seed for the randomizer>,
    "fraction": <decimal fraction>
}
```

For **from_file**:
```
{
    "file_path": <The path to a Juneberry dataset configuration file>
}
```

# Version History

* 0.4.0 - Updated to incorporate Reports.
* 0.3.0 - Changed from platform/task to extensible trainer/evaluator.
* 0.2.0 - Big conversion to snake case in Juneberry 0.4.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
