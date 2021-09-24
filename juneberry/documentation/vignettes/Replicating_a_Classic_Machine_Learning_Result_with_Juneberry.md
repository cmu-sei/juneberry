# Replicating a Classic Machine Learning Result with Juneberry 

Authors: Michael Vicente, Nick Winski, Violet Turri, and Nathan VanHoudnos \
Version: Juneberry 0.4 \
Date: August 2021

This vignette demonstrates how to replicate a classic machine learning result with Juneberry. The vignette focuses on 
reproducing the CIFAR-10 results reported in the He et al. (2015) paper that introduced ResNets for two reasons:

1. CIFAR-10 is both widely available and small enough to experiment on quickly. 
2. The He et al. (2015) neural network architecture for CIFAR-10 is simple to implement. It provides an instructive 
case for how to implement a new architecture in Juneberry. 

Over the course of replicating the He et al. (2015) results, you will learn how to:

* [Get started with Juneberry](#get-started-with-juneberry).
* [Load a new dataset into Juneberry with a dataset specification](#load-a-new-dataset-into-juneberry-with-a-dataset-specification).
* [Wrap a new PyTorch model for use by Juneberry](#wrap-a-new-pytorch-model-for-use-by-juneberry). 
* [Implement the training strategy outlined in a paper with a Juneberry model config](#implement-the-training-strategy-outlined-in-a-paper-with-a-juneberry-model-config).
* [Train a model with Juneberry](#train-a-model-with-juneberry) 
* [Evaluate a model with Juneberry](#evaluate-a-model-with-juneberry) 
* [Write an experiment outline to implement the various conditions within a results table in a paper](#write-an-experiment-outline-to-implement-the-various-conditions-within-a-results-table-in-a-paper) 
* [Execute an experiment with Juneberry](#execute-an-experiment-with-juneberry).
* [Compare the results of the Juneberry experiment with the published results](#compare-the-results-of-the-juneberry-experiment-with-the-published-results). 

## Get started with Juneberry

Juneberry improves the experience of machine learning experimentation by providing a framework for automating the 
training, evaluation, and comparison of multiple models against multiple datasets, reducing errors and improving 
reproducibility.

### Installation
Juneberry is open-source and can be cloned or downloaded via [Github](https://github.com/cmu-sei/Juneberry "Juneberry").
The Juneberry [README](../../../README.md) describes how to create your own Juneberry Docker container in the section 
named "How to Install Juneberry".

This vignette assumes you have been able to successfully build your container and have entered it via the 
`./docker/enter_juneberry_container` script. After running the `pip install -e .` command inside your container, 
you should be able to run the following command to perform a quick operational test of Juneberry:

```
jb_train -w . -d . tabular_multiclass_sample
```

This command trains a simple model using a tabular dataset. The training process should complete fairly quickly, and 
the log messages should report a training accuracy of 97.58% in the final epoch.


### Potential Installation Errors

The following are common errors you might experience when installing Juneberry, along with potential solutions:

> 🚨 Error: -e option requires an argument
>
> 🔨 Solution: You forgot the period "." following the e.
>
> 🚨 Error: The container appears to load, but you don't see the "CPU DEVELOPMENT" banner
>
> 🔨 Solution: Run the `enter_juneber_container` script from inside the top level project directory.
>
> 🚨 Error: The term 'docker/enter_juneberry_container' is not recognized as the name of a cmdlet, function, script file
>
> 🔨 Solution: Change the file extension to .sh and try again or manually construct the `docker run` command from 
>the `enter_juneberry_container` script and use it at the command line.

### Using Juneberry

Juneberry relies on two special directories to organize input and output. The "workspace" refers to a storage 
location for model configs, experiment configs, trained models, and logging output. The "data root" directory 
identifies the location of input data for your models. The workspace must be writeable, the data root may be read-only. 

The docker container provides a default `juneberry.ini` configuration file which specifies the location of the 
workspace and data root within the container. In general, Juneberry will attempt to locate a `juneberry.ini` file 
according to the following hierarchy:

1. `./juneberry.ini` - The current working directory.
2. `~/juneberry.ini` - The home directory.
3. A path directly to the file (not a directory) specified by the $JUNEBERRY_CONFIG environment variable.

If Juneberry locates a `juneberry.ini` file, it will attempt to read the contents of the text file and establish the 
indicated properties. The "data_root" and "workspace_root" are specified within a `juneberry.ini` file like this:

```
data_root = /path/to/your/data/store/root/directory
workspace_root = /path/to/your/workspace
```

Even when the fields are specified in a `juneberry.ini`, they can still be overridden by the `-d` or `-w` command line 
switches, as the switches are given higher priority. The `README.md` in the Juneberry repo contains more information 
about `juneberry.ini` files, command line switches, workspace and data roots, and the associated hierarchies.

## Load a new dataset into Juneberry with a dataset specification 

This section describes one way to add CIFAR-10 into a data store for use with Juneberry. 

### What is CIFAR-10?

CIFAR-10 is one of the most commonly used training datasets in the field of machine learning. The dataset comes from 
the Canadian Institute for Advanced Research and contains 60,000 32x32 color images. The images represent 10 classes 
(airplanes, birds, cars, cats, deer, dogs, frogs, horses, ships, and trucks) with each class containing 5,000 
training images and 1,000 testing images, for a total of 60,000 images.

> 📚 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html "Source") | 
> [Benchmarks](https://paperswithcode.com/sota/image-classification-on-cifar-10 "Papers with Code") | 
> [Visualize CIFAR-10](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=cifar10 "Know your data")

One way to obtain CIFAR-10 is to download it from the University of Toronto, unpickle it, and create human-readable 
labels for the 10 classes. A simpler way is to use the functions provided in `torchvision.datasets` to download the 
data and associate it with human-readable labels. 

Juneberry supports importing `torchvision.datasets` data into your data store. This can be accomplished by creating a 
dataset config in your workspace that defines the relationships between the labels, the training set, and the test set. 
When the dataset config is used for the first time, the `download` argument can be used to place a copy of the 
torchvision dataset into your data store.

![JB_FS](references/jb_cifar_torchvision_FS.PNG)

For simplicity, the "workspace_root" will be the location of the Juneberry directory as cloned from the repo. The 
"data_sets" directory within the workspace contains JSON files that define datasets for Juneberry. In this example, 
you will create a config at 📝`data_sets/torchvision/cifar10.json` to establish the CIFAR-10 dataset from 
`torchvision.datasets`. 

Start the process for creating a CIFAR-10 dataset config by navigating to the 📁`data_sets/torchvision` directory and 
creating a file named 📝`cifar10.json`. You may need to create a 📁`torchvision` sub-directory within 📁`data_sets/` 
if it does not exist.

The dataset specification file contains detailed information about dataset config fields.

| Dataset config path | Specification file |
| --------------- | --------------- |
| 📝data_sets/torchvision/cifar10.json | 📝juneberry/documentation/dataset_specification.md |

The fields responsible for establishing a CIFAR-10 dataset via `torchvision.datasets` in your 📝`cifar10.json` dataset 
config are as follows:

```json
{
    "num_model_classes": 10 ,                                 💬[indicates 10 classes in this dataset]
    "label_names": {"0": "airplane", ...},                    💬[establishes string label names for the integer class labels]
    "data_type": "torchvision",                               💬[Juneberry recognizes "torchvision", "tabular" and "image" as valid data types]
    "task_type": "classification",                            💬[Juneberry recognizes "classification" and "objectDetection" as valid task types]
    "torchvision_dataset": {                                  💬[describes how to load torchvision data]
        "fqcn": "torchvision.datasets.CIFAR10",               💬[indicates which torchvision dataset to load]
        "train_kwargs": { "train": true, "download": true },  💬[kwargs to pass to the torchvision.datasets function for training]
        "eval_kwargs": { "train": false, "download": true },  💬[kwargs to pass to the torchvision.datasets function for evaluation]
    }
}
```

Notice the kwargs in `torchvision_dataset` define which portions of CIFAR-10 to use for the training and evaluation 
phases. A `train` arg equal to true uses the training set portion of CIFAR-10, while the test set portion is chosen 
when the 'train' arg is false.

Since the training and evaluation datasets are both defined here, you will be able to use this dataset config for both 
training and evaluation. In other situations, Juneberry allows you to specify one dataset config for training and a 
different dataset config for evaluation.

At this point, try to create your own CIFAR-10 dataset config by following the outline above and referencing the 
dataset specifications file. When you are finished, the resulting config should look something like this: 

<details>
  <summary>👾Contents of a valid data_sets/torchvision/cifar10.json</summary>

```json
{
    "data_type": "torchvision",
    "description": "CIFAR-10 dataset using the torchvision.datasets",
    "format_version": "0.2.0",
    "timestamp": "2021-08-10T08:30:00",
    "label_names": {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck"
    },
    "num_model_classes": 10,
    "task_type": "classification",
    "torchvision_data": {
        "fqcn": "torchvision.datasets.CIFAR10",
        "root": "",
        "train_kwargs": {
            "train": true,
            "download": true            
        },
        "eval_kwargs": {
            "train": false,
            "download": true     
        }
    }
}
```
</details>

## Wrap a new PyTorch model for use by Juneberry

At this point, you should have configured your Juneberry environment and established a dataset configuration for 
the CIFAR-10 data. The next step is to define the neural network (NN) architecture you intend to train. 

Specifically, He et al. (2015) outlined the following structure for their CIFAR-10 style networks: 

> The network inputs are 32 x 32 images, with the per-pixel mean subtracted. The first layer is 3 x 3
> convolutions. Then we use a stack of 6n layers with 3 x 3 convolutions on the feature maps of sizes
> {32, 16, 8} respectively, with 2n layers for each feature map size. The numbers of filters are 
> {16, 32, 64} respectively. The subsampling is performed by convolutions with a stride of 2. 
> The network ends with a global average pooling, a 10-way fully-connected layer, and softmax. 
> There are totally 6n+2 stacked weighted layers. 

> When shortcut connections are used, they are connected to the pairs of 3 x 3 layers 
> (totally 3n shortcuts). On this dataset we use identity shortcuts in all cases (i.e., option A),
> so our residual models have exactly the same depth, width, and number of parameters as the plain counterparts.

> [Option] (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing 
> dimensions. This option introduces no extra parameter; ... when the shortcuts go across feature maps of two sizes, 
> they are performed with a stride of 2.

One way to implement this CIFAR style ResNet is to follow the `torchvision.models` implementation of the He et al. 
(2015) ImageNet style ResNets in the 
[resnet class](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html "ResNet"). Note that the 
ImageNet style ResNet accepts images of arbitrary size and does not normalize inputs or outputs. Therefore, using this 
pattern will require the use of other Juneberry components to handle the restriction to 32 x 32 images, the per-pixel 
mean subtraction, and the softmax normalization.

The first difference between CIFAR-10 style ResNets and the `torchvision.models` implementation is that `torchvision`
uses a projection style skip connection between the stacks of layers, while the CIFAR-10 style ResNets use the 
"Option (A)" described above. This can be implemented by modifying the downsample object instantiated in the
`_make_layer` function from 

```python
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
```
to 

```python
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = PoolPadSkip(stride)
```

where PoolPadSkip is defined as:

```python
import torch.cuda


class PoolPadSkip(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.avgpool = nn.AvgPool2d(1, stride)

    def forward(self, x):
        out = self.avgpool(x)
        device = out.get_device() if torch.cuda.is_available() else None
        pad = torch.zeros(out.shape[0], out.shape[1], out.shape[2], out.shape[3], device=device)
        return torch.cat((out, pad), 1)
```

The second set of differences between CIFAR-10 style ResNets and the `torchvision.models` implementation involve the 
feature map, filter, and kernel size. In order to follow the text of He et al. (2015), the constructor of the `ResNet` 
class must be modified as in this `CustomResNet` constructor: 

```python
class CustomResNet:

    def __init__(
        # ... skip ... 

        # The first layer is 3 x 3 convolutions.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)

        # Then we use a stack of 6n layers with 3 x 3 convolutions on the feature maps of sizes
        # {32, 16, 8} respectively, with 2n layers for each feature map size. The numbers of filters are 
        # {16, 32, 64} respectively. The subsampling is performed by convolutions with a stride of 2.
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, n, stride=2, dilate=replace_stride_with_dilation[1])
        
        # The network ends with a global average pooling, a 10-way fully-connected layer, and softmax. 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # ... skip ... 
```

The `_make_layer` function implements the stacks of layers as discussed. The rest of the implementation follows the 
`torchvision.models.resnet.ResNet` class closely, including the weight initialization. You can find the full 
implementation of the custom class at the end of this section.

Having implemented a novel neural network architecture, the next step is to write a model factory to wrap the 
architecture for use in Juneberry.  

This CIFAR-10 style ResNet architecture can be instantiated with 5 arguments:
1. `img_width`: This argument indicates the width of the input images in pixels. This should be 32.
2. `img_height`: This argument indicates the height of the input images in pixels. This should also be 32.
3. `channels`: This argument describes the color channels in the image and should be set to 3 for color images (RGB).
4. `num_classes`: This argument indicates the number of unique classes in the dataset, which should be equal to 10 for
   CIFAR-10.
5. `layers`: This argument controls the structure of the customized architecture. The expected value for this argument 
is an integer equal to 6n + 2, yielding values of 8, 14, 20, 26, 32, and so on. 

The complete model factory enforces both the 32 x 32 image restriction and checks that the number of layers is valid 
before returning the instantiated custom model:

```python
class Resnet32x32:
    def __call__(self, img_width, img_height, channels, num_classes, layers):
        if img_width != 32 or img_height != 32 or channels != 3:
            logging.error("The model only works with 32x32 RGB images.")
            sys.exit(-1)
        elif(layers - 2) % 6 != 0:
            logging.error("Layers argument missing or incorrect. (Layers - 2) % 6 must be zero for ResNet6n2.")
            sys.exit(-1)
        else:
            model = ResNetCustom(block=BB, n=int((layers - 2)/6), num_classes=num_classes)
            return model
```

Inside your Juneberry repo, navigate to the `juneberry/architectures/pytorch` directory and create a new file named 
**📝resnet_simple.py**. Add the following content to the file:

<details>
  <summary>👾File content for 'juneberry/architectures/pytorch/resnet_simple.py'</summary>

```python
import sys
import logging
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock as BB
from torchvision.models.resnet import Bottleneck as BN
from typing import Type, Union


class PoolPadSkip(nn.Module):

    # Implements "Option A" from He et al. (2015):
    #
    #    > When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options: 
    #    > (A) The shortcut still performs identity mapping, with extra zero entries padded for 
    #    > increasing dimensions. This option introduces no extra parameter; (B) The projection 
    #    > shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both 
    #    > options, when the shortcuts go across feature maps of two sizes, they are performed 
    #    > with a stride of 2.

    def __init__(self, stride):
        super().__init__()
        self.avgpool = nn.AvgPool2d(1, stride)

    def forward(self, x):
        out = self.avgpool(x)
        device = out.get_device() if torch.cuda.is_available() else None
        pad = torch.zeros(out.shape[0], out.shape[1], out.shape[2], out.shape[3], device=device)
        return torch.cat((out, pad), 1)


class ResNetCustom(nn.Module):

    def __init__(self, block, n, num_classes=10, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # The first layer is 3 x 3 convolutions.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)

        # Then we use a stack of 6n layers with 3 x 3 convolutions on the feature maps of sizes
        # {32, 16, 8} respectively, with 2n layers for each feature map size. The numbers of filters are 
        # {16, 32, 64} respectively. The subsampling is performed by convolutions with a stride of 2.
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, n, stride=2, dilate=replace_stride_with_dilation[1])

        # The network ends with a global average pooling, a 10-way fully-connected layer, and softmax. 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note: torchvision.ResNet uses 'fan_out'; He et al. (2015) uses the fan_in equivalent 
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BB, BN]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = PoolPadSkip(stride)

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, 
                        norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class Resnet32x32:
    def __call__(self, img_width, img_height, channels, num_classes, layers):
        if img_width != 32 or img_height != 32 or channels != 3:
            logging.error("The model only works with 32x32 RGB images.")
            sys.exit(-1)
        elif (layers - 2) % 6 != 0:
            logging.error("Layers argument missing or incorrect. (Layers - 2) % 6 must be zero for ResNet6n2.")
            sys.exit(-1)
        else:
            model = ResNetCustom(block=BB, n=int((layers - 2) / 6), num_classes=num_classes)
            return model

```
</details>

Once you've added the content to your `resnet_simple.py` file, you can then reference the architecture inside a 
Juneberry model config file.

## Implement the training strategy outlined in a paper with a Juneberry model config.

To replicate a machine learning result, you must fully specify how the model was trained. 

He et al. (2015) reports in Section 4.2:

> The network inputs are 32 x 32 images, with the per-pixel mean subtracted. \
> ... \
> We use a weight decay of 0.0001 and momentum of 0.9, and adopt the weight initialization 
> in [He et al., 2015b] and BN [Ioffe and Szegedy, 2015] but with no dropout. These models
> are trained with a mini-batch size of 128 on two GPUs. We start with a learning rate of 0.1, 
> divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations, which 
> is determined on a 45k/5k train/val split. We follow the simple data augmentation in [24] 
> for training: 4 pixels are padded on each side, and a 32×32 crop is randomly sampled from 
> the padded image or its horizontal flip. For testing, we only evaluate the single view of 
> the original 32 x 32 image.

Let's take a closer look at the individual parameters defined in this excerpt. 

Juneberry is an epoch based trainer, so the number of iterations must be converted to epochs. There are 50,000 training 
images, but if 10% of them are held out for the validation set, then 45,000 images remain. Dividing 45,000 by a batch 
size of 128 yields 352 training iterations to complete one epoch. The paper mentions training terminated after 64,000 
iterations, so dividing that number by 352 means training lasted for 182 epochs. You can apply similar math to 
determine which epochs should trigger an adjustment in the learning rate.

### Training Parameters from He et al. (2015)

> **Batch Size:** 128
> 
> **Epochs:** 182
>
> **Learning Rates:**  0.1 for epochs [1-91], 0.01 for epochs [92-137], 0.001 for epochs [138-182]
>
> **Weight Decay:** 0.0001
>
> **Momentum:** 0.9
>
> **Training/Validation split:** 90% to 10%
>
> **Augmentation:** random horizontal flip, padding to 36x36, then a random 32x32 crop.

A model configuration file defines how Juneberry will construct and train a machine learning model. A model config 
contains [several variables](#a-closer-look-at-the-model-config) defining various aspects of the model, such as the 
model architecture, the training dataset, and other parameters. To reproduce the results of He et al. (2015), you will 
build a model config to match the training parameters summarized above. 

### Building the Model Config

Juneberry expects model configuration files to be stored in a subdirectory within your workspace called 📁`models`. To 
build the model config demonstrated in this vignette:

* Create a directory within the 📁`models` directory called 📁`cifar_R20`. This shorthand name describes a model that 
trains a 20 layer ResNet using CIFAR-10 data.
* Inside the 📁`cifar_R20` directory, create a file called 📝`config.json`. This will be your model config file.
* The last step is to add content to the model config file. 

Model config files also have a specification file which defines all the fields supported inside a Juneberry model 
configuration file. For more information about any model config fields, refer to the following specification file:

| Model config path | Specification file |
| --------------- | --------------- |
| 📝models/cifar_R20/config.json | 📝juneberry/documentation/model_configuration_specification.md |

### A Closer Look at the Model Config

The code block below shows many common Juneberry model config file fields, along with a brief description of each 
field's purpose. The next few sections describe how to fill out these fields to produce a PyTorch model for training a 
20-layer ResNet using CIFAR-10 data.

```json
{
    "format_version": <A Juneberry specific variable identifying the format version of this file.>,
    "description": <A string summarizing the purpose of this configuration.>,
    "timestamp": <yyyy-mm-ddThr:min:sec format for when this config was created.>,
    "platform": <The machine learning platform the config is compatible with.>,
    "model_architecture": {<The model architecture and associated args.>},
    "seed":  <A numerical value used for randomization.>,
    "batch_size": <The number of samples viewed between updates of the model's internal parameters.>,
    "task":  <The type of data processing task; could be classification or object detection.>,
    "hints": {<Properties that relate to performance or general computation.>},
    "evaluation_procedure": <The path to a class defining the evaluation procedure.>,
    "evaluation_output": <The path to a class defining the format of the evaluation output.>, 
    "training_dataset_config_path": <The path to the dataset configuration file to use for training.>,
    "validation": {<The method for establishing a validation dataset.>},
    "training_transforms": [<A list of transforms performed on data prior to training.>],
    "epochs":  <The number of epochs for which to train.>,
    "pytorch": {Note: These fields are specific to PyTorch.
        "loss_fn": <The loss function used in training.>,
        "optimizer_fn": <The optimizer used in training.>,
        "optimizer_args": {
            "lr": <The learning rate, i.e. how sensitive your model is to change.>,
            "momentum": <Used during training to adjust weights based on a gradient.>,
            "weight_decay": <Decreases impact of the weights from prior iterations.>
        },
        "lr_schedule_fn": <The schedule to follow when adjusting the learning rate.>,
        "lr_schedule_args": {
            "milestones": [<The epochs where the learning rate will adjust.>], 
            "gamma": <The multiplier for adjusting the learning rate.>
        }
    }
}
```

#### The Model Architecture

The model architecture portion of the model config establishes the structure of the neural network. This section will 
rely on the neural network architecture file and class created in the 
[model architecture](#wrap-a-new-pytorch-model-for-use-by-juneberry) section of this vignette. 

The code below demonstrates how to implement the `model_architecture` field to produce a 20-layer CIFAR-10 style ResNet 
in He et al. (2015). Juneberry uses JSON configs to manage its workflows. In the model config, there is a 
`model_architecture` stanza that takes the fully qualified name of callable factory class, along with the various 
arguments to pass to the factory during construction. 

The following stanza shows the desired configuration for the model architecture portion of the cifar_R20 model config: 
```json
    "model_architecture": {
        "module": "juneberry.architectures.pytorch.resnet_simple.Resnet32x32",
        "args": {
            "img_width": 32,
            "img_height": 32,
            "channels": 3,
            "num_classes": 10,
            "layers": 20
        }
    }
```
* The `module` field references the class from the `resnet_simple.py` file. 
* The `args` field contains values for the five arguments to pass to the class during construction. 
* The `img_width`, `img_height`, and `channels` fields all describe properties of the input images (size and number of 
channels). 
* The `num_classes` field matches the 10 possible classes in the CIFAR-10 dataset. 
* The value in the `layers` field indicates the desired number of layers for the ResNet. Note that a value of 20 
satisfies the 6n + 2 constraint discussed in the neural network architecture section.

#### Data Transforms

The He et al. (2015) paper describes several techniques for augmenting the images prior to training. Juneberry
implements these augmentations in the form of transformations for the input data. Just like model architectures, 
Juneberry supports both torchvision transforms and user-defined transform classes, typically stored in 
`juneberry/transforms`.

The transforms stanzas of the model config will represent the image pre-processing described in the paper. 

* **RandomCropMirror** adds 4 pixels of padding around the image, performs a random 32x32 crop of the padded result, 
and has a 50% chance of mirroring the resulting cropped image.
* **SizeCheckImageTransform** confirms the resulting image is 32x32 pixels in size.
* **ToTensor** converts the image into tensor format.
* **Normalize** normalizes the image using the specified mean and standard deviation per channel. We use the 
published CIFAR-10 means and standard deviations.

```json
    "training_transforms": [
        {
            "fqcn": "juneberry.transforms.random_crop_mirror.RandomCropMirror",
            "kwargs": {
                "width_pixels": 4,
                "height_pixels": 4,
                "mirror": 1
            }
        },
        {
            "fqcn": "juneberry.transforms.debugging_transforms.SizeCheckImageTransform",
            "kwargs": {
                "width": 32,
                "height": 32,
                "mode": "RGB"
            }
        },
        {
            "fqcn": "torchvision.transforms.ToTensor"
        },
        {
            "fqcn": "torchvision.transforms.Normalize",
            "kwargs": {
                "mean": [
                    0.4914,
                    0.4822,
                    0.4465
                ],
                "std": [
                    0.2023,
                    0.1994,
                    0.2010
                ]
            }
        }
    ],
```

The `fqcn` field, short for Fully Qualified Class Name, indicates the desired transform class. The example above 
demonstrates the use of two custom Juneberry transforms located in `juneberry.transforms`, as well as two transforms
directly from `torchvision.transforms`. More Torchvision transformations are available
[here](https://pytorch.org/vision/stable/transforms.html  "View Transforms").

The evaluation images are converted to tensors and normalized for compatibility with the model.

```json
    "evaluation_transforms": [
        {
            "fqcn": "torchvision.transforms.ToTensor"
        },
        {
            "fqcn": "torchvision.transforms.Normalize",
            "kwargs": {
                "mean": [
                    0.4914,
                    0.4822,
                    0.4465
                ],
                "std": [
                    0.2023,
                    0.1994,
                    0.2010
                ]
            }
        }
    ],
```

#### Training Parameters

The next collection of Juneberry model config fields describe properties used during model training. This section 
focuses primarily on the following lines from He et al. (2015):

> These models are trained with a mini-batch size of 128 on two GPUs. We start with a learning rate of 0.1, divide it by
> 10 at 32k and 48k iterations, and terminate training at 64k iterations, which is determined on a 45k/5k train/val
> split.

One thing that isn't clear is if each GPU receives 128 images, or if that batch size should be split across the two 
GPUs so that each GPU receives 64 images. Digging into the 
[code released by He et al. (2015)](https://github.com/KaimingHe/deep-residual-networks), we found the latter 
interpretation to be true. The computation of the mini-batches is parallelized across GPUs so each GPU receives 64 
images at a time.

Juneberry defines its batch size the same way; it refers to the size of the mini-batches that are parallelized
across the available GPUs.

For example, if you have two GPUs available, you would specify the config as follows:

```json
    "batch_size": 128,
    "hints": {
        "max_gpus": 2    
    },
    "epochs": 182,
    "pytorch": {
        "loss_fn": "torch.nn.CrossEntropyLoss",
        "optimizer_fn": "torch.optim.SGD",
        "optimizer_args": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001
        },
        "lr_schedule_fn": "torch.optim.lr_scheduler.MultiStepLR",
        "lr_schedule_args": {
            "milestones": [92, 138], 
            "gamma": 0.1
        }
    },
```

The **loss function** and **optimizer** are PyTorch classes configured to match the parameters described in the paper. 
Furthermore, the optimizer arguments capture the starting value for the **learning rate**, as well as the **momentum** 
and **weight decay** listed in the paper. 

The paper also indicates the learning rate changed twice after a certain number of iterations, so MultiStepLR 
is an appropriate choice for the learning rate schedule. The milestones indicate the epochs when the learning rate 
should change, and the gamma value indicates how much the learning rate should change. In this case, a gamma value of 
0.1 will divide the current learning rate by 10 when the epoch milestones are reached.

#### Validation Dataset

The `validation` field defines the validation dataset used during training. The `algorithm` field indicates if the 
validation set should be built from a dataset config file or split from the training dataset. The example code below 
indicates ten percent of the training dataset will be randomly selected and used for validation. The `seed` field 
affects which inputs are chosen at random. Setting the seed value to a fixed integer ensures the same images are 
consistently chosen for the validation set each time the model config is used for training.

```json
    "validation": {
        "algorithm": "randomFraction",
        "arguments": {
            "fraction": 0.1,
            "seed": 31415
        }
    },
```

#### Model Summary

A useful practice for replicable research is to log the model summary each run. This can help you catch possible inaccuracies in the way your model architecture 
was implemented. You can add the following model transform to your model config file to add a summary of the constructed 
model architecture to the training log output:

```json
"model_transforms": [
    {
        "fqcn": "juneberry.pytorch.model_transforms.LogModelSummary",
        "kwargs": {
            "imageShape": [
                3,
                32,
                32
            ]
        }
    }
    ]
```

----------

#### Complete Implementation of CIFAR-10 Model Configuration

Assuming 2 GPUs are available, you can assemble the previously described sections into the contents of a single JSON, 
and it should look something like the following block of code:

<details>
  <summary>👾Contents of models/cifar_R20/config.json</summary>

```json
{
    "format_version": "0.2.0",
    "description": "CIFAR-10 ResNet Model Configuration",
    "timestamp": "2021-06-11T10:00:00",
    "platform": "pytorch",
    "model_architecture": {
        "module": "juneberry.architectures.pytorch.resnet_simple.Resnet32x32",
        "args": {
            "img_width": 32,
            "img_height": 32,
            "channels": 3,
            "num_classes": 10,
            "layers": 20
        }
    },
    "seed": 31415,
    "batch_size": 128,
    "task": "classification",
    "hints": {
        "num_Workers": 0,
        "max_gpus": 2
    },
    "evaluation_procedure": "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationProcedure",
    "evaluation_output": "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationOutput",
    "training_dataset_config_path": "data_sets/torchvision/cifar10.json",
    "validation": {
        "algorithm": "random_fraction",
        "arguments": {
            "fraction": 0.1,
            "seed": 31415
        }
    },
    "training_transforms": [
        {
            "fqcn": "juneberry.transforms.random_crop_mirror.RandomCropMirror",
            "kwargs": {
                "width_pixels": 4,
                "height_pixels": 4,
                "mirror": true
            }
        },
        {
            "fqcn": "juneberry.transforms.debugging_transforms.SizeCheckImageTransform",
            "kwargs": {
                "width": 32,
                "height": 32,
                "mode": "RGB"
            }
        },
        {
            "fqcn": "torchvision.transforms.ToTensor"
        },
        {
            "fqcn": "torchvision.transforms.Normalize",
            "kwargs": {
                "mean": [
                    0.4914,
                    0.4822,
                    0.4465
                ],
                "std": [
                    0.2023,
                    0.1994,
                    0.2010
                ]
            }
        }
    ],
    "evaluation_transforms":[
        {
            "fqcn": "torchvision.transforms.ToTensor"
        },
        {
            "fqcn": "torchvision.transforms.Normalize",
            "kwargs": {
                "mean": [
                    0.4914,
                    0.4822,
                    0.4465
                ],
                "std": [
                    0.2023,
                    0.1994,
                    0.2010
                ]
            }
        }
    ],
    "model_transforms": [
    {
        "fqcn": "juneberry.pytorch.model_transforms.LogModelSummary",
        "kwargs": {
            "imageShape": [
                3,
                32,
                32
            ]
        }
    }
    ],
    "epochs": 182,
    "pytorch": {
        "loss_fn": "torch.nn.CrossEntropyLoss",
        "optimizer_fn": "torch.optim.SGD",
        "optimizer_args": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001
        },
        "lr_schedule_fn": "torch.optim.lr_scheduler.MultiStepLR",
        "lr_schedule_args": {
            "milestones": [92, 138], 
            "gamma": 0.1
        }
    }
}
```
</details>
<br/>

After adding this content into your `models/cifar_R20/config.json` file, you will finally have all the pieces you need 
to begin training a model in Juneberry.

## Train a model with Juneberry. 

With the required components in place, you can begin training your 20-layer ResNet model with the following command:

```
jb_train cifar_R20
```

This command tells Juneberry to run the training script using the model config located in the `cifar_R20` sub-directory 
within the `models` directory of your workspace. You should see the following log messages (and many others) during 
training:

> **Perform data transforms:**
>
> ​	>INFO Constructing transform: juneberry.transforms.random_crop_mirror.RandomCropMirror
>
> **Construct the model:**
>
> ​	>INFO Constructing the model juneberry.architectures.pytorch.resnet_simple.Resnet32x32
>
> **Begin training:**
>
> ​	>INFO Starting Training...
>
> **Finalize and save output:**
>
> ​	>INFO Writing output file: models/cifar_R20/train_out.json


### Training Output:

While the model trains, Juneberry will log training metrics at the end of every epoch. The following message format 
depicts the metrics output shown for a typical PyTorch classifier.

```
INFO 1/182, time_sec: 15.489, eta: 13:40:37, remaining: 0:42:04, lr: 1.00E-01, loss: 1.7700, accuracy: 0.3359, val_loss: 1.5658, val_accuracy: 0.4127
```

* **1/182:** The current epoch number out of the total number of epochs.
* **time_sec:** The amount of time it took to train this epoch.
* **ETA:** The estimated time training will be complete.
* **Remaining:** The amount of time estimated before training is complete.
* **lr**: The learning rate used in the most recent batch.
* **loss:** The average loss across the batches in the epoch.
* **accuracy:** The average of batch accuracies across the epoch.
* **val_loss:** The average loss across the validation data.
* **val_accuracy:** The accuracy of the model on the validation data.

The final epoch of training the 20-layer ResNet using CIFAR-10 data should look something like this:
```
INFO 182/182, time_sec: 12.918, eta: 13:34:32, remaining: 0:00:00, lr: 1.00E-03, loss: 0.0064, accuracy: 0.9989, val_loss: 0.3518, val_accuracy: 0.9244
```

### Post-Training Files:

At the conclusion of training, the following files and directories will appear in your 📁`models/cifar_R20` directory:

* 📝**model.pt:** The model weights for the model with the best validation loss.
* 📁`train`
    * 📝**log.txt:** A text file containing the messages logged during training.
    * 📝**output.json:** A JSON file containing data about the training session.
    * 🖼️**output.png:** A plot of the training and validation loss and accuracy of the most recent training run.

The 🖼️**output.png:** plot, shown below, depicts how the  training and validation loss and accuracy metrics changed 
over the training session. Figure 1 of He et al. (2015) depicts the training error and test error of the 20-layer 
network as a function of the iteration number. Since accuracy is equal to one minus the error, the curves in the 
following plot would resemble Figure 1 of the paper if you were to invert the accuracy curves. Note that they 
qualitatively match the results reported in He et al. (2015). 

![Error Plot](references/output.png)

## Evaluate a model with Juneberry.

Once a trained model file exists, it can be evaluated with the `jb_evaluate_data` command. The command takes two 
arguments: the name of the trained model and the path to a Juneberry dataset config describing the dataset to be 
evaluated.

You can use the following command to initiate the evaluation:
```
jb_evaluate_data cifar_R20 data_sets/torchvision/cifar10.json
```

This command follows the following structure:

```
jb_evaluate_data [model to evaluate] [dataset config to evaluate]
```

The "model to evaluate" corresponds to the name of a sub-directory inside the `models` directory which holds the model 
you would like to evaluate. The "dataset config to evaluate" refers to the Juneberry dataset config file describing the 
dataset you would like the model to evaluate. 

#### Evaluation Method

When you added the file content to the model config, you may have noticed the following two lines:

```json
    "evaluation_procedure": "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationProcedure",
    "evaluation_output": "juneberry.pytorch.evaluation.evals.default.DefaultEvaluationOutput"
```

The `evaluation_procedure` indicates the class Juneberry should use to evaluate the data. The `evaluation_output` field 
indicates a class Juneberry should use to format the raw evaluation data produced by the evaluation procedure. In both 
cases, this vignette relies on the default classes for a Pytorch classifier that have been implemented in Juneberry.

#### Evaluation Output

When you conduct the indicated evaluation, you should see something similar to the following in your evaluation output 
log messages:
```
INFO ******          Accuracy: 0.9189
INFO ****** Balanced Accuracy: 0.9189
INFO Emitting output to: models/cifar_R20/predictions_cifar10.json
```
While the exact values for the accuracy metrics reported here may vary from machine to machine, your result should be 
reasonably close to these values.
<br />

At the end of an evaluation, Juneberry creates an 📁`eval` sub-directory within the model directory. Inside the eval 
directory, there will be one or more sub-directories matching the name of datasets that were evaluated. This directory 
structure allows Juneberry to keep evaluation results separate when the same model has evaluated multiple datasets.

* 📁`eval`
    * `cifar10`: This directory name matches the base name of the dataset config used to evaluate the model.
        * 📝`log.txt`: A text file containing the messages logged during the evaluation.
        * 📝`metrics.json`: A JSON file containing data about the evaluation session.
        * 📝`predictions.json`: The raw evaluation data produced during the evaluation.
<br />

### Generating Plots: 

After an evaluation, Juneberry offers some basic support for common plots. ROC curves are one of the supported plot 
types. You can generate this type of plot with the following command:

```
jb_plot_roc -f models/cifar_R20/eval/cifar10/predictions.json -p all models/cifar_R20/cifar10_roc.png
```

This command follows the following structure:

```
jb_plot_roc -f [predictions file path] -p [classes] [output location]
```

The predictions file path indicates the location of the predictions file containing the data for generating the ROC 
curves. The `-p` indicates which classes to draw curves for. "all" is a supported keyword which produces curves for all 
known classes. The final argument, the output location, indicates where to save the resulting plot file. When you run 
this command, it should produce a plot similar to the following figure:

![Example](references/example_plot.PNG)

At this point you have completed the training and evaluation phases for a single model in Juneberry. In the next 
section, you will see how to group multiple models and datasets together to perform bulk training and evaluation 
without having to manually provide individual training and evaluation commands.

## Write an experiment outline to implement the various conditions within a results table in a paper. 

An experiment outline file defines how Juneberry should construct several related model configs, train those 
models, perform evaluations of the trained models, and then summarize the results. 

### Experiment Outline Schema

The code block below contains many of the common fields found in a Juneberry experiment outline file.

```json
{
    "baseline_config": <The name of a model in the model directory.>,
    "description": <A human-readable description (or purpose) of this experiment.>,
    "filters": [<The list of experiment filters to include verbatim.>]
    "format_version": <A Linux-style version string of the format of this file.>,
    "model" : {
        "filters": [<A list of filters to add to each model.>]
    }
    "reports": [
        {
            "type": <The report type: [plot_roc | plot_pr | summary | all_roc | all_pr].>,
            "description": <A brief description of this report.>,
            "test_tag": <REQUIRED, type must be plot_roc - A tag from the tests stanza above.>,
            "classes": <OPTIONAL, type must be plot_roc - The comma-separated classes to plot (e.g. 0,1,2,3,8,0+1,2+3).>,
            "iou": <OPTIONAL, type must be plot_pr or all_pr - A float between 0.5 and 1.0.>,
            "output_name": <REQUIRED, type must be summary - The filename for the output file.> 
        }
    ],
    "tests": [
        {
            "tag": <An internal tag that will reference this test.>,
            "dataset_path": <A path to a dataset config file to evaluate.>,
            "classify": <An integer that controls how many of the top predicted classes get recorded.>
        }
    ],
    "timestamp": <An optional ISO time stamp for when the file was created.>,
    "variables": [
        {
            "nickname": <A short string to describe the variable; used in the derived model name>, 
            "config_field": <A string indicating which ModelConfig parameter to change>,
            "vals": [<A list of desired options for this variable.>] | <"RANDOM" (if the config_field is a seed).>
        }
    ]
}
```

Experiment outline files also have a specification file which details the fields in an experiment outline file. For 
more information about any of the fields in an experiment outline, refer to the following specification file:

| Experiment outline path | Specification file |
| --------------- | --------------- |
| 📝experiments/cifar_layer/experiment_outline.json | 📝juneberry/documentation/experiment_outline_specification.md |

## Building the Custom Experiment Outline

Constructing your own Juneberry experiment outline is a matter of organizing your desired models, tests, and reports 
and adhering to the format described by the experiment outline specification file. 

In this example, you will write an experiment outline to replicate the portion of Table 6 in He et al. (2015) that 
examines the depth of the network. The outline will use the `models/cifar_R20` model config you created earlier as 
a baseline config, with a variable that modifies the `model_architecture.args.layers` field to four different values. 

The resulting experiment outline would look something like this:
```json
{
    "baseline_config": "cifar_R20",
    "description": "Layer depth experiment for CIFAR style He et al. (2015) ResNets",
    "format_version": "0.2.0",
    "reports": [
        {
            "description": "Experiment Summary",
            "output_name": "Experiment Summary.md",
            "type": "summary"
        }
    ],
    "tests": [
        {
            "classify": 0,
            "dataset_path": "data_sets/torchvision/cifar10.json",
            "tag": "CIFAR-10 Test"
        }
    ],
    "timestamp": "2021-06-238T10::00",
    "variables": [
        {
            "config_field": "model_architecture.args.layers",
            "nickname": "layers",
            "vals": [ 20, 32, 44, 56 ]
        }
    ]
}
```
Save this file content to `experiments/cifar_layer/experiment_outline.json`.

It is also possible to specify multiple variables for a full factorial experiment, as well as various other 
experimental designs. Please see the examples in `experiments/*/README.md` for more experiment configurations.

## Execute an experiment with Juneberry.

Once you have a valid experiment outline file, you may begin conducting experiment operations in Juneberry. The 
`jb_run_experiment` command is your primary method for interacting with Juneberry experiments. The command supports 
several flags:

- The `--commit` or `-X` flag executes the experiment command outside of preview mode.
- The `--dryrun` or `-D` flag runs the "dryrun" workflow. This flag on its own will provide a preview of the tasks to be 
  executed. The `--commit` or `-X` flag must be included to actually execute this workflow.
- The `--clean` or `-C` flag cleans the files associated with the workflow ("main" is the default). The flag on its own
  will provide a preview of the files to be cleaned. The `--commit` or `-X` flag must be included to actually clean the
  files.
- The `--processes` or `-N` flag specifies the number of GPUs on which to run parallel model trainings. The default
  behavior is to run one training process at a time across all available GPUs.
- The `--regen` or `-R` flag regenerates experiment config files from an experiment outline.

Use the following command to run a preview of the experiment outline file you saved to `experiments/cifar_layer`:
```
jb_run_experiment cifar_layer
```

The output of this command tells you several things. First, the experiment runner uses the `jb_generate_experiments` 
script to convert your experiment outline into a Juneberry experiment. This action produces an experiment config file 
which outlines the training, evaluation, and reporting tasks that will be performed during the experiment. 

Additionally, the experiment generation script automatically creates the model configs for the experiment by combining 
the baseline model config with any variables defined in the experiment outline. Inside the `models` directory, you will 
find a new directory named `cifar_layer`, matching the name of the experiment. This new directory contains four 
sub-directories, one for each of the model configs Juneberry generated for you. When comparing the model configs, 
you'll find the only significant difference between the files is in the model_architecture.args.layers field. Each 
config has a unique entry from the set of values [ 20, 32, 44, 56 ] listed in the experiment outline.

### Running the Experiment:

When you run the experiment outside of preview mode, Juneberry will execute the list of tasks in the `main_dodo.py` 
file. `main_dodo.py` consists of a list of tasks to be executed by the doit task runner in the order determined by the
dependency graph of your experiment.

Broadly speaking, the experiment runner starts by training all the experiment models using `jb_train`. Next, 
the runner uses `jb_evaluate_data` to evaluate each model using the datasets listed in the `tests` section of the 
experiment. Finally, the runner will construct the desired reports. In this case, `jb_summary_report` will produce a 
markdown file summarizing the training and evaluation information produced during the experiment.

You have a few options for running the experiment, depending on the number of GPUs available to you. 

   * Single thread: 
   
   ```
   juneberry$ jb_run_experiment cifar_layer -X
   ```

   * Multiple threads, one GPU per task (e.g. if you wanted to run 2 processes):

   ```
   juneberry$ jb_run_experiment cifar_layer -XN 2
   ```

### Post-Experiment Files:

From the experiment outline file, Juneberry produces `rules.json`, `main_dodo.py`, and `dryrun_dodo.py` files. These
files will appear in the `experiments/cifar_layer` directory, and they are used to execute the experiment. Subsequent 
runs of the experiment will make use of these doit files and will only execute tasks which are not deemed "up to date".

To re-run an experiment in its entirety, you can clean the experiment using the `-XC` flag and run the experiment again 
using one of the above run commands. If changes are made to the experiment outline, you can use the `--regen` flag to 
regenerate these files to include the updated tasking.

Below is a small snippet of the `main_dodo.py` file for the `cifar_layer` experiment:
```python
def task_0():
    return {
        'doc': "jb_train cifar_layer/layers_0",
        'file_dep': [
            "models/cifar_layer/layers_0/config.json",
            "data_sets/torchvision/cifar10.json"
        ],
        'targets': [
            "models/cifar_layer/layers_0/model.pt",
            "models/cifar_layer/layers_0/train/output.json",
            "models/cifar_layer/layers_0/train/output.png",
            "models/cifar_layer/layers_0/train/log.txt",
            "models/cifar_layer/layers_0/train"
        ],
        'actions': [["jb_train", "cifar_layer/layers_0"]],
        'clean': [clean_targets]
    }
```

Each task lists a `targets` field which identifies the files generated during the execution of the task. By the end of 
the experiment, your Juneberry workspace should contain the targets listed throughout the `main_dodo.py` file.

Documentation for doit is available [here](https://pydoit.org/contents.html).

## Compare the results of the Juneberry experiment with the published results.

The table below summarizes evaluation results generated by Juneberry and compares them with the results published in 
He et al. (2015). You can see how close Juneberry came to replicating the results from the paper by comparing the error 
percentages in the table. Remember, the error percentage is calculated by subtracting the evaluation accuracy 
percentage from 100%. 

|   ResNet Layers   |  Juneberry Evaluation Accuracy    |  Juneberry Error (%)    |   Paper Error (%)   |
| ---- | ---- | ---- | ---- |
|   20   |  90.96%  |   9.04   |   8.75   |
|   32   |  91.33%  |   8.67   |   7.51   |
|   44   |  92.13%  |   7.87   |   7.17   |
|   56   |  92.47%  |   7.53   |   6.93   |

As an additional test of your understanding, you could produce dataset and model config files to train a ResNet using 
the [CIFAR-100](https://www.tensorflow.org/datasets/catalog/cifar100 "View Dataset") dataset.

At this point you should have a general understanding of Juneberry and how it can help you to reproduce, verify, and 
validate machine learning models. You can find additional documentation and vignettes inside the 
`juneberry/documentation` directory of the project.

# References 

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv:1512.03385 [cs], Dec. 
2015, Accessed: May 01, 2020. [Online]. Available: http://arxiv.org/abs/1512.03385

[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet 
classification. In ICCV, 2015.

[3] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate 
shift. In ICML, 2015.


# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
