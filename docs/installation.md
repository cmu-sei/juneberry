Installation
==========

# Juneberry with a Docker Container

Using a prebuilt Docker container is the recommended method for accessing Juneberry.

# Direct Installation

If you wish to just install Juneberry without any particular platform, clone Juneberry from GitHub and
install using `pip install .` from within the repo. By default, Juneberry doesn't install any platforms
because it makes no assumptions about which platforms the user wants. The available platforms are:

* tf - TensorFlow
* torch - PyTorch
* onnx - ONNX support for CPU.
* onnx-gpu - ONNX support for GPUs.
* opacus - Support for Opacus.

# Juneberry with a Virtual Environment

**Apple Silicon Note:** Currently, there is no configuration for Apple Silicon.

Juneberry can be installed in a virtual environment. However, if you want to access all the available 
platforms, then a specific order of operations and a set of platform versions is required in order to get 
all the included platforms to install properly.

## Set up a virtual environment

Any virtual environment,such as venv or pyenv, can be used. The first step is to construct and enter 
the virtual environment. The following example demonstrates pyenv. Instructions for how to 
install pyenv-virtualenv can be found [here - external link](https://github.com/pyenv/pyenv-virtualenv).

```shell script
pyenv install 3.8.5
pyenv virtualenv 3.8.5 jb
pyenv activate jb
python3 -m pip install --upgrade pip
```

## Clone MMDetection and Juneberry

The following commands can be used to clone MMDetection into the project directory as a sibling to the 
Juneberry repository directory.

```shell script
# Get a copy of mmdetection
git clone --depth 1 --branch v2.18.0 https://github.com/open-mmlab/mmdetection.git ./mmdetection
```

## Install from Requirements

There are two versions of the requirements.txt file for Juneberry. The file named "requirements.txt" 
contains the packages and version numbers for a Juneberry installation on a system with an available 
GPU. The version named "requirements-cpu.txt" contains the CPU-only version of those packages.

By default, pip does NOT honor the ordering of packages in the requirements file, so the requirements 
are provided one at a time.

Use the following commands to install the packages in requirements.txt.

**NOTE:** If you are performing a CPU-only installation of Juneberry, you replace "requirements.txt" with 
"requirements-cpu.txt" in the second command.

```shell script
cd juneberry
cat requirements.txt | xargs pip install --no-cache-dir
```

## Install MMDetection

The initial MMDetection installation step varies depending on whether a GPU is available 
for use on your system. In this section, perform one of the "Step 1" commands, followed 
by the "Step 2" code block.

1. For OSX or CPU-only systems:
```shell script
pip3 install mmcv-full==1.3.17
```

#### _OR_

1. For systems with GPU support:
```shell script
# Install the base MMCV package.  The FORCE_CUDA isn't always needed, but doesn't hurt. 
# This may take a while to compile everything.
MMCV_WITH_OPS=1 FORCE_CUDA="1" pip3 install mmcv-full==1.3.17
 ```

2. After performing one of the previous blocks, execute the following commands:

```shell script
# Install the actual MMDetection part.
pip3 install -r ../mmdetection/requirements/build.txt
pip3 install -v -e ../mmdetection/.
   
# Verify the installation was successful.
python ../mmdetection/mmdet/utils/collect_env.py
```

## Install Detectron2

To install Detectron2 on OSX, use the first code block. All other systems should follow the second 
code block.

### Detectron2 OSX Installation
```shell script
# For OSX
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
```

#### _OR_

### Detectron2 Typical Installation
```shell script
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
```
   
## Install Juneberry

After the previous packages have been installed successfully, you can install Juneberry.

```shell script
# From the juneberry directory.
pip install -e .
```

## Bash Completion

At this point, Juneberry should be fully operational. As an added convenience, there is a 
completion support script which provides tab completion for dataset and model names. 

To activate this enhancement, use the following command to `source` the completion script:

```shell script
source scripts/juneberry_completion.sh
```
