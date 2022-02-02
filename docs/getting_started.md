GETTING STARTED
==========

This document describes the actions to take in order to get started with using Juneberry. There are two 
courses of actions described here. The recommended approach, described first, involves making use of a 
Juneberry Docker container. The second approach describes how to install Juneberry in a virtual Python 
environment. The latter approach has not been tested on a variety of platforms, and therefore the Docker 
container method is preferred.

# Juneberry with a Docker Container

## Project Layout

Juneberry requires several directories to store its components, such as the source code, the data,
Tensorboard logs, and caches.  To simplify all this, create a single directory for your project. 
This directory will be referred to as the **project-root**.

Inside the project-root, your goal is to create the sub-directories for the various components. Use 
the following commands to obtain a copy of the Juneberry source code, and create the various 
sub-directories the Juneberry Docker container will look for:

```shell script
git clone https://github.com/cmu-sei/juneberry.git
mkdir cache
mkdir cache/hub
mkdir cache/torch
mkdir dataroot
mkdir tensorboard
mkdir workspace
```

At this point, your project-root should resemble the following directory structure:

```
project-root/
    cache/
        hub/
        torch/
    dataroot/
    juneberry/
    tensorboard/
    workspace/
```

## Starting the Container

**Apple Silicon Note:** Currently, there is no container for Apple Silicon.

In order to use the Juneberry Docker Container, you will need to have a Docker environment installed
on your system. Refer to the following installation steps
[on the Docker website - (external link)](https://docs.docker.com/get-docker/).

The Juneberry Docker containers can be built from scratch as described in 
[Building Juneberry Docker Containers](building_docker.md), but they can also be downloaded from a repo, such as 
the following repo on [Docker Hub - (external link)](https://hub.docker.com/repository/docker/amellinger/juneberry). 
Assuming you have configured your Docker instance to work with this repo, you could execute the following command 
to pull the CPU-only Juneberry Docker image:

```shell script
docker pull amellinger/juneberry:cpudev
```

Alternatively, the following command would pull the CUDA-enabled Juneberry Docker image, which is much larger:

```shell script
docker pull amellinger/juneberry:cudadev
```

The Docker images themselves do not contain any Juneberry source code or data.  The code and data get
_mounted_ into the container whenever either container starts. This means the directories containing the 
source code and data in the host environment are directly accessible from inside the container. Therefore, 
any changes made to these files in the host environment will also be available inside the container, since 
the container is not working on copies of those files. Any editor available in the host environment can be 
used to change those files, and those modifications will be available inside the container. Similarly, outputs 
created by Juneberry inside the container such as models, log files, and plots will be available 
outside the container after the container terminates, due to this relationship with the host filesystem.

A sample script called `enter_juneberry_container` starts (by default) a **temporary** 'cudadev' container 
on the host using all available GPUs. It assumes the project directory structure described above.

From inside your project-root directory, run the enter_juneberry_container script.  If you downloaded
a container other than the 'cudadev' version, you can specify which container to use via the second argument 
to the script.

For example, the following command will start an instance of the downloaded CPU-only container 
from within the _project-root_ directory:

```shell script
juneberry/docker/enter_juneberry_container -c amellinger/juneberry:cpudev `pwd`
```

**NOTE:** Docker does not like the shortcuts provided by the `.` or `~` symbols and requires the use of expanded paths.

There are a variety of other configurations available in the script; you can examine the contents for more details. 
Users are encouraged to create a copy of the `enter_juneberry_container` script and customize it to 
suit their needs.

## Using the Container

Once inside the container, the user will find themselves in the `/juneberry` directory by default.
There are a few more initialization tasks to complete before Juneberry is operational:

* Install Juneberry (from a python package perspective)
* Set up user id mapping so outputs use proper user/group ids
* Activate shell completion

The following commands will achieve these tasks:

```shell script
pip install -e .
scripts/set_user.sh
source scripts/juneberry_completion.sh
```

**NOTE:** These steps are required _every_ time the container is initialized because the files and scripts
involved come from Juneberry source code and thus do not persist inside the container.

### container_start.sh

For convenience, users can create a bash script containing the previous commands and name the file 
`container_start.sh`. If a script with that name is found inside the juneberry directory of the project-root, 
or inside a custom workspace (see below), it will be executed during the container's initialization. The 
`juneberry/docker` directory contains a sample `container_start.sh` script.

## Using a Custom Workspace

When a user first enters a Juneberry container, they should find themselves inside the `/juneberry` directory 
by default. However, the user can also choose another directory, known as a "custom workspace", to be the 
default directory.

A custom workspace can be any directory, but it MUST contain the following files and subdirectories:

* A `data_sets` directory
* An `experiments` directory
* A `models` directory
* A `src` directory
* A `juneberry.ini` file
* An optional `container_start.sh` file

To start the Juneberry container in a custom workspace, use the '-w' switch and provide the path to the 
custom workspace:

```shell script
/juneberry/docker/enter_juneberry_container -w [workspace_dir] <project_dir>
```

If the `workspace_dir` does not exist on the host filesystem, or if the workspace is missing any of the 
above subdirectories or files, the missing content will be created during container startup.

Once the container is initialized, the custom workspace will be mounted under `/workspace` in the container, 
and the user will be placed inside this directory.

# Juneberry with a Virtual Environment

**Apple Silicon Note:** Currently, there is no configuration for Apple Silicon.

Juneberry can be installed in a virtual environment. However, a specific order of operations and a set of 
platform versions is required in order to get all the included platforms to install properly. 

## Project Layout

Assuming the same project-root structure as before, the manual installation will introduce another 
directory inside the project-root. Perform these installation steps **INSIDE** the project-root.

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

The following commands can be used to clone MMDetection and Juneberry into your current directory.

```shell script
# Get a copy of mmdetection
git clone --depth 1 --branch v2.18.0 https://github.com/open-mmlab/mmdetection.git ./mmdetection

# Get a copy of juneberry
git clone https://github.com/cmu-sei/juneberry.git
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

The initial MMDetection installation step vary depending on whether a GPU is available 
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

## Add a juneberry.ini

The overview contains a deeper discussion of the Juneberry ini.  For now, create a file called 
`juneberry.ini` _inside_ your newly cloned juneberry directory and add the following contents
to the file:

```shell script
[DEFAULT]
workspace_root = /path-to-project-root/juneberry
data_root = /path-to-project-root/dataroot
tensorboard_root = /path-to-project-root/tensorboard
```

## Bash Completion

At this point, Juneberry should be fully operational.  As an added convenience, there is a 
completion support script which provides tab completion for dataset and model names. 

To activate this enhancement, use the following command to `source` the completion script:

```shell script
source scripts/juneberry_completion.sh
```





