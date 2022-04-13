Getting Started
==========

This document describes how to get started using Juneberry. The recommended installation method
relies on a Juneberry Docker container. Juneberry can also be used from a virtual environment if
necessary. Be aware that installation and management of all the supporting CUDA enabled packages
can be a challenging task.

# The Basics

Juneberry is a Python package that can be cloned from GitHub and installed using a normal
installation process in a properly configured environment. 

```shell script
git clone https://github.com/cmu-sei/juneberry.git
git clone https://github.com/cmu-sei/juneberry-example-workspace.git
pip install juneberry
```

**IMPORTANT**: Juneberry requires a lot of infrastructure (depending on platform) to actually run models,
so in the vast majority of cases, a lot of other packages such as tensorflow, pytorch, detectron2, mmdetection 
or onnx may need to be installed. While these can be installed with extras 
(e.g. `pip install juneberry[tf]` - see `setup.py` for details), the dependencies are complex 
and the ordering is challenging. The rest of this document explains how pre-configured Docker containers 
can be used as experiment environments. An alternative method, describing how to construct a virtual environment 
for Juneberry, will also be presented as an alternative to the Docker containers.

# Preparation Steps

## Lab Layout

Juneberry requires several directories to store its components, such as source code, data,
Tensorboard logs, and caches. The purpose of these directories is described in the 
[Workspace and Experiment Overview](overview.md). 

While `juneberry-example-workspace` isn't required for your Juneberry installation, it serves as an example 
workspace and contains useful baseline models which can be used to test your installation.

Juneberry has a default structure that it knows how to work with. The structure can be customized, but for
introductory purposes the default structure should be sufficient. To start, a single directory for the 
project should be created. This directory is referred to as the **lab-root**.

Inside the lab-root, the goal is to create sub-directories for the various Juneberry components.
These sub-directories can be created manually, or via the `setup_lab.py` script located in the 
`scripts` directory.

### Using setup_lab.py

Inside the lab root, clone the Juneberry repository from GitHub, then use the `setup_lab.py` script 
from inside the newly cloned repository to create the remaining sub-directories in the lab-root.

```shell script
git clone https://github.com/cmu-sei/juneberry.git
git clone https://github.com/cmu-sei/juneberry-example-workspace.git
juneberry/scripts/setup_lab.py . 
```

### Manually Creating the Lab Structure

The following commands can be used to create the structure manually:

```shell script
git clone https://github.com/cmu-sei/juneberry.git
git clone https://github.com/cmu-sei/juneberry-example-workspace.git
mkdir cache
mkdir dataroot
mkdir tensorboard
```

NOTE: The cache directories are only needed for containers (recommended). Virtual environments
do not need the cache directory.

### The Lab Structure

At this point, the lab-root should resemble the following directory structure:

```
lab-root/
    cache/
    dataroot/
    juneberry/
    juneberry-example-workspace/
    tensorboard/
```
# Juneberry with a Docker Container

This section describes how to run Juneberry using a Docker container.

## Acquiring the Container

**Apple Silicon Note:** Currently, there is no container for Apple Silicon.

Prebuilt Juneberry containers are provided on [Docker Hub - (external link)](https://hub.docker.com/r/cmusei/juneberry). 

In order to use the Juneberry Docker Container, a Docker environment must be installed
on the system. Refer to the following installation steps
[on the Docker website - (external link)](https://docs.docker.com/get-docker/).

The following command will download the basic Juneberry Docker images from the Docker Hub:

```shell script
docker pull cmusei/juneberry:cpudev
```

Alternatively, the following command would pull the CUDA-enabled Juneberry Docker image, which is much larger:

```shell script
docker pull cmusei/juneberry:cudadev
```

### Building a container

If a Juneberry container can't be acquired (or needs to be modified), it can be built from scratch as described in 
[Building Juneberry Docker Containers](building_docker.md).

## Starting the Container

The Docker images themselves do not contain any Juneberry source code or data.  The code and data get
_mounted_ into the container when the container starts. This means the directories containing the
source code and data in the host environment are directly accessible from inside the container. Therefore, 
any changes made to these files in the host environment will also be available inside the container, since 
the container is not working on copies of those files. Any editor available in the host environment can be 
used to change those files, and those modifications will be available inside the container. Similarly, outputs 
created by Juneberry inside the container such as models, log files, and plots will be available 
outside the container after the container terminates, due to this relationship with the host filesystem.

A sample script called `enter_juneberry_container` can be used to start a **temporary** container. By default, 
this script starts a 'cpudev' container. The script assumes the lab directory structure described above in 
['The Lab Structure'](getting_started.md#The-Lab-Structure), and offers two strategies for accommodating multiple 
workspaces. 
1) You may run the script from inside the workspace directory. In this situation the script chooses 
the current directory as the workspace directory, and the parent of the workspace directory as the lab root.
current workspace directory, since you can have multiple workspaces in your environment.
2) You may run the script from lab root, and provide one argument. The argument provided to the script will 
be interpreted as your desired workspace directory. As a result of how Docker interprets that directory, we advise 
providing the absolute path to the desired workspace directory.

For example, the following command will start an instance of the downloaded CPU-only container 
from within the _workspace_ (i.e. juneberry-example-workspace) directory:

```shell script
juneberry/docker/enter_juneberry_container
```

The `enter_juneberry_container` script can be controlled via a wide variety of environment variables
or can be copied elsewhere and modified.  
See the contents of [enter_juneberry_container](../docker/enter_juneberry_container) for details.

For example, to start the CUDA container with "all" GPUs using environment variables:

```shell script
JUNEBERRY_CONTAINER=cmusei/juneberry:cudadev JUNEBERRY_GPUS=all juneberry/docker/enter_juneberry_container
```

## Using the Container

Once inside the container, the user will find themselves in the `/workspace` directory by default. 
The directory may be called "workspace" inside the container, but it maps to whichever workspace directory 
was provided to the `enter_juneberry_container` script either by command line argument or via
the `JUNEBERRY_WORKSPACE` override.

There are a few more initialization tasks to complete before Juneberry is operational:

* Install Juneberry (from a python package perspective)
* Optional: Set up user ID mapping so outputs use proper user/group IDs.
* Optional: Activate bash shell completion (if desired)

The following commands will achieve these tasks:

```shell script
pip install -e /juneberry
/juneberry/scripts/set_user.sh
source /juneberry/scripts/juneberry_completion.sh
```

**NOTE:** These steps are required _every_ time the container is initialized because the files and scripts
involved come from Juneberry source code and thus do not persist inside the container.

## Bash Completion

As an added convenience, there is a completion support script which provides tab completion for dataset and model names. 

To activate this enhancement, use the following command to `source` the completion script:

```shell script
source /juneberry/scripts/juneberry_completion.sh
```

NOTE: If you use a temporary container (by default `enter_juneberry_container` creates temporary containers)
you'll need to source the juneberry completion script every time you start the container.

## Testing Your Installation

After installing Juneberry, you can run the following training command to test your installation:
```shell
jb_train -w /juneberry tabular_multiclass_sample
```

This command will quickly train a small, basic tabular dataset. The test is successful if the final epoch 
reports a training accuracy of 97.58%.

## container_start.sh

For convenience, users can create a bash script containing the previous commands and name the file 
`container_start.sh`. When a script with that name is found inside the juneberry directory of the lab-root, 
or inside a custom workspace (see below), it will be executed during the container's initialization. The 
`juneberry/docker` directory contains a sample `container_start.sh` script.

# Custom Workspaces

Juneberry uses "workspaces" to house the user's model configurations, experiments, and all outputs, such as  
trained models, log files, charts, and reports. The structure is described in the 
[Workspace and Experiment Overview](overview.md). The Juneberry repository includes some sample models
for testing the installation.

When a user first enters a Juneberry container, they should find themselves inside the `/juneberry` directory 
by default. However, the user can also choose another directory, known as a "custom workspace", to be the 
default directory.

To start the Juneberry container in a custom workspace use the `JUNEBERRY_WORKSPACE` environment variable
or modify a copy of the `enter_juneberry_container` script.

Workspaces require a particular layout as described in [Workspace and Experiment Overview](overview.md).
The script `setup_workspace.py' from the 'scripts' directory can be used to initialize a new workspace.
The following command creates a workspace "my-workspace" in the lab directory on the host system:

```shell script
juneberry/scripts/setup_workspace my-workspace
```

Optionally, a simple `container_start.sh` script is provided to simplify container startup. If desired,
copy it into the new workspace.

```shell script
cp junebery/docker/container_start.sh myworkspace/.
```

Once the custom workspace is created, either use the `JUNEBERRY_WORKSPACE` environment variable or
modify a copy of the `enter_juneberry_container` script.  The following command uses a custom
workspace:

```shell script
JUNEBERRY_WORKSPACE=my-workspace juneberry/docker/enter_juneberry_container
```

# Configuring Juneberry paths

**NOTE:** This section applies when multiple workspaces are mounted into a container (using a custom
`enter_juneberry_container` script), or to non-container virtual environments. 
Within the container the environment variables have already been set to point to the in-container locations.

By default, when Juneberry tools are executed (e.g. jb_train), the **current working directory** will 
be used for the workspace. The default locations for the dataroot and the tensorboard directories that 
Juneberry will use are peers to the workspace directory.  Thus, if one has multiple workspaces within the lab
then executing commands from within that workspace directory will result in using that workspace.
Multiple workspaces aren't exposed in a container without adding extra workspace mounts.

The workspace can be specified to a Juneberry tool using the `-w` switch, or by setting the
`JUNEBERRY_WORKSPACE` environment variable.  For example, if one was in the lab-root, with a workspace 
called `test-workspace`, if you wanted to train a model named `test-model` located
in the `test-workspace` models directory, then the following commands would be equivalent and would 
initiate training for the desired model.

```shell script
jb_train -w test-workspace test-model
```

```shell script
JUNEBERRY_WORKSPACE=test-workspace jb_train test-model
```

## Specifying the structure manually

If the default structure doesn't work (e.g. the dataroot or tensorboard are stored elsewhere), then deviations can
be specified directly using command line switches or environment variables.  Command line switches take precedence
over environment variables, and environment variables take precedence over default locations. The following table 
summarizes the switches, environment variables, and default values.

| Configuration | switch | environment variable | default value |
| --- | --- | --- | --- |
| workspace | -w | JUNEBERRY_WORKSPACE | cwd() |
| data_root | -d | JUNEBERRY_DATAROOT | -workspace-/../dataroot |
| tensorboard | -t | JUNEBERRY_TENSORBOARD | -workspace/../tensorboard |

It is important to note that the default locations for dataroot and tensorboard are workspace relative,
not cwd() relative. Meaning, if the workspace is specified and the dataroot or tensorboard are not, 
then Juneberry assumes the dataroot and tensorboard directories are peer directories of the specified workspace.  
Of course, if the data root or tensorboard directories are specified in any way (switch or environment variable), 
those values are used instead.
