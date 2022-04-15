Getting Started
==========

This document describes how to get started using Juneberry. The recommended installation method
relies on a Juneberry Docker container. Juneberry can also be used from a virtual environment if
necessary. Be aware that installation and management of all the supporting CUDA enabled packages
can be a challenging task if you do it on your own.

# The Basics

Fundamentally, Juneberry is just another Python package that can be cloned from GitHub and installed 
into your environment using a normal Python installation process: 

```shell script
git clone https://github.com/cmu-sei/juneberry.git
pip install juneberry
```

**IMPORTANT**: Juneberry can require lots of infrastructure to actually run models, depending on which machine 
learning platforms are involved. So in the vast majority of cases, a lot of other packages such as 
tensorflow, pytorch, detectron2, mmdetection, or onnx may also need to be installed. While these additional 
packages can be installed using pip "extras" (e.g. `pip install juneberry[tf]` - see `setup.py` for details), 
the dependencies are complex and determining the correct installation order can be challenging. The rest of this 
document explains how pre-configured Docker containers can be used as experiment environments. An alternative method, 
describing how to construct a virtual environment for Juneberry, will also be presented as an alternative to the 
Docker containers.

# Preparation Steps

## Lab Layout

Juneberry requires several directories to store its components, such as source code, data,
Tensorboard logs, and caches. While the structure can be customized, the default structure should be sufficient 
for getting started. The first step is to create a single directory for the project.
This directory is referred to as the **lab-root**.

Inside the lab-root, the goal is to create sub-directories for the various Juneberry components.
These sub-directories can be created manually, or via the `setup_lab.py` script located in the 
`scripts` directory.

While the `juneberry-example-workspace` isn't required for a Juneberry installation, a workspace directory 
of some sort **is** required by Juneberry. Inside `juneberry-example-workspace`, you'll find 
sample models, datasets, and experiments which can be used to test your Juneberry installation. The rest of 
this document assumes the example workspace has been installed.

The lab structure is described in more detail in the [Workspace and Experiment Overview](overview.md). The details 
in that document aren't required in order to just get started with Juneberry, but it would be a good 
idea to review the information in that file at some point.

### Using setup_lab.py

Inside the lab-root, clone the "Juneberry" and "Juneberry Example Workspace" repositories from GitHub, 
and then use the `setup_lab.py` script from inside the newly cloned Juneberry repository to create 
the remaining sub-directories in the lab-root.

```shell script
git clone https://github.com/cmu-sei/juneberry.git
git clone https://github.com/cmu-sei/juneberry-example-workspace.git
juneberry/scripts/setup_lab.py . 
```

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
['The Lab Structure'](getting_started.md#The-Lab-Structure), and offers many strategies for specifying the  
workspace. 
1) Assuming the workspace is within the root of the lab, you may run the script from inside the workspace directory.
In this situation the script chooses the current directory as the workspace directory and 
the parent of the workspace directory as the lab-root.
2) You may run the script from the lab-root and provide one argument. The argument provided to the script will 
be interpreted as your desired workspace directory. As a result of how Docker interprets that directory, we advise 
providing the full path to the desired workspace directory.
3) Set the workspace using the `JUNEBERRY_WORKSPACE` environment variable to override the other values.
4) Copy and modify the `enter_juneberry_container` as desired to accommodate different workspaces.

For example, the following command demonstrates option 1, which starts an instance of the downloaded 
CPU-only container from inside the _workspace_ (i.e. juneberry-example-workspace) directory:

```shell script
../juneberry/docker/enter_juneberry_container
```

The `enter_juneberry_container` script can be controlled via a wide variety of environment variables
or can be copied elsewhere and modified.  
See the contents of [enter_juneberry_container](../docker/enter_juneberry_container) for details.

This example demonstrates how to start the CUDA container with "all" GPUs using environment variables from 
inside the desired workspace directory:

```shell script
JUNEBERRY_CONTAINER=cmusei/juneberry:cudadev JUNEBERRY_GPUS=all ../juneberry/docker/enter_juneberry_container
```

## Using the Container

Once inside the container, the user will find themselves in their workspace directory by default. 
There are a few more initialization tasks to complete before Juneberry is operational:

* Install Juneberry (from a python package perspective)
* Optional: Set up user ID mapping so outputs use proper user/group IDs.
* Optional: Activate bash shell completion (if desired)
* Optional: Install any packages from the workspace, assuming it has a setup.py.

The following commands will achieve these tasks:

```shell script
pip install -e /juneberry
/juneberry/scripts/set_user.sh
source /juneberry/scripts/juneberry_completion.sh
pip install -e .
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

After installing Juneberry and launching the container, you should find yourself inside the 
`/juneberry-example-workspace` directory. From there, you can run the following training command to 
test your installation:
```shell
jb_train tabular_multiclass_sample
```

This command will quickly train a small, basic tabular dataset. The test is successful if the final epoch 
reports a training accuracy of 97.58%.

## container_start.sh

For convenience, users can create a bash script containing the startup commands and name the file 
`container_start.sh`. When a script with that name is found inside the workspace directory, 
it will be executed (technically it is `sourced`) during the container's initialization. The 
`juneberry/docker` directory contains a sample `container_start.sh` script.

# Creating Workspaces

As described above, Juneberry uses "workspaces" to house the user's model configurations, experiments, 
and all outputs such as trained models, log files, charts, and reports. The `juneberry-example-workspace` 
repository includes some sample models for testing the installation.

The structure of a workspace is detailed in the [Workspace and Experiment Overview](overview.md), but again 
these details aren't necessary to just get started with using Juneberry.

The script `setup_workspace.py' from the 'scripts' directory can be used to properly initialize a new workspace 
for you. The following command creates the workspace "my-workspace" in the lab directory on the host system:

```shell script
juneberry/scripts/setup_workspace my-workspace
```

Optionally, a simple `container_start.sh` script is provided to simplify container startup. If desired,
copy it into the new workspace.

```shell script
cp junebery/docker/container_start.sh myworkspace/.
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
