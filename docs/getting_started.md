Getting Started
==========

This document describes how to get started with using Juneberry. The recommended installation method, 
relies on a Juneberry Docker container. Juneberry can also be used from a virtual environment.

The second approach describes how to install Juneberry in a virtual Python 
environment. The latter approach has not been tested on a variety of platforms, so the Docker containers are 
the preferred choice. Both installation methods are tested using the same steps. 

# Preparation Steps

## Lab Layout

Juneberry requires several directories to store its components, such as the source code, data,
Tensorboard logs, and caches.  
These purposes of these directories is described in the [Workspace and Experiment Overview](overview.md).
While this structure isn't required in the long term, for the purposes of getting started,
we'll go ahead and use it as is.
Let's start by creating a single directory for your project which we'll refer to as the **lab-root**.

Inside the lab-root, your goal is to create sub-directories for the various Juneberry components.
You can create these sub-directories manually, or use the `setup_lab.py` script located in the 
`scripts` directory to automatically create the required directories.

### Using setup_lab.py

Inside your lab root, clone the Juneberry repository from GitHub, then use the `setup_lab.py` script 
from inside the newly cloned repository to create the remaining sub-directories in the project root.

```shell script
git clone https://github.com/cmu-sei/juneberry.git
juneberry/scripts/setup_lab.py . 
```

### Manually Creating the Project Structure

If you need to manually create the various sub-directories for the Juneberry Docker, you can 
use the following commands to obtain the Juneberry source code and establish the expected directories:

```shell script
git clone https://github.com/cmu-sei/juneberry.git
mkdir cache cache/hub cache/torch
mkdir dataroot
mkdir tensorboard
```

NOTE: The cache directories are only needed for containers (recommended). If you use a virtual environment
(not recommended) then you don't need the cache directory.

### The Project Structure

At this point, your lab root should resemble the following directory structure:

```
lab-root/
    cache/
        hub/
        torch/
    dataroot/
    juneberry/
    tensorboard/
```
# Juneberry with a Docker Container

This section describes how run Juneberry using a Docker container.

## Acquiring the Container

**Apple Silicon Note:** Currently, there is no container for Apple Silicon.

Prebuilt Juneberry containers are provided on [Docker Hub - (external link)](https://hub.docker.com/r/cmusei/juneberry). 

In order to use the Juneberry Docker Container, you will need to have a Docker environment installed
on your system. Refer to the following installation steps
[on the Docker website - (external link)](https://docs.docker.com/get-docker/).

Assuming you have configured your Docker instance to work with this repo, you could execute the following command 
to pull the CPU-only Juneberry Docker image:

```shell script
docker pull cmusei/juneberry:cpudev
```

Alternatively, the following command would pull the CUDA-enabled Juneberry Docker image, which is much larger:

```shell script
docker pull cmusei/juneberry:cudadev
```

### Building a container

If you aren't able to acquire a Juneberry container it can be built from scratch as described in 
[Building Juneberry Docker Containers](building_docker.md). 

## Starting the Container

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
juneberry/docker/enter_juneberry_container -c cmusei/juneberry:cpudev `pwd`
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

## Bash Completion

As an added convenience, there is a completion support script which provides tab completion for dataset and model names. 

To activate this enhancement, use the following command to `source` the completion script:

```shell script
source scripts/juneberry_completion.sh
```

NOTE: If you use a temporary container (by default `enter_juneberry_container` creates temporary containers)
you'll need to source the juneberry completion script every time you start the container.

## Testing Your Installation

After installing Juneberry, you can run the following training command to test your installation:
```shell
jb_train -w . tabular_multiclass_sample
```

This command will quickly train a small, basic tabular dataset. The test is successful if the final epoch 
reports a training accuracy of 97.58%.

## container_start.sh

For convenience, users can create a bash script containing the previous commands and name the file 
`container_start.sh`. If a script with that name is found inside the juneberry directory of the project-root, 
or inside a custom workspace (see below), it will be executed during the container's initialization. The 
`juneberry/docker` directory contains a sample `container_start.sh` script.

# Custom Workspaces

Juneberry uses "workspaces" to house the user model configurations, experiments, and all outputs such as 
trained models, log files, charts and reports. The structure is described in the 
[Workspace and Experiment Overview](overview.md). The Juneberry repository comes with some sample models
for testing the installation which will be sufficient for now.

When a user first enters a Juneberry container, they should find themselves inside the `/juneberry` directory 
by default. However, the user can also choose another directory, known as a "custom workspace", to be the 
default directory.

To start the Juneberry container in a custom workspace, use the '-w' switch and provide the path to the 
custom workspace:

```shell script
/juneberry/docker/enter_juneberry_container -w [workspace_dir] <project_dir>
```

If the `workspace_dir` does not exist on the host filesystem, or if the workspace is missing any of the 
above subdirectories or files, the missing content will be created during container startup.

Once the container is initialized, the custom workspace will be mounted under `/workspace` in the container, 
and the user will be placed inside this directory.

## Creating a workspace

TBD

# Configuring Juneberry paths

**TODO:** Not really sure this goes here

By default, when Juneberry tools are executed (e.g. jb_train), the **current working directory** will be used for the 
workspace. The default locations for the data_root and the tensorboard directories that Juneberry
will use are peers to the workspace directory.  Thus, if one has multiple workspaces then executing
commands from that workspace directory will result in using that workspace.

The workspace can be specified to a Juneberry tool using the `-w` switch or the by setting the
`JUNEBERRY_WORKSPACE` environment variable.  For example, if one was in the project root, and they
had a workspace called `test-workspace` and wanted to to train a model `test-model` contained
in the `test-workspace` models directory, then these commands would work similarly.

```shell script
jb_train -w test-workspace test-model
```

```shell script
JUNEBERRY_WORKSPACE=test-workspace jb_train test-model
```

## Specifying the structure

If the default structure won't work (e.g. the dataroot or tensorboard are stored elsewhere) then they can
be specified directly using command line switches or environment variables.  Command line switches take precedence
over environment variables which take precedence over default locations. The following table summarizes
the switches, environment variables, and default values.

| Configuraton | switch | environment variable | default value |
| --- | --- | --- | --- |
| workspace | -w | JUNEBERRY_WORKSPACE | cwd() |
| data_root | -d | JUNEBERRY_DATAROOT | -workspace-/../dataroot |
| tensorboard | -t | JUNEBERRY_TENSORBOARD | -workspace/../tensorboard |

It is important to note that the default locations for dataroot and tensorboard are workspace relative,
not cwd() relative. Meaning, if the workspace is specific and the dataroot or tensorboard are not, 
then they default to peers of the specified workspace.  Of course, if the data root or tensorboard directories
are specified in any way (switch or environment variable), those values are used directly.




