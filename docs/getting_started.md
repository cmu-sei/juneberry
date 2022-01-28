GETTING STARTED
==========

# Juneberry with a Docker Container

## Project Layout

Juneberry requires several directories to contain its contents, such as the source code, the data,
Tensorboard logs, and caches.  To simplify all this, create a single directory for your project, 
which we will refer to as the **project-root**.

Change into your project-root and run the following commands:

```shell script
git clone https://github.com/cmu-sei/juneberry.git
mkdir dataroot
mkdir cache
mkdir cache/hub
mkdir cache/torch
mkdir tensorboard
mkdir workspace
```

At this point, your project-root should contain the following directory structure:

```
project-name/
    cache/
        hub/
        torch/
    dataroot/
    juneberry/
    tensorboard/
    workspace/
```

## Starting the container

In order to use the Juneberry Docker Container, you will need to have a Docker environment installed
on your system. Refer to the following installation steps
[on the Docker website - (external link)](https://docs.docker.com/get-docker/).

The Juneberry Docker containers can be built from scratch (see building_docker.md) or 
downloaded from a repo, such as the following repo on 
[Docker Hub - (external link)](https://hub.docker.com/repository/docker/amellinger/juneberry). To pull 
the CPU-only Juneberry Docker image, execute the following command:

```shell script
docker pull amellinger/juneberry:cpudev
```

For the CUDA-enabled Juneberry Docker image (quite large), you can use the following command:

```shell script
docker pull amellinger/juneberry:cudadev
```

These Docker images themselves do not contain any Juneberry code or data.  When either container
starts, the Juneberry code and data get _mounted_ into the container. This means the directory in 
the host environment is directly accessible from inside the container. Therefore, any edits made to 
content via the host environment are available inside the container and are not working on copies
of those files. One can use any editor in the host environment, and those modifications 
will be available in the container. Similarly, outputs created by the container such as models, 
log files, and plots inside the models directory will be available outside the container will 
persist after the container terminates.

A sample script called `enter_juneberry_container` starts a **temporary** 'cudadev' container 
on the host using all available GPUs. It assumes the project directory structure described above.

Change _into_ the project directory and run the enter_juneberry_container script. By default, the script 
will attempt to use a locally built `juneberry/cudadev:dev` container. If you downloaded
a different container, you can specify which container to use via the second argument to the script.

For example, you can use the following command to start an instance of the downloaded CPU-only container:

TODO: Fix seocnd/third script argument.

```shell script
docker/enter_juneberry_container . amellinger/juneberry:cpudev
```

Users are encouraged to create a copy of the `enter_juneberry_container` script and customize it to 
suit their needs.

## Using the Container

Once inside the container, the user will be inside the `/juneberry` directory by default.
There are a few additional tasks to complete before Juneberry can be used:

* Install Juneberry (from a python package perspective)
* Setup up user id mapping so outputs use proper user/group ids
* Activate shell completion

These steps are required _every_ time the container is started, because these files and scripts
are not available when the image was created, and thus do not persist. The following commands will 
achieve these tasks:

```shell script
pip install -e .
scripts/set_user.sh
source scripts/juneberry_completion.sh
```
### container_start.sh

For convenience, the user may place the above commands (or any other commands) in a bash script named `container_start.sh` placed in the user's initial directory (`/juneberry` by default, or in a custom workspace, see below). This script will be executed when the container is started. A sample `container_start.sh` is provided in the `juneberry/docker` directory.

## Using a custom workspace

When the container is started, the user will be inside the `/juneberry` directory by default. However, the user can choose another directory called a "custom workspace" to be in when the container starts.

A custom workspace is a directory that contains the following files and subdirectories:

* A `data_sets` directory
* An `experiments` directory
* A `models` directory
* A `src` directory
* A `juneberry.ini` file
* An optional `container_start.sh` file

To start the Juneberry container in a custom workspace, use the following command:

```shell script
juneberry/docker/enter_juneberry_container <project_dir> [workspace_dir]
```

If `workspace_dir` does not exist, or is missing any of the above subdirectories or files, they will be created on container startup.

The user's custom workspace will be mounted under `/workspace` in the container, and the user will start inside of this directory.

# Juneberry using a virtual environment

Juneberry can be installed in a virtual environment. However, a specific order of operations and a set of versions
is required in order to get all the included platforms to install properly. 

## Project Layout

Assuming the same project-root as before, the difference that a manual installation will introduce will be 
another directory inside the project-root. Perform this installation **INSIDE** the project-root.

## Set up a virtual environment

Any virtual environment,such as venv or pyenv, can be used. The first step is to construct and enter 
the virtual environment. The following example demonstrates pyenv/venv. 

```shell script
pyenv virtualenv 3.8.5 jb
pyenv activate jb
python3 -m pip3 install --upgrade pip
```

## Clone MMDetection and Juneberry

```shell script
# Get a copy of mmdetection
git clone --depth 1 --branch v2.18.0 https://github.com/open-mmlab/mmdetection.git /mmdetection

# Get a copy of juneberry
git clone https://github.com/cmu-sei/juneberry.git
```

## Install from Requirements

By default, pip does NOT honor the ordering. So the requirements are provided one at a time.

```shell script
cd juneberry
cat requirements.txt | xargs pip install
```

## Install MMDetection

```shell script
# Install the base MMCV package.  The FORCE_CUDA isn't always needed, but doesn't hurt. 
# This may take a while to compile everything.
MMCV_WITH_OPS=1 FORCE_CUDA="1" pip3 install mmcv-full==1.3.17
  
# For OSX or CPU only
# pip3 install mmcv-full==1.3.17
  
# Install the actual MMDetection part.
pip3 install -r ../mmdetection/requirements/build.txt
pip3 install -v -e ../mmdetection/.
   
# Verify the installation was successful.
python mmdet/utils/collect_env.py
```

## Install Detectron2

```shell script
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
   
# For OSX
# CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
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
tensorboard_root = /path-to-project-root/juneberry
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





