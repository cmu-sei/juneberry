GETTING STARTED
==========

## Project Layout

Juneberry needs a few different directories in which to contain all the parts. The source code, the data,
Tensorboard logs, and caches.  To simplify all this create a directory for your project which we will 
call the **project-root**.

Change into this subdirectory and run the following.

```shell script
git clone https://github.com/cmu-sei/juneberry.git
mkdir dataroot
mkdir cache
mkdir tensorboard
```

There should now be a project directory structure such as:

```
project-name/
    cache/
    dataroot/
    juneberry/
    tensorboard/
```

# Using the Docker Container

First, you'll need to have a docker environment installed.
See the following the installation steps
[on the Docker website - (external link)](https://docs.docker.com/get-docker/).

The Juneberry docker containers can be built from scratch (see building_docker.md) or from using a 
downloaded container such as: 
[Docker Hub - (extrnal link)](https://hub.docker.com/repository/docker/amellinger/juneberry). To pull 
the CPU-only docker image execute:

```shell script
docker pull amellinger/juneberry:cpudev
```

For the CUDA-enabled image (quite large)

```shell script
docker pull amellinger/juneberry:cudadev
```

These docker images do not contain any of the Juneberry code or data.  When the container
is started, the juneberry code, data and code is _mounted_ into the container. This means that
the directory in the host environment is directly accessible from withing the container. Therefore,
all edits made to content via the host environment are available inside the container. This means 
that one can use any editor in the host environment and have those modifications available in the container.
Also, any outputs created by the container such as models, log files, and plots which are in the models 
directory are available outside the container and are therefore peristed when the container terminates.

There is a sample script called `enter_juneberry_container` tha starts up a **temporary** 'cudadev' container 
on the host using all available gpus. It assumes the project directory structure described above.

Change _into_ the project directory and run the enter_juneberry_container.  By default it tries
to use the `juneberry/cudadev:dev` container that would be built with locally. If you downloaded
a different image, one can specify which one to use as a second argument.

To start a container using the dowloaded cpu only image:

```shell script
docker/enter_juneberry_container . amellinger/juneberry:cpudev
```

Users are encouraged to copy the `enter_juneberry_container` script and customize for their needs.

## Using the container

Once the container is started, the user will be inside the `/juneberry` directory by default.
There are a few last minute things we need to do:

* Install Juneberry (from a python package perspective)
* Setup up user id mapping so outputs use proper user/group ids
* Activate shell completion

This is required _every_ time the container is started, because these files and scripts
are not available when the image was created.

```shell script
pip install -e .
scripts/set_user.sh
source scripts/juneberry_completion.sh
```

# Creating an environment

It is possible to install Juneberry in a virtual environment, but requires a specific ordering and set of versions
in order to get the platforms to properly install. This installation 

## Project Layout

The above project layout is assumed.  However, the manual installation will create another
directory inside the project directory.  Perform this installation **INSIDE** the project root.

## Set up virtual environment

Any virtual environment can be used such as venv or pyenv/venv. The first step is to construct and enter 
such a virtual environment. What follows is an example using pyenv/venv. 

```shell script
pyenv virtualenv 3.8.5 jb
pyenv activate jb
python3 -m pip install --upgrade pip
```

## Clone MMDetection and Juneberry

```shell script
# Get a copy of mmdetection
git clone --depth 1 --branch v2.18.0 https://github.com/open-mmlab/mmdetection.git /mmdetection

# Get a copy of juneberry
git clone https://github.com/cmu-sei/juneberry.git
```

## Install from Requirements

By default, pip will NOT honor the ordering.  So, we feed the requirements in one at a time.

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
   
# Check to see that it installed fine
python mmdet/utils/collect_env.py
```

## Install Detectron2

```shell script
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
   
# For OSX
# CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
```

## Install Juneberry Itself

Once everything else has been installed, then install Juneberry.

```shell script
# From the juneberry directory.
pip install -e .
```

## Add a juneberry.ini

See the overview for a deeper discussion of the Juneberry ini.  For now create a file called 
`juneberry.ini` _inside_ your newly cloned juneberry directory with these contents:

```shell script
[DEFAULT]
tensorboard_root = /path-to-project-root/juneberry
data_root = /path-to-project-root/dataroot
tensorboard_root = /path-to-project-root/tensorboard
```

## Bash Completion

Juneberry is ready to go.  As an extra convenince we have Junberry completion support.

To activate bash completion `source` the completion setup:

```shell script
source scripts/juneberry_completion.sh
```





