Building Juneberry Docker Containers
==========

***

**WARNING: These containers and scripts create containers with NO SECURITY PRACTICES,
such as separate user accounts, unprivileged users, etc.**

**USE AT YOUR OWN RISK.**

***


# Overview

This directory contains **Dockerfile**s, scripts for building various images for use with Juneberry, and some
convenience scripts for running images.


# Dockerfiles

## cpudev.Dockerfile

An image with full cpu development support. Checkpoints NOT included. 

## cudadev.Dockerfile

The image to be used for development on cuda platforms. Checkpoints NOT included.

# Building

To build a particular docker image, use normal docker build commands, or the convenience script `build.sh`. 
The build script takes one argument, which is the part before the period in the Dockerfile name. 
For example, to build the cudadev image use `./build.sh cudadev`.

# Automatic command execution on start

When the containers start up they will look for a script called "container_start.sh" in the /juneberry
directory (well, the one mounted as /juneberry) and, if found, will execute it. This is useful for
automatically installing juneberry such as `pip install -e .` or running some test or something else.

# Container layout

The development process is based around the following lab layout:

* /juneberry - Mount from the external users directory
* /datasets - Mount to the external data directories.
* /tensorboard - Mount point for tensorboard output
* /root/.cache/torch/hub - Mounted for model caches for PyTorch and MMDetection
* /root/.torch - Mounted for model caches for Detectron2

The containers set the dataroot and tensorboard environment variables automatically. The current working 
directory will be chosen as the workspace, unless specified otherwise.

# Convenience Scripts

In addition to the script for building images, there are also some convenience scripts here.

## enter_juneberry_container

This script starts up a **temporary** 'cudadev' container on your host using all available gpus.
It assumes a project directory structure that contains a set of special subdirectories where each
subdirectory becomes a mound point within the container. This parent directory should be passed as the argument
into enter_juneberry_container.  

The structure is:

* juneberry <- This is the Juneberry repo that was pulled
* datasets <- This is where the source data is located, i.e. the "dataroot" that Juneberry will look at.
* tensorboard <- This is where the tensorboard outputs will be stored.
* cache <- This where the model downloads are cached.

For example, if this structure was in the directory `~/proj` then to use the `enter_juneberry_container`
change into `~/proj` and run: 

`./juneberry/docker/enter_juneberry_container .`

See the comments within the script for how to configure it to use a cpu-only container, adjust environment
variables, add other mount points and configure gpus.

## set_user.sh

This optional convenience script can create a user inside the container to match an external
user, resulting in the correct permissions for volumes mounted inside the container. See the script
for an explanation of how it works in conjunction with enter_juneberry_container.

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.
