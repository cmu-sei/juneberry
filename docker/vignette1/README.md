BUILDING THE VIGNETTE IMAGE
========

# Introduction

This document is intended for Juneberry Developers (not Juneberry users) and describes the process for 
building the self-contained docker image to run the Vignette. Vignette images have a copy of Juneberry
embedded as well as necessary sample data to run a simple experiment.

# Download CIFAR

The first step is to download (or copy) a copy of cifar 10 into the vignette directory.

curl -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Setting up the context directory.

To properly build this image we need to prepare a context directory containing:

* dataroot/cifar-10-batches-py/ - Unzip the tar.gz from above
* juneberry/ - a clone of the repo modified/pruned as desired, consider --depth 1
* juneberry/setup.py - A copy of the stripped down setup.py from this directory.

There is a convenience script `setup_context.sh` which performs these actions.

# Make the image

With the context directory set up, run `build.sh` to build the image.

