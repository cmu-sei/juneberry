BUILDING THE VIGNETTE IMAGE
========

# Introduction

This document is intended for Juneberry developers and describes the process for building 
the Vignette Docker image. To use a built image, pull it from docker hub as `cmusei/juneberry:vignette`.

To properly build this image we need to prepare a context directory and must contain:

* cifar-10-batches-py/ - directions below
* juneberry/ - a clone of the repo modified/pruned as desired, consider --depth 1
* juneberry.ini - copy the one from this directory there

With the context directory set up, then run `build.sh context` to bulld the image.

## Download CIFAR-10

The cifar10 dataset can be downloaded via the torchvision dataset. This
can be done by running the script `load_cifar.py` which uses torchvision to download the dataset
into the 'dataroot' directory and untar it. The untarred 'cifar-10-batches-py'
directory should then be copied to the context directory. The Dockerfile will
copy this directory when the container is built. Although the original tar.gz can be deleted,
it is convenient to keep the tar.gz around so that it can be copied to different machines and unpacked
so it doesn't need to be downloaded again.

