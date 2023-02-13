# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

# Python 3.8.10 is used in our cudadev.Dockerfile from the nvcr.io/nvidia/pytorch:22.11-py3 base container
# For consistency, we will use this version of Python
FROM python:3.8.10

# ============ BASE PLATFORM ============

# For some reason the base image doesn't always have the right permissions on /tmp
RUN chmod 1777 /tmp

# These are needed for opencv - not by default on some platforms
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1-mesa-glx figlet sudo tzdata tmux vim emacs nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip3 install --upgrade pip

# ============ Tensorflow ============

# We install tensorflow FIRST because it is very picky about versions of things such
# as numpy. And when other things need numpy (such as to compile against it, such as
# pycocotools) we can't have tensorflow moving to old versions.
#RUN pip install tensorflow tensorflow-datasets
# Tensorflow: https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0
RUN pip install tensorflow==2.11.0 tensorflow-datasets==4.7.0

# ============ JUNEBERRY ============

# Some of these may No-Op because they are in the pytorch distribution
# Some of these Juneberry may not need, but many tools do.
# NOTE: For consistency, we use these versions because that is how we build our cuda container.
RUN pip3 install adversarial-robustness-toolbox==1.12.2 \
    doit==0.36.0 numpy==1.22.2 pycocotools==2.0.6 matplotlib==3.6.2 \
    pillow==9.3.0 prodict==0.8.18 hjson==3.1.0 jsonschema==4.17.0 \
    sklearn==0.0.post1 tensorboard==2.11.0 \
    torch==1.13.0 torchvision==0.14.0 \
    torch-summary==1.4.5 albumentations==1.3.0 \
    pandas==1.4.4 brambox==4.1.1 pyyaml==6.0 natsort==8.2.0 \
    opacus==1.3.0 \
    protobuf==3.19.6 onnx==1.12.0 onnxruntime \
    tf2onnx==1.13.0 \
    opencv-python==4.6.0.66 \
    tqdm==4.64.1 \
    pytest==7.2.0 pylint==2.15.8 \
    ray==2.1.0 jsonpath-ng==1.5.3 \
    torchmetrics==0.11.0

# ============ DETECTRON2 ============

# Detectron2 v0.6
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ============ MMDETECTION STUFF ============

# We don't force CUDA here because we don't expect any
# ENV FORCE_CUDA="1"

RUN MMCV_WITH_OPS=1 pip3 install mmcv-full==1.7.0

# Build MMDetection v2.26.0
RUN git clone --depth 1 --branch v2.26.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip3 install -r requirements/build.txt
RUN pip3 install -v -e .

# ============ JUNEBERRY PATHS ============

# Since everything is mounted to specific directories, we can specify data root and tensorboard.
ENV JUNEBERRY_DATA_ROOT="/dataroot"
ENV JUNEBERRY_TENSORBOARD="/tensorboard"

# ============ CONVENIENCE ============

# Add some settings to the bashrc to make it easier for folks to know we are in a container
ENV JUNEBERRY_CONTAINER_VERSION="cpudev:v13.0"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CPUDev:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "alias jb_comp='source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "alias jb_setup='pip install -e /juneberry; pip install -e .; source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "if [ -f ./container_start.sh ]; then" >> /root/.bashrc; \
    echo "    echo 'SOURCING ./container_start.sh'"  >> /root/.bashrc; \
    echo "    source ./container_start.sh" >> /root/.bashrc; \
    echo "fi" >> /root/.bashrc; \
    echo "figlet -w 120 CPU - ${JUNEBERRY_CONTAINER_VERSION}" >> /root/.bashrc
