# ======================================================================================================================
# Juneberry - General Release
#
# Copyright 2021 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software subject to its own license.
#
# DM21-0884
#
# ======================================================================================================================

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

# Make sure we are using the latest pip
RUN pip3 install --upgrade pip

# ============ Tensorflow ============

# We install tensorflow FIRST because it is very picky about versions of things such
# as numpy. And when other things need numpy (such as to compile against it, such as
# pycocotools) we can't have tensorflow moving to old versions.
#RUN pip install tensorflow tensorflow-datasets
RUN pip install tensorflow==2.7.0 tensorflow-datasets==4.4.0

# ============ JUNEBERRY ============

# Some of these may No-Op because they are in the pytorch distribution
# Some of these Juneberry may not need, but many tools do.
# NOTE: We use these torch version because that is what comes with the cuda container.
RUN pip3 install llvmlite==0.38.0 --ignore-installed
RUN pip3 install adversarial-robustness-toolbox \
    doit numpy pycocotools matplotlib pillow prodict hjson jsonschema \
    sklearn tensorboard \
    torch==1.10.0 torchvision \
    torch-summary\>=1.4.5 albumentations \
    pandas brambox pyyaml natsort \
    opacus==0.14.0 \
    protobuf==3.16.0 onnx onnxruntime \
    tf2onnx \
    opencv-python \
    tqdm \
    pytest pylint

# ============ DETECTRON2 ============

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ============ MMDETECTION STUFF ============

# We don't force CUDA here because we don't expect any
# ENV FORCE_CUDA="1"

#RUN pip install mmcv-full
RUN pip install mmcv-full==1.4.8
#RUN pip install mmcv-full==1.4.8 https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html

# This is pretty straightforward
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone --depth 1 --branch v2.23.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
WORKDIR /

# ============ JUNEBERRY PATHS ============
# Since everything is mounted to specific directories, we can specify data root and tensorboard.

ENV JUNEBERRY_DATA_ROOT="/dataroot"
ENV JUNEBERRY_TENSORBOARD="/tensorboard"

# ============ CONVENIENCE ============

# Add some settings to the bashrc to make it easier for folks to know we are in a container
ENV JUNEBERRY_CONTAINER_VERSION="cpudev:v11"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CPUDev:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "alias jb_comp='source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "alias jb_setup='pip install -e /juneberry; pip install -e .; source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "if [ -f ./container_start.sh ]; then" >> /root/.bashrc; \
    echo "    echo 'SOURCING ./container_start.sh'"  >> /root/.bashrc; \
    echo "    source ./container_start.sh" >> /root/.bashrc; \
    echo "fi" >> /root/.bashrc; \
    echo "figlet -w 120 CPU - ${JUNEBERRY_CONTAINER_VERSION}" >> /root/.bashrc

