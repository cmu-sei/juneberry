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
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-11.html#rel-22-11
# NVIDIA CUDA 11.8.0
# NVIDIA driver release 450.51(or later R450), 470.57(or later R470), 510.47(or later R510), or 515.65(or later R515)
# PyTorch container image version 22.11 is based on 1.13.0a0+936e930.
# GPU Requirements: CUDA compute capability 6.0 and later.
FROM nvcr.io/nvidia/pytorch:22.11-py3

# ============ BASE PLATFORM ============

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
# Tensorflow: https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0
RUN pip install tensorflow==2.12.0 tensorflow-datasets==4.7.0 tensorboard==2.12.0 \
                tensorrt==8.6.0 tf2onnx==1.13.0
# Suppress Tensorflow Warnings
ENV TF_CPP_MIN_LOG_LEVEL = "3"

# ============ JUNEBERRY ============

# We do NOT install torch, torchvision as they come as part of the base install.
# We do NOT install pytorch as it comes in this nvidia base container.
# Some of these may No-Op because they are in the pytorch distribution.
# All of these tools are pinned to a specific version to support nvcr.io/nvidia/pytorch:22.11.
RUN pip3 install adversarial-robustness-toolbox==1.12.2 \
    doit==0.36.0 numpy==1.22.2 pycocotools==2.0.6 matplotlib==3.6.2 \
    pillow==9.3.0 prodict==0.8.18 hjson==3.1.0 jsonschema==4.17.0 \
    scikit-learn==0.24.2 \
    torch-summary==1.4.5 albumentations==1.3.0 \
    pandas==1.4.4 brambox==4.1.1 pyyaml==6.0 natsort==8.2.0 \
    opacus==1.3.0 protobuf==3.20.3 onnx==1.13.1 onnxruntime-gpu==1.14.1 \
    opencv-python==4.6.0.66 \
    tqdm==4.64.1 \
    pytest==7.2.0 pylint==2.15.8 \
    ray==2.1.0 jsonpath-ng==1.5.3 \
    torchmetrics==0.10.2

# ============ DETECTRON2 ============

# Use force cuda so that if built on something like github on a none-gpu machine we still get cuda
ENV FORCE_CUDA="1"
# Detectron2 v0.6
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ============ MMDETECTION ============

# Use force cuda so that if built on something like github on a none-gpu machine we still get cuda
ENV FORCE_CUDA="1"

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
ENV JUNEBERRY_CONTAINER_VERSION="cudadev:v13.1"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CudaDev:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "alias jb_comp='source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "alias jb_setup='pip install -e /juneberry; pip install -e .; source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "if [ -f ./container_start.sh ]; then" >> /root/.bashrc; \
    echo "    echo 'SOURCING bash ./container_start.sh'"  >> /root/.bashrc; \
    echo "    source ./container_start.sh" >> /root/.bashrc; \
    echo "fi" >> /root/.bashrc; \
    echo "figlet -w 120 CUDA - ${JUNEBERRY_CONTAINER_VERSION}" >> /root/.bashrc
