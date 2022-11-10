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
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-05.html
# CUDA 11.6.2, Driver 510 or later, Python 3.8.?, pytorch 1.12.0a0+bd13bc6
FROM nvcr.io/nvidia/pytorch:22.04-py3

# ============ BASE PLATFORM ============
# Took parts of this from https://github.com/databricks/containers/blob/master/ubuntu/gpu/cuda-11/base/Dockerfile

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1-mesa-glx figlet sudo tzdata tmux vim emacs nano \
    && apt-get install --yes \
        openjdk-8-jdk \
        iproute2 \
        bash \
        sudo \
        coreutils \
        procps \
    && /var/lib/dpkg/info/ca-certificates-java.postinst configure \
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

# NOTE: This container differs in that we don't install torch or torchvision as they
# come as part of the base install.

# Some of these may No-Op because they are in the pytorch distribution
# Some of these Juneberry may not need, but many tools do.
# NOTE: We do NOT install pytorch as it comes in this nvidia base container
RUN pip3 install llvmlite==0.38.0 --ignore-installed
RUN pip3 install --root-user-action=ignore \
    adversarial-robustness-toolbox \
    doit numpy pycocotools matplotlib pillow prodict hjson jsonschema \
    sklearn tensorboard \
    torch-summary\>=1.4.5 albumentations \
    pandas brambox pyyaml natsort \
    opacus==0.14.0 \
    protobuf==3.16.0 onnx onnxruntime-gpu \
    tf2onnx \
    opencv-python==4.5.5.62 \
    tqdm \
    pytest pylint \
    ray==1.13.0 jsonpath-ng \
    torchmetrics

# Necessary to install Pillow.libs?
RUN pip3 install --upgrade Pillow

# ============ DETECTRON2 ============

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ============ MMDETECTION ============

# We MUST use the FORCE_CUDA=1 to get the MMCV to compile with CUDA and MMDetection to include it.
# For some reason they can't detect the cuda drivers.
ENV FORCE_CUDA="1"

#RUN MMCV_WITH_OPS=1 pip3 install mmcv-full
RUN MMCV_WITH_OPS=1 pip3 install mmcv-full==1.4.8
#RUN MMCV_WITH_OPS=1 pip3 install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# Build MMDetection
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone --depth 1 --branch v2.23.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip3 install -r requirements/build.txt
RUN pip3 install -v -e .

# ============ R ============
# Copied from: https://github.com/databricks/containers/blob/master/ubuntu/R/Dockerfile

# We add RStudio's debian source to install the latest r-base version (4.1)
# We are using the more secure long form of pgp key ID of marutter@gmail.com
# based on these instructions (avoiding firewall issue for some users):
# https://cran.rstudio.com/bin/linux/ubuntu/#secure-apt
RUN apt-get update \
  && apt-get install --yes software-properties-common apt-transport-https \
  && sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
  && sudo apt-key adv -a --export E298A3A825C0D65DFD57CBB651716619E084DAB9 | sudo apt-key add - \
  && add-apt-repository -y "deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
  && apt-get update \
  && apt-get install --yes \
    libssl-dev \
    r-base \
    r-base-dev \
  && add-apt-repository -r "deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
  && apt-key del E298A3A825C0D65DFD57CBB651716619E084DAB9 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# hwriterPlus is used by Databricks to display output in notebook cells
# hwriterPlus is removed for newer version of R, so we hardcode the dependency to archived version
# Rserve allows Spark to communicate with a local R process to run R code
RUN R -e "options(repos = list(MRAN = 'https://mran.microsoft.com/snapshot/2022-04-08', CRAN = 'https://cran.microsoft.com/')); install.packages(c('hwriter', 'TeachingDemos', 'htmltools'))" \
 && R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/hwriterPlus/hwriterPlus_1.0-3.tar.gz', repos=NULL, type='source')" \
 && R -e "install.packages('Rserve', repos='http://rforge.net/')"

# ============ JUNEBERRY PATHS ============
# Since everything is mounted to specific directories, we can specify data root and tensorboard.

ENV JUNEBERRY_DATA_ROOT="/dataroot"
ENV JUNEBERRY_TENSORBOARD="/tensorboard"

# ============ CONVENIENCE ============

# Add some settings to the bashrc to make it easier for folks to know we are in a container
ENV JUNEBERRY_CONTAINER_VERSION="cudabricks:v12.3"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CudaBricks:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "alias jb_comp='source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "alias jb_setup='pip install -e /juneberry; pip install -e .; source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "if [ -f ./container_start.sh ]; then" >> /root/.bashrc; \
    echo "    echo 'SOURCING bash ./container_start.sh'"  >> /root/.bashrc; \
    echo "    source ./container_start.sh" >> /root/.bashrc; \
    echo "fi" >> /root/.bashrc; \
    echo "figlet -w 120 CUDA - ${JUNEBERRY_CONTAINER_VERSION}" >> /root/.bashrc

