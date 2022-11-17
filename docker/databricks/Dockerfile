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
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-04.html
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

# Use force cuda so that if built on something like github on a none-gpu machine we still get cuda
ENV FORCE_CUDA="1"

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ============ MMDETECTION ============

# Use force cuda so that if built on something like github on a none-gpu machine we still get cuda
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

# NOTE: If building inside behind a proxy we need to add a proxy option to the keyserver
# this should go right after "--keyesever <addr>" on the "apt-key adv" line:
# --keyserver-options "--http-proxy=${http_proxy}"

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
ENV JUNEBERRY_CONTAINER_VERSION="cudabricks:v12.4"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CudaBricks:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "alias jb_comp='source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "alias jb_setup='pip install -e /juneberry; pip install -e .; source /juneberry/scripts/juneberry_completion.sh'" >> /root/.bashrc; \
    echo "if [ -f ./container_start.sh ]; then" >> /root/.bashrc; \
    echo "    echo 'SOURCING bash ./container_start.sh'"  >> /root/.bashrc; \
    echo "    source ./container_start.sh" >> /root/.bashrc; \
    echo "fi" >> /root/.bashrc; \
    echo "figlet -w 120 CUDA - ${JUNEBERRY_CONTAINER_VERSION}" >> /root/.bashrc

# ============ GANGLIA ============

# https://github.com/databricks/containers/blob/master/experimental/ubuntu/ganglia/Dockerfile

RUN apt-get update \
  && apt-get install -y openssh-server \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /databricks

# set environment variables which Databricks shell script (in tarball) uses upon startup to create crontab for Ganglia to capture metrics
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -q -y --force-yes --fix-missing --ignore-missing \
        ganglia-monitor \
        ganglia-webfrontend \
        ganglia-monitor-python \
        python3-pip \
        rsync \
        cron \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade Ganglia to 3.7.2 to patch XSS bug, see CJ-15250
# Upgrade Ganglia to 3.7.4 and use private forked repo to patch several security bugs, see CJ-20114
# SC-17279: We run gmetad as user ganglia, so change the owner from nobody to ganglia for the rrd directory
RUN cd /tmp \
  && export GANGLIA_WEB=ganglia-web-3.7.4-db-4 \
  && wget https://s3-us-west-2.amazonaws.com/databricks-build-files/$GANGLIA_WEB.tar.gz \
  && tar xvzf $GANGLIA_WEB.tar.gz \
  && cd $GANGLIA_WEB \
  && make install \
  && chown ganglia:ganglia /var/lib/ganglia/rrds
# Install Phantom.JS
RUN cd /tmp \
  && export PHANTOM_JS="phantomjs-2.1.1-linux-x86_64" \
  && wget https://s3-us-west-2.amazonaws.com/databricks-build-files/$PHANTOM_JS.tar.bz2 \
  && tar xvjf $PHANTOM_JS.tar.bz2 \
  && mv $PHANTOM_JS /usr/local/share \
  && ln -sf /usr/local/share/$PHANTOM_JS/bin/phantomjs /usr/local/bin
# Apache2 config. The `sites-enabled` config files are loaded into the container
# later.
RUN rm /etc/apache2/sites-enabled/* && a2enmod proxy && a2enmod proxy_http

RUN mkdir -p /etc/monit/conf.d

ADD ganglia-monitor-not-active /etc/monit/conf.d
ADD gmetad-not-active /etc/monit/conf.d
ADD spark-slave-not-active /etc/monit/conf.d

RUN echo $'\n\
check process spark-slave with pidfile /tmp/spark-root-org.apache.spark.deploy.worker.Worker-1.pid\n\
      start program = "/databricks/spark/scripts/restart-workers"\n\
      stop program = "/databricks/spark/scripts/kill_worker.sh"\n\
' > /etc/monit/conf.d/spark-slave-not-active


# add Ganglia configuration file indicating the DocumentRoot - Databricks checks this to enable Ganglia upon cluster startup
RUN mkdir -p /etc/apache2/sites-enabled
ADD ganglia.conf /etc/apache2/sites-enabled
RUN chmod 775 /etc/apache2/sites-enabled/ganglia.conf

ADD gconf/* /etc/ganglia/
RUN mkdir -p /databricks/spark/scripts/ganglia/
RUN mkdir -p /databricks/spark/scripts/
ADD start_spark_slave.sh /databricks/spark/scripts/start_spark_slave.sh

# add local monit shell script in the right location
RUN mkdir -p /etc/init.d
ADD monit /etc/init.d
RUN chmod 775 /etc/init.d/monit
