FROM python:3.8.5

# ============ BASE PLATFORM ============

# For some reason the base image doesn't always have the right permissions on /tmp
RUN chmod 1777 /tmp

# These are needed for opencv - not by default on some platforms
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx figlet sudo \
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
RUN pip3 install doit numpy pycocotools matplotlib pillow prodict hjson jsonschema \
    sklearn tensorboard \
    torch==1.8.0 torchvision==0.9.0 \
    torch-summary\>=1.4.5 albumentations \
    pandas brambox pyyaml natsort \
    opacus==0.14.0 \
    protobuf==3.16.0 onnx onnxruntime \
    tf2onnx \
    opencv-python \
    tqdm \
    pytest pylint

# ============ DETECTRON2 ============

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
#RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# ============ MMDETECTION STUFF ============

# We don't force CUDA here because we don't expect any
# ENV FORCE_CUDA="1"

#RUN pip install mmcv-full
RUN pip install mmcv-full==1.3.17
#RUN pip install mmcv-full==1.3.17 https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html

# This is pretty straightforward
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone --depth 1 --branch v2.18.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
WORKDIR /

# ============ JUNEBERRY INIT ============
# Since everything is mounted in the same place we can specify a 'default'
# juneberry ini in the last place we look.  The 'home' directory of
# the user.

COPY juneberry.ini /root/juneberry.ini

# ============ CONVENIENCE ============

# Add some settings to the bashrc to make it easier for folks to know we are in a container
ENV JUNEBERRY_CONTAINER_VERSION="cpudev:v9.1"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CPUDev:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "figlet -w 120 CPU Development v9.1" >> /root/.bashrc; \
    echo "if [ -f ./container_start.sh ]; then" >> /root/.bashrc; \
    echo "    echo 'SOURCING ./container_start.sh'"  >> /root/.bashrc; \
    echo "    source ./container_start.sh" >> /root/.bashrc; \
    echo "else" >> /root/.bashrc; \
    echo "    echo './container_start.sh NOT found.'"  >> /root/.bashrc; \
    echo "fi" >> /root/.bashrc

