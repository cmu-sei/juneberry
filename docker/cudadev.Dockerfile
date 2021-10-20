# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-02.html
# CUDA 11.2.0, Driver 460.27.04, Python 3.8.?, pytorch 1.8
FROM nvcr.io/nvidia/pytorch:21.02-py3

# ============ BASE PLATFORM ============

# These are needed for opencv - not by default on some platforms
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx figlet \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make sure we are using the latest pip
RUN pip3 install --upgrade pip

# ============ Tensorflow ============

# We install tensorflow FIRST because it is very picky about versions of things such
# as numpy. And when other things need numpy (such as to compile against it, such as
# pycocotools) we can't have tensorflow moving to old versions.
RUN pip install tensorflow tensorflow-datasets

# ============ JUNEBERRY ============

# NOTE: This container differs in that we don't install torch or torchvision as they
# come as part of the base install.

# Some of these may No-Op because they are in the pytorch distribution
# Some of these Juneberry may not need, but many tools do.
RUN pip3 install numpy matplotlib pillow pandas pyyaml natsort \
    pycocotools tensorboard torch-summary>=1.4.5 albumentations \
    opacus sklearn brambox opencv-python prodict jsonschema pytest pylint doit \
    protobuf==3.16.0 onnx onnxruntime-gpu

# ============ DETECTRON2 ============

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# ============ MMDETECTION ============

# We MUST use the FORCE_CUDA=1 to get the MMCV to compile with CUDA and MMDetection to include it.
# For some reason they can't detect the cuda drivers.
ENV FORCE_CUDA="1"

# We can't seem to find a version that matches our cuda version so just compile the one we need
RUN MMCV_WITH_OPS=1 pip3 install mmcv-full

# These do not work right now but might work some day and might be a better way to get specific versions
#RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
#RUN pip3 install mmcv-full==latest+torch1.8.0+cu111 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Build MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip3 install -r requirements/build.txt
RUN pip3 install -v -e .

# ============ JUNEBERRY INIT ============
# Since everything is mounted in the same place we can specify a 'default'
# juneberry ini in the last place we look.  The 'home' directory of
# the user.

COPY juneberry.ini /root/juneberry.ini

# ============ CONVENIENCE ============

# Add some settings to the bashrc to make it easier for folks to know we are in a container
ENV JUNEBERRY_CONTAINER_VERSION="cudadev:v6"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CudaDev:\w\$ '"  >> /root/.bashrc; echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; echo "figlet -w 120 CUDA Development v6" >> /root/.bashrc
RUN echo "if [ -f /juneberry/container_start.sh ]; then echo 'Running bash /juneberry/container_start.sh'; bash /juneberry/container_start.sh; fi" >> /root/.bashrc


