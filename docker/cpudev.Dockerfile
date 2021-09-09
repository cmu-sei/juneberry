FROM python:3.8.5

# ============ BASE PLATFORM ============

# For some reason on GPU5 the bamboo image doesn't have /tmp set right...
RUN chmod 1777 /tmp

# These are needed for opencv - not by default on some platforms
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx figlet \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make sure we are using the latest pip
RUN pip3 install --upgrade pip

# ============ JUNEBERRY ============

# TODO - There is an opencv-python-headless that might be better
# Some of these may No-Op because they are in the pytorch distribution
RUN pip3 install numpy matplotlib pillow pandas pyyaml natsort \
    torch==1.8.0 torchvision==0.9.0 \
    pycocotools tensorboard torch-summary>=1.4.5 albumentations \
    opacus sklearn brambox opencv-python prodict jsonschema pytest pylint doit

# ============ DETECTRON2 ============

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.4'

# ============ MMDETECTION STUFF ============

# We don't force CUDA here because we don't expect any
# ENV FORCE_CUDA="1"

# Could we do a lite version?
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html

# This is pretty straightforward
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
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
ENV JUNEBERRY_CONTAINER_VERSION="cpudev:v4"
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+CPUDev:\w\$ '"  >> /root/.bashrc; echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; echo "figlet -w 120 CPU Development v4" >> /root/.bashrc
RUN echo "if [ -f /juneberry/container_start.sh ]; then echo 'Running bash /juneberry/container_start.sh'; bash /juneberry/container_start.sh; fi" >> /root/.bashrc


