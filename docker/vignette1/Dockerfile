FROM python:3.8.5

# ============ BASE PLATFORM ============

# For some reason the base image doesn't always have the right permissions on /tmp
RUN chmod 1777 /tmp

# Get the latest, install some OS packages, and clean up
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx figlet sudo vim emacs nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make sure we are using the latest pip
RUN pip3 install --upgrade pip

# ============ JUNEBERRY ============

# Some of these may No-Op because they are in the pytorch distribution
# Some of these Juneberry may not need, but many tools do.
# NOTE: We use these torch version because that is what comes with the cuda container.
RUN pip3 install doit numpy pycocotools matplotlib pillow prodict hjson jsonschema \
    sklearn tensorboard \
    torch==1.8.0 torchvision==0.9.0 \
    torch-summary>=1.4.5 albumentations \
    pandas brambox pyyaml natsort \
#    protobuf==3.16.0 onnx onnxruntime \
#    opencv-python \
    pytest pylint

# ============ JUNEBERRY  ============

# Now, copy the ENTIRE state of juneberry into the container and install it
COPY juneberry /juneberry
RUN pip install -e /juneberry

# ============ TENSORBOARD ============

RUN mkdir /tensorboard
ENV JUNEBERRY_TENSORBOARD="/tensorboard"

# ============ DATAROOT =============

# Copy in the files to the dataroot
RUN mkdir /dataroot
COPY dataroot/cifar-10-batches-py /dataroot/cifar-10-batches-py/.
ENV JUNEBERRY_DATA_ROOT="/dataroot"

# ============ CONVENIENCE ============

# Add some settings to the bashrc to make it easier for folks to know we are in a container
ENV JUNEBERRY_CONTAINER_VERSION="vignette1:v1"
RUN echo "source /juneberry/scripts/juneberry_completion.sh" >> /root/.bashrc
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\u@\h+vignette1:\w\$ '" >> /root/.bashrc; \
    echo "alias ll='ls -l --color=auto'" >> /root/.bashrc; \
    echo "figlet -w 120 CPU Vignette-1 v1" >> /root/.bashrc

WORKDIR /juneberry
