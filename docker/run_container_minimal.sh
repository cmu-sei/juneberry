#! /usr/bin/env bash

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

# =================================================================================================
# WARNING: These images and scripts create containers with NO SECURITY PRACTICES, such as
# separate user accounts, unprivileged users, etc.
#
# USE AT YOUR OWN RISK
# =================================================================================================

# This script provides a starting point for creating your own container launcher. If your layout
# follows the basic Juneberry lab layout, then this script should basically work as-is.
#
# Run this script from inside your workspace of choice.

WS=${PWD}
LAB="$(dirname "$WS")"
CACHE="${LAB}/cache"
docker run -it --rm --network=host --ipc=host --name ${USER} \
    --env HTTP_PROXY --env http_proxy --env HTTPS_PROXY --env https_proxy --env NO_PROXY --env no_proxy \
    -e USER_NAME=${USER} -e USER_ID=$(id -u ${USER}) -e USER_GID=$(id -g ${USER}) -e HOST_UNAME=$(uname) \
    -v ${WS}:/workspace -w /workspace \
    -v ${LAB}/juneberry:/juneberry \
    -v ${LAB}/dataroot:/dataroot:ro \
    -v ${LAB}/tensorboard:/tensorboard \
    -v ${CACHE}/hub:/root/.cache/torch/hub \
    -v ${CACHE}/torch:/root/.torch \
    -v ${CACHE}/tensorflow:/root/tensorflow_datasets \
    cmusei/juneberry:cpudev \
    bash