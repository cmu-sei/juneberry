#! /usr/bin/env bash

# Setup juneberry
echo "Installing Juneberry..."
pip install -e /juneberry

# Add in the bash completion
source /juneberry/scripts/juneberry_completion.sh

# For non Darwin platforms setup the user
# OSX Docker desktop already does this right
if [ "${HOST_UNAME}" != "Darwin" ]; then
  /juneberry/docker/set_user.sh
fi

# Install any worksapc code
if [ -e "./setup.py" ]; then
    echo "Installing workspace..."
    pip install -e .
fi
