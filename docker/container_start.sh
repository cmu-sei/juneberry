#! /usr/bin/env bash

# Setup juneberry
pip install -e /juneberry

# Add in the bash completion
source /juneberry/scripts/juneberry_completion.sh

# For non Darwin platforms setup the user
# OSX Docker desktop already does this right
if [ "${HOST_UNAME}" != "Darwin" ]; then
  /juneberry/docker/set_user.sh
fi

# Install any worksapc code
if [ -f "./setup.py" ]; then
    pip install -e .
fi
