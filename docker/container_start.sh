#! /usr/bin/env bash

pip install -e /juneberry
if [ -d "src" ]; then
  export PYTHONPATH=/workspace/src:${PYTHONPATH}
fi
source /juneberry/scripts/juneberry_completion.sh

# Only run set_user.sh if the host OS is not OSX (Mac)
# Note: HOST_OS will be written into this file
# when the workspace is created.
if [ "${HOST_OS}" != "Darwin" ]; then
  /juneberry/docker/set_user.sh
fi
