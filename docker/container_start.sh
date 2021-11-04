#! /usr/bin/env bash

pip install -e /juneberry
if [ -d "src" ]; then
  export PYTHONPATH="$(pwd)/src:/juneberry"
fi
source /juneberry/scripts/juneberry_completion.sh
/juneberry/docker/set_user.sh
