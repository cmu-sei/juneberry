#! /usr/bin/env bash

pip install -e /juneberry
if [ -d "src" ]; then
  export PYTHONPATH=/workspace/src:${PYTHONPATH}
fi
source /juneberry/scripts/juneberry_completion.sh
/juneberry/docker/set_user.sh
