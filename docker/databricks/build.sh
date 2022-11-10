#! /usr/bin/env bash

TARGET_TAG="juneberry/cudabricks:dev"

# SCRIPT DIR is our context dir.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Building: Dockerfile into ${TARGET_TAG}"

docker build --no-cache \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg NO_PROXY=${NO_PROXY} \
  --build-arg no_proxy=${no_proxy} \
  --network=host -f Dockerfile -t ${TARGET_TAG} ${SCRIPT_DIR}
