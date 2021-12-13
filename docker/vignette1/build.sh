#! /usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "This script requires one argument: a path to the context directory that has juneberry installed."
  exit -1
fi

TARGET_TAG="juneberry/vignette1:dev"

echo "Building: context dir '${1}' into ${TARGET_TAG}"

docker build \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg NO_PROXY=${NO_PROXY} \
  --build-arg no_proxy=${no_proxy} \
  --network=host -f Dockerfile -t ${TARGET_TAG} ${1}
