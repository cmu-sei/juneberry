#! /usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "This script requires one argument, the part BEFORE the '.Dockerfile'"
  exit -1
fi

TARGET_TAG="juneberry/${1}:dev"
echo "Building ${1}.Dockerfile into ${TARGET_TAG}"

docker build \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg NO_PROXY=${NO_PROXY} \
  --build-arg no_proxy=${no_proxy} \
  --network=host -f ${1}.Dockerfile -t ${TARGET_TAG} .
