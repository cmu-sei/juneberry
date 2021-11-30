#! /usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "This script requires one argument, the part BEFORE the '.Dockerfile'"
  echo "e.g. 'cudadev' or 'cpudev'"
  exit -1
fi

REV="dev"
if [ $# -eq 2 ]; then
	REV=${2}
fi

TARGET_TAG="juneberry/${1}:${REV}"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOCKERFILE=${SCRIPT_DIR}/${1}.Dockerfile

echo "Building: ${DOCKERFILE} into ${TARGET_TAG}"

docker build \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg NO_PROXY=${NO_PROXY} \
  --build-arg no_proxy=${no_proxy} \
  --network=host -f "${DOCKERFILE}" -t ${TARGET_TAG} ${SCRIPT_DIR}
  