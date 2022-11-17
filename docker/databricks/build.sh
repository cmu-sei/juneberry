#! /usr/bin/env bash

REV="dev"
if [ $# -eq 1 ]; then
	REV=${1}
fi

# SCRIPT DIR is our context dir.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

TARGET_TAG="juneberry/cudabricks-base:${REV}"

echo "Building: Base into ${TARGET_TAG}"

docker build --no-cache \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg NO_PROXY=${NO_PROXY} \
  --build-arg no_proxy=${no_proxy} \
  --network=host -f base.Dockerfile -t ${TARGET_TAG} ${SCRIPT_DIR}

TARGET_TAG="juneberry/cudabricks:${REV}"

echo "Building: Ganglia into ${TARGET_TAG}"

docker build --no-cache \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg NO_PROXY=${NO_PROXY} \
  --build-arg no_proxy=${no_proxy} \
  --network=host -f ganglia.Dockerfile -t ${TARGET_TAG} ${SCRIPT_DIR}
