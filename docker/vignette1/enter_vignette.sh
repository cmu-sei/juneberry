#! /usr/bin/env bash

#
CONTAINER="juneberry/vignette1:dev"

# The -rm means the container is destroyed upon exit.  This is okay because the useful data persists
# in the mounted locations.
BASE="-it --rm --network=host --ipc=host"
ENVS="--env HTTP_PROXY --env http_proxy --env HTTPS_PROXY --env https_proxy --env NO_PROXY --env no_proxy"
NAME="--name ${USER}"

# TODO: What if they wanted to save results?

# Assemble the command, show it, and run it.
CMD="docker run ${BASE} ${ENVS} ${NAME} ${CONTAINER} bash"
echo "${CMD}"
${CMD}
