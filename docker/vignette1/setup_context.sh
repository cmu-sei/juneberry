#! /usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "This script requires one argument: a path to the context directory."
  exit -1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make all the necessary dirs
[ ! -d ${1} ] && mdkir -p "${1}"
[ ! -d ${1}/dataroot ] && mdkir -p "${1}/dataroot"

# Copy all the files we need, if not there
cp ${SCRIPT_DIR}/juneberry.ini ${1}/.

# If the cifar data isn't there

# Clone Juneberry
if [ ! -d ${1}/juneberry ]; then
  git clone --depth 1 https://github.com/cmu-sei/juneberry.git ${1}/juneberry
fi