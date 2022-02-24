#! /usr/bin/env bash

# This script makes a context directory BESIDE the setup_context script and populates
# it with all the necessary pieces.

# Find out where this script is, if they ran it from somewhere else
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd ${SCRIPT_DIR}

if [ ! -f cifar-10-python.tar.gz ]; then
    echo "This script requires a copy of cifar-10-python.tar.gz to be beside this script."
    echo "Try 'curl -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'"
    echo "Exiting."
    exit -1
fi


# If a context directory already exists, bail
if [ -d ./context ]; then
    echo "A context directory already exists BESIDE the setup_context.py script. Exiting."
    exit -1
fi

# Make the context dir
mkdir context

# Set up dataroot
echo "Extracting cifar-10 data..."
mkdir context/dataroot
pushd context/dataroot
tar -xzf ${SCRIPT_DIR}/cifar-10-python.tar.gz
popd

# Clone Juneberry
echo "Cloning Juneberry repo..."
git clone --depth 1 https://github.com/cmu-sei/juneberry.git context/juneberry

# Fix the setup.py
echo "Copying in simplified setup.py"
cp ${SCRIPT_DIR}/setup.py context/juneberry/setup.py
