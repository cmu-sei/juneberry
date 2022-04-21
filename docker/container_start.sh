#! /usr/bin/env bash

# Setup juneberry
echo "Installing Juneberry..."
pip install -e /juneberry

# Add in the bash completion
source /juneberry/scripts/juneberry_completion.sh

# Install any workspace code
if [ -e "./setup.py" ]; then
    echo "Installing workspace..."
    pip install -e .
fi

