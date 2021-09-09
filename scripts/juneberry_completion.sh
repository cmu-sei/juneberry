#/usr/bin/env bash

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

# USAGE
#
# This script needs to be 'sourced' from within your bash process so that its changes persist. e.g.:
#
#    source scripts/juneberry_completion.sh
#
# This will need to be done every time you start a new bash process, unless you add that to your bashrc.

# ======================================
# MAINTENANCE DOCS

# TL;DR on how bash completion works
# Functions are registered with 'complete'
# Functions get called to provide suggestions to the user via the COMPREPLY variables.
# Functions are pass:
#    COMP_WORDS - An ARRAY (not string) of current arguments to the command
#    COMP_CWORD - The number if the current argument they are trying to file. Remember 0 is the script name
#    COMP_LINE - The current command line
# Builtins:
#    complete - Used to register things with bash
#    compgen - Used to help generate, filter and modify lists of options
#

# You can find more details here
# https://www.gnu.org/software/bash/manual/html_node/Programmable-Completion.html#Programmable-Completion
# https://iridakos.com/programming/2018/03/01/bash-programmable-completion-tutorial

# For other compgen automatic things see -A here
# https://www.gnu.org/software/bash/manual/html_node/Programmable-Completion-Builtins.html#index-compgen
# as suggested by
# https://unix.stackexchange.com/questions/33236/configure-autocomplete-for-the-first-argument-leave-the-others-alone

# =============================

# COMMON FUNCTIONS

# Find things of a specific name ($2) somewhere in a directory ($1).  So like config.json files in models.
function find_things() {
  for ITEM in $(find ${1} -iname ${2}); do
    # Strip the leading part
    ITEM="${ITEM#${1}/}"

    # Strip the trailing part
    ITEM="${ITEM%/${2}}"

    # Put this in our our "return value"
    TRAVREPLY+=("${ITEM}")
  done
}

# This function walks the directory supplied as $1 and "returns" an array
# of names in TRAVREPLY. This is useful for things where we don't know specific file names
function traverse() {
  if [ -d "${1}" ]; then
    for item in $(ls "$1/$2"); do
      if [[ -d ${1}/${2}/${item} ]]; then
        ITEM_NAME="${2}/${item}"

        # Strip the trailing slash
        ITEM_NAME="${ITEM_NAME#/}"

        # Put this in our our "return value"
        TRAVREPLY+=("${ITEM_NAME}")
        traverse "${1}" "${2}/${item}"
      fi
    done
  fi
}

# Uses the traverse function to find model directories and
# then uses compgen to downselect to useful options
function _add_model_dirs() {
  TRAVREPLY=()
  find_things models config.json
  # Flatten all the replies into one list for compgen
  FOO="${TRAVREPLY[@]}"
  COMPREPLY+=($(compgen -W "${FOO}" "${COMP_WORDS[${1}]}"))
}

# Uses the traverse function to find model directories and
# then uses compgen to downselect to useful options
function _add_experiment_configs() {
  TRAVREPLY=()
  find_things experiments config.json
  # Flatten all the replies into one list for compgen
  FOO="${TRAVREPLY[@]}"
  COMPREPLY+=($(compgen -W "${FOO}" "${COMP_WORDS[${1}]}"))
}

# Uses the traverse function to find model directories and
# then uses compgen to downselect to useful options
function _add_experiment_outlines() {
  TRAVREPLY=()
  find_things experiments experiment_outline.json
  # Flatten all the replies into one list for compgen
  FOO="${TRAVREPLY[@]}"
  COMPREPLY+=($(compgen -W "${FOO}" "${COMP_WORDS[${1}]}"))
}

# =============================================
# Specific functions for each script we have

_jb_train_comp() {
  case ${COMP_CWORD} in
  1)
    _add_model_dirs 1
    ;;
  esac
  return 0
}

_jb_evaluate_data_comp() {
  # By default we turn off the normal behavior

  compopt +o default
  case ${COMP_CWORD} in
  1)
    _add_model_dirs 1
    ;;
  2)
    # We want the "normal" autocomplete behavior here.
    # NOTE The compgen -f does NOT work as expected as directory names satisfy -f.
    compopt -o default
    #COMPREPLY=($(compgen -f -- "${COMP_WORDS[COMP_CWORD]}"))
    ;;
  esac
  return 0
}

_jb_run_experiment_comp() {
  case ${COMP_CWORD} in
  1)
    _add_experiment_configs 1
    _add_experiment_outlines 1
    ;;
  esac
  return 0
}

_jb_experiment_to_rules_comp() {
  case ${COMP_CWORD} in
  1)
    _add_experiment_configs 1
    ;;
  esac
  return 0
}

_jb_rules_to_pydoit_comp() {
  case ${COMP_CWORD} in
  1)
    _add_experiment_configs 1
    ;;
  esac
  return 0
}

_jb_generate_experiments_comp() {
  case ${COMP_CWORD} in
  1)
    _add_experiment_outlines 1
    ;;
  esac
  return 0
}

# Sign them up!

complete -F _jb_train_comp jb_train
complete -F _jb_evaluate_data_comp jb_evaluate_data
complete -F _jb_run_experiment_comp jb_run_experiment
complete -F _jb_experiment_to_rules_comp jb_experiment_to_rules
complete -F _jb_generate_experiments_comp jb_generate_experiments
