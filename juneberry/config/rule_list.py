#! /usr/bin/env python3

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

import datetime
import logging
import sys

from prodict import Prodict, List

import juneberry.config.util as conf_utils
import juneberry.filesystem as jbfs

logger = logging.getLogger(__name__)


class Rule(Prodict):
    clean_extras: List[str]
    command: List[str]
    doc: str
    id: int
    inputs: List[str]
    outputs: List[str]
    requirements: List[str]


class Workflow(Prodict):
    name: str
    rules: List[Rule]


class RulesList(Prodict):
    FORMAT_VERSION = '0.2.0'
    SCHEMA_NAME = 'rules_list_schema.json'

    description: str
    format_version: str
    timestamp: str
    workflows: List[Workflow]

    @staticmethod
    def construct(data: dict, file_path: str = None):
        """
        Load, validate, and construct a config object.
        :param data: The data to use to construct the object.
        :param file_path: Optional path to a file that may have been loaded. Used for logging.
        :return: The constructed object.
        """
        # Validate with our schema
        conf_utils.require_version(data, RulesList.FORMAT_VERSION, file_path, 'RulesList')
        if not conf_utils.validate_schema(data, RulesList.SCHEMA_NAME):
            logger.error(f"Validation errors in {file_path}. See log. EXITING")
            sys.exit(-1)

        # Finally, construct the object
        config = RulesList.from_dict(data)
        return config

    @staticmethod
    def load(data_path: str):
        """
        Loads the RulesList from the provided path, validates, and constructs the RulesList.
        :param data_path: Path to RulesList.
        :return: Loaded, validated, and constructed object.
        """
        # Load the raw file.
        logger.info(f"Loading RULES LIST from {data_path}")
        with open(data_path) as file:
            data = jbfs.load(file)

        # Construct the config.
        return RulesList.construct(data, data_path)

    def save(self, data_path: str) -> None:
        """
        Save the RulesList to the specified resource path.
        :param data_path: The path to the resource.
        :return: None
        """
        conf_utils.validate_and_save_json(self.to_json(), data_path, RulesList.SCHEMA_NAME)

    def to_json(self) -> dict:
        """ :return: A pure dictionary version suitable for serialization to json."""
        return conf_utils.prodict_to_dict(self)

    def get_workflow(self, name: str) -> Workflow:
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow

        raise KeyError(f"Workflow '{name}' does not exist")


def get_mixed_to_snake() -> dict:
    return {
        'formatVersion': 'format_version',
    }


class WorkflowBuilder:
    """
    This class is used to build a workflow.  This class should be constructed by a RulesListsBuilder.
    """

    def __init__(self, workflow, rules_builder):
        """
        Initializes the workflow with this name.
        :param workflow: The name of the workflow.
        :param rules_builder: The rules list builder this is associated with.
        """
        self.workflow = workflow
        self.rules_builder = rules_builder

    def add_rule(self, inputs, outputs, command, doc, clean_extras, requirements) -> int:
        """
        Adds a specific rule for this set of inputs, outputs, commands, and rule requirements.
        :param inputs: A list of inputs that are expected to exist.
        :param outputs: Outputs to be generated by executing this rule.
        :param command: The command to execute.
        :param doc: A documentation string
        :param clean_extras: A list of additional targets to clean.
        :param requirements: IDs of rules that should have been executed before this rule.
        :return: A unique ID of this rule in the overall rules list.
        """
        as_dict = {
            "id": self.rules_builder.next_id,
            "inputs": [str(x) for x in inputs],
            "outputs": [str(x) for x in outputs],
            "command": [str(x) for x in command],
            "doc": doc,
            "requirements": requirements
        }

        if clean_extras and len(clean_extras) > 0:
            as_dict['clean_extras'] = [str(x) for x in clean_extras]

        self.workflow.rules.append(Workflow.from_dict(as_dict))

        self.rules_builder.next_id += 1
        return self.rules_builder.next_id - 1


class RuleListBuilder:
    """
     This class is used to build a rules list which contains a set of named workflows.
     """

    def __init__(self, description):
        """
        Initializes a rules list with a particular description.
        :param description: A description of this rules list.
        """
        self.rules_list = RulesList()
        self.rules_list.description = description
        self.rules_list.workflows = []
        self.rules_list.format_version = RulesList.FORMAT_VERSION
        self.rules_list.timestamp = str(datetime.datetime.now().replace(microsecond=0).isoformat())

        self.next_id = 0

    def get_workflow(self, name) -> WorkflowBuilder:
        """
        Returns the workflow object for the workflow with that name. If the workflow
        does not exist, a new one will be created.
        :param name: The name of the desired workflow.
        :return: The existing workflow builder object for this name or a new workflow builder object.
        """
        for workflow in self.rules_list.workflows:
            if workflow.name == name:
                return WorkflowBuilder(workflow, self)

        # We don't have a workflow for this.  Add one
        workflow = Workflow(name=name, rules=[])
        self.rules_list.workflows.append(workflow)
        return WorkflowBuilder(workflow, self)
