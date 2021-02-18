#! /usr/bin/env python3

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import logging
import importlib
import inspect


def construct_instance(fq_name, kwargs):
    """
    Constructs an instance of the class from the fully qualified name expanding
    the kwargs during construction.  e.g. fq_name(**kwargs)
    :param fq_name: The fully qualified class name.
    :param kwargs: Keyword args (a dict) to be expanded.
    :return: An instance of the class
    """

    class_data = fq_name.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    if kwargs is None:
        kwargs = {}

    module = importlib.import_module(module_path)
    return getattr(module, class_str)(**kwargs)


def verify_parameters(code, kwargs, err_message):
    """
    Used to verify the that the parameters in kwarg match the signature of the code.
    :param code: The code fragment's that has the signature to check.
    :param kwargs: The kwargs to look for
    :param err_message: An error message to show if the signature doesn't match.
    :return:
    """
    params = inspect.signature(code).parameters
    verified = True

    # The kwargs should all be in the function.  Anything left should have a default
    param_keys = set(params.keys())
    param_keys.discard('self')

    # Make sure they have all the ones we asked for.
    missing = set(kwargs.keys()) - set(param_keys)
    for param_name in missing:
        logging.error(f"Missing param '{param_name}' on function {err_message}")
        verified = False

    remaining = set(param_keys) - set(kwargs.keys())
    for param_name in remaining:
        if params[param_name].default == inspect.Parameter.empty:
            logging.error(f"Param '{param_name}' not passed for {err_message}")
            verified = False

    return verified


def verify_method(module_path: str, class_name: str, method_name: str, method_args):
    """
    Used to verify that the specific module has the specified class with that
    method that takes those arguments.
    :param module_path: The path to the module.
    :param class_name: The class name.
    :param method_name: The method name
    :param method_args: A dictionary of arguments.  The values are ignored.
    :return: True if everything is found, otherwise false.
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logging.error(f"Failed to find module '{module_path}'")
        return False

    # Make sure we have the class
    if not hasattr(mod, class_name):
        logging.error(f"Missing class '{class_name}' in module='{module_path}'")
        return False

    my_class = getattr(mod, class_name)

    # Make sure we have the method
    if not hasattr(my_class, method_name):
        logging.error(f"Missing method '{method_name}' from module='{module_path}', class='{class_name}'")
        return False

    my_method = getattr(my_class, method_name)

    # Make sure the method has all the parameters
    return verify_parameters(my_method, method_args,
                             f"module='{module_path}', class='{class_name}', method='{method_name}'")


def invoke_method(module_path: str, class_name: str, method_name: str, method_args, dry_run=False):
    """
    Loads the modules, loads instantiates the class and executes the method expanding those
    arguments,.
    :param module_path: The path to the module.
    :param class_name: The class name.
    :param method_name: The method name
    :param method_args: A dictionary of arguments.  The values are ignored.
    :param dry_run: If set to true verifies the module, class, method and signature without execution.
    :return: The result of invoking the function. If dry_run None is returned.
    """
    if dry_run:
        verify_method(module_path, class_name, method_name, method_args)
        return None

    # We don't do any special error handling.  There is no reason to continue if anything fails.
    # Load the module, find the class, find the method and call it.
    mod = importlib.import_module(module_path)
    my_class = getattr(mod, class_name)
    my_instance = my_class()
    my_method = getattr(my_instance, method_name)
    return my_method(**method_args)


def load_verify_fqn_function(fully_qualified_name: str, function_args):
    """
    Loads and verifies a function specified by fully qualified name.
    :param fully_qualified_name: The fully qualified function name and path.
    :param function_args: A dictionary of arguments.  The values are ignored.
    :return: The function if verified, None otherwise.
    """
    return load_verify_function(*split_fully_qualified_name(fully_qualified_name), function_args)


def load_verify_function(module_path: str, function_name: str, function_args):
    """
    Used to verify that the specified module has the specified function that takes those arguments.
    :param module_path: The path to the module.
    :param function_name: The method name
    :param function_args: A dictionary of arguments.  The values are ignored.
    :return: The function if verified, None otherwise.
    """

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logging.error(f"Failed to find module '{module_path}'")
        return False

    # Make sure we have the func
    if not hasattr(mod, function_name):
        logging.error(f"Missing function '{function_name}' from module='{module_path}'")
        return False

    my_function = getattr(mod, function_name)

    # Make sure we have the arguments
    if verify_parameters(my_function, function_args, f"module='{module_path}', function='{function_name}'"):
        return my_function

    return None


def invoke_function(module_path: str, function_name: str, kwargs, dry_run=False):
    """
    Loads the modules, loads instantiates the class and executes the method expanding those
    arguments,.
    :param module_path: The path to the module.
    :param function_name: The method name
    :param kwargs: A dictionary of arguments.  The values are ignored.
    :param dry_run: If set to true verifies the module, class, method and signature without execution.
    :return: The result of invoking the function. If dry_run None is returned.
    """
    if dry_run:
        load_verify_function(module_path, function_name, kwargs)
        return None

    # We don't do any special error handling.  There is no reason to continue if anything fails.
    # Load the module, find the class, find the method and call it.
    mod = importlib.import_module(module_path)
    my_function = getattr(mod, function_name)
    return my_function(**kwargs)


def split_fully_qualified_name(fully_qualified_name: str):
    """
    Splits the fully qualified dotted name into path and name parts.
    :param fully_qualified_name:
    :return: Tuple of path part and name part
    """
    class_data = fully_qualified_name.split(".")
    path_part = ".".join(class_data[:-1])
    name_part = class_data[-1]

    return path_part, name_part


if __name__ == "__main__":
    print("Move along")
