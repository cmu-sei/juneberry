#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - General Release
#
# Copyright 2021 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software subject to its own license.
#
# DM21-0884
#
# ======================================================================================================================

import logging
import importlib
import inspect
import sys

logger = logging.getLogger(__name__)


def takes_args(method, param_names: list) -> bool:
    """
    Determines if the method takes all the parameter name.
    :param method: The method to check
    :param param_names: A list of parameter names that must be in the function.
    :return:
    """
    params = set(inspect.signature(method).parameters.keys())
    for param_name in param_names:
        if param_name not in params:
            return False


def add_optional_args(kwargs: dict, optional_kwargs: dict, method) -> None:
    """
    Adds arguments from optional_kwargs to kwargs if they exist in the method signature.
    :param kwargs: The kwargs
    :param optional_kwargs: The optional_kwargs to search
    :param method: The method
    :return: None
    """
    if optional_kwargs:
        params = set(inspect.signature(method).parameters.keys())
        for k, value in optional_kwargs.items():
            if k in params:
                kwargs[k] = value


def extract_kwargs(instance_dict):
    """
    Examines the provided dict for 'fqcn' (required) and 'kwargs' (optional) and
    extracts those values and returns them as a dict of 'fq_name' and 'kwargs' in preparation
    for passing into other functions in the module.  If 'fqcn' is not found then
    None is returned. If 'kwargs' is not found, an empty dict is returned as the 'kwargs' value.
    :param instance_dict: The dict to examine for the argument.
    :return: None or prepared fq_name and kwargs.
    """
    fqn_key = 'fqcn'
    if instance_dict is None:
        logger.warning(f"In extract_kwargs, no input was provided. Returning None.")
        return None

    fq_name = instance_dict.get(fqn_key, None)

    if fq_name is None:
        logger.warning(f"Failed to get '{fqn_key}' from {instance_dict}.")
        return None

    return {'fq_name': fq_name, 'kwargs': instance_dict.get('kwargs', {})}


def extract_kwarg_names(func):
    """
    Extracts the names of arguments (past the first) that are of type POSITIONAL_OR_KEYWORD
    or KEYWORD_ONLY so we omit positional only, or VAR_KEYWORD args.
    :return:
    """
    params = []
    for i, (k, v) in enumerate(inspect.signature(func).parameters.items()):
        if i != 0 and (v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or v.kind == inspect.Parameter.KEYWORD_ONLY):
            params.append(k)

    return params


def split_fully_qualified_name(fully_qualified_name: str):
    """
    Splits the fully qualified dotted name into path and name parts.
    :param fully_qualified_name: A string of the fully qualified to be split.
    :return: Tuple of path part and name part
    """
    class_data = fully_qualified_name.split(".")
    path_part = ".".join(class_data[:-1])
    name_part = class_data[-1]

    return path_part, name_part


def load_class(fqcn: str):
    """
    Loads the class from the FQCN. The module and class are expected to exist and if not, then we exit.
    :param fqcn: A string indicating which fully qualified class name to load.
    :return: The class or False when the class cannot be found.
    """
    module_path, class_name = split_fully_qualified_name(fqcn)

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logger.error(f"Failed to find module '{module_path}'")
        sys.exit(-1)

    # Make sure we have the class
    if not hasattr(mod, class_name):
        logger.error(f"Missing class '{class_name}' in module='{module_path}'")
        sys.exit(-1)

    return getattr(mod, class_name)


def construct_instance(fq_name, kwargs: dict, optional_kwargs: dict = None):
    """
    Constructs an instance of the class from the fully qualified name expanding
    the kwargs during construction.  e.g. fq_name(**kwargs)
    :param fq_name: The fully qualified class name.
    :param kwargs: Keyword args (a dict) to be expanded.
    :param optional_kwargs: A set of additional kwargs to add IF in the __init__ signature.
    """

    # IMPORTANT
    # While we prefer a class with an __init__ method and a __call__ method, we can
    # actually accept a callable that returns a callable. NOTE: We do NOT look directly
    # at the signature of the __init__ method because for functions this gives us the
    # wrong signature. So the signature of the direct callable works for both.
    # NOTE: We don't actually check that the returns/constructed thing is a callable
    # or has any particular signature.

    module_path, leaf_part = split_fully_qualified_name(fq_name)

    if kwargs is None:
        kwargs = {}
    else:
        kwargs = dict(kwargs)

    # Load the thing that makes the other callable. By default assume it is a function,
    # and the thing we call and inspect is the same.
    module = importlib.import_module(module_path)
    try:
        direct_callable = getattr(module, leaf_part)
        inspection_point = direct_callable
    except AttributeError as e:
        logger.error(f"Error when trying to load {leaf_part} from module path {module_path}")
        raise e

    # Now, if this is NOT a function then we assume it is an object that we construct.
    # Get the init dunder so we can inspect it
    if not inspect.isfunction(inspection_point):
        try:
            inspection_point = getattr(direct_callable, "__init__")
        except AttributeError as e:
            logger.error(f"Error when trying to load get __init__ from non-function: {inspection_point}")
            raise e

    # Get all the parameter names from the signature of the callable
    # and then add any optional kwargs if in the signature
    add_optional_args(kwargs, optional_kwargs, inspection_point)

    # Call the callable and get the callable
    return direct_callable(**kwargs)


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
        logger.error(f"Missing param '{param_name}' on function {err_message}")
        verified = False

    remaining = set(param_keys) - set(kwargs.keys())
    for param_name in remaining:
        if params[param_name].default == inspect.Parameter.empty:
            logger.error(f"Param '{param_name}' not passed for {err_message}")
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
        logger.error(f"Failed to find module '{module_path}'")
        return False

    # Make sure we have the class
    if not hasattr(mod, class_name):
        logger.error(f"Missing class '{class_name}' in module='{module_path}'")
        return False

    my_class = getattr(mod, class_name)

    # Make sure we have the method
    if not hasattr(my_class, method_name):
        logger.error(f"Missing method '{method_name}' from module='{module_path}', class='{class_name}'")
        return False

    my_method = getattr(my_class, method_name)

    # Make sure the method has all the parameters
    return verify_parameters(my_method, method_args,
                             f"module='{module_path}', class='{class_name}', method='{method_name}'")


def invoke_method(module_path: str, class_name: str, method_name: str, method_args, dry_run=False,
                  *, optional_kwargs: dict = None):
    """
    Loads the modules, loads instantiates the class and executes the method expanding those
    arguments,.
    :param module_path: The path to the module.
    :param class_name: The class name.
    :param method_name: The method name
    :param method_args: A dictionary of arguments.  The values are ignored.
    :param dry_run: If set to true verifies the module, class, method and signature without execution.
    :param optional_kwargs: A set of additional kwargs to add IF in the method signature.
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

    add_optional_args(method_args, optional_kwargs, my_method)

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
        logger.error(f"Failed to find module '{module_path}'")
        return False

    # Make sure we have the func
    if not hasattr(mod, function_name):
        logger.error(f"Missing function '{function_name}' from module='{module_path}'")
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


def invoke_call_function_on_class(fqcn: str, args: dict, optional_kwargs=None):
    """
    Invokes the __call__ function on a constructed instance of the class specified by the
    fully-qualified class name and pass in the arguments.
    :param fqcn: The fully-qualified class name.
    :param args: The arguments to pass in.
    :param optional_kwargs: A set of additional kwargs to add IF in the method signature.
    :return: The result of the __call__ function.
    """
    # Split the module name to module and path
    class_data = fqcn.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]
    return invoke_method(module_path=module_path,
                         class_name=class_str,
                         method_name="__call__",
                         method_args=args,
                         dry_run=False,
                         optional_kwargs=optional_kwargs)


if __name__ == "__main__":
    print("Move along")
