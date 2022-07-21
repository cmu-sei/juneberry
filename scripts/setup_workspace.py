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

import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(filename)s:%(levelname)s - %(message)s")


def create_dir(some_dir: Path) -> None:
    """
    Create a directory if it doesn't already exist.
    :param some_dir: The directory to create.
    :return: None
    """
    # Doing it this way instead of using the flags on mkdir
    # so we can print the status message.
    if not some_dir.exists():
        logging.info(f"Creating {some_dir.name} directory...")
        some_dir.mkdir()


def create_workspace_dirs(workspace_dir: str) -> None:
    """
    Create the Juneberry workspace directory and subdirectories, if necessary.
    :param workspace_dir: The Juneberry workspace directory.
    :return: None
    """
    workspace_dir_path = Path(workspace_dir)
    create_dir(workspace_dir_path)
    create_dir(workspace_dir_path / "data_sets")
    create_dir(workspace_dir_path / "experiments")
    create_dir(workspace_dir_path / "models")
    create_dir(workspace_dir_path / "drafts")


def create_package_and_setup(workspace_dir: str, requirements_list: list):
    # Make a package directory and put in an empty init file
    package_dir = Path(workspace_dir) / workspace_dir
    create_dir(package_dir)
    (package_dir / "__init__.py").touch()

    # TODO: Add juneberry version?
    requires = ['juneberry']
    requires.extend(requirements_list)
    install_requires = [f"install_requires = [ {', '.join(requires)} ]\n"]

    setup_args = [
        f"setuptools.setup(\n",
        f"    name='{workspace_dir}',\n",
        f"    version='0.1',\n",
        f"    packages=setuptools.find_packages(),\n",
        f"    install_requires=install_requires,\n",
        f"    python_required='>=3.7'\n"
        f")\n",
    ]

    with open(str(Path(workspace_dir) / "setup.py"), "w") as out_file:
        out_file.write("#! /usr/bin/env python3\n\n")
        out_file.write("import setuptools\n\n")
        out_file.writelines(install_requires)
        out_file.write("\n\n")
        out_file.writelines(setup_args)
        out_file.write("\n")

    # Now make a container start

    container_sh_start = [
        "#! /usr/bin/env bash\n\n",
        "# Setup juneberry\n",
        "echo \"Installing Juneberry...\"\n",
        "pip install -e /juneberry\n\n",
        "echo \"Installing requiremnts..\"\n"]

    container_sh_start.extend([f"pip install -e /lab/{x}\n" for x in requirements_list])

    container_sh_end = """
# Add in the bash completion
source /juneberry/scripts/juneberry_completion.sh

# Install any workspace code
if [ -e "./setup.py" ]; then
    echo "Installing workspace..."
    pip install -e .
fi      
"""

    with open(str(Path(workspace_dir) / "container_start.sh"), "w") as out_file:
        out_file.writelines(container_sh_start)
        out_file.writelines(container_sh_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Juneberry workspace to set up.")
    parser.add_argument("requirements", help="Python requirements.", type=str, nargs='*', )
    args = parser.parse_args()

    logging.info(f"Setting up workspace in {args.workspace} ...")

    logging.info("Checking specified requirements ...")
    for package in args.requirements:
        if not Path(f"./{package}/setup.py").exists():
            logging.error(f"Required package ./{package}/setup.py does not exist. No workspace created.")
            sys.exit(1)

    # Create the Juneberry workspace directories and files.
    # Return 0 for success, 1 for failure.
    try:
        create_workspace_dirs(args.workspace)
        create_package_and_setup(args.workspace, args.requirements)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
