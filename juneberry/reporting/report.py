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
from pathlib import Path

logger = logging.getLogger(__name__)


class Report:
    """
    This is the base class for all reports.
    """
    def __init__(self, output_str: str = ""):
        # If an empty output string was provided, set the output directory for the report to the
        # current directory.
        if output_str == "":
            logger.warning(f"An output path for the report was not provided. Saving the report to the "
                           f"current working directory.")
            self.output_dir = Path.cwd()
        else:
            # If a string was provided, first convert it to a Path.
            self.output_dir = Path(output_str)

            # Now check the final component in the Path. If it contains a "." that means the final
            # component contains a file extension. Therefore the provided output_str was for a file
            # and not a directory. Therefore, the output_dir must be set to the parent directory of
            # the file.
            if "." in self.output_dir.parts[-1]:
                self.output_dir = self.output_dir.parent

        # Create the output directory (and any parent directories) if it does not exist.
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def create_report(self) -> None:
        """
        Creates the report file and writes it to the desired output file.
        :return: Nothing
        """
        logger.warning("'create_report' is not implemented in the base Report class.")
