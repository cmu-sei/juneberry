Maintenance
============

# Overview

This page covers standards, techniques, etc.

# Python 
## Installation & Usage
- It is suggested that you use a virtual environment
    - The choice is yours, but the use of [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) is suggested
    - This framework is designed for python versions 3.7+
    - You can use [pyenv](https://github.com/pyenv/pyenv) to manage different python versions on your computer
    - Required python packages are documented in the Pipfile
## Coding Guidelines
- Standard python naming standards, etc.
- Configure logging for info

### Structure
For coding structure we like to separate the argument parsing from the business functionality to allow the 
script to be loaded externally and have the business functions called. The usual pattern is to have the main() 
routine do all the argument parsing, validity checks, open files, and read configs then call the business functions. 
So, something like:

```
#! /usr/bin/env python3
 
import argparse
import logging
 
 
# Use some better name than business logic...
def hello_world(data_root):
    print(f"Hello world from {data_root}")
 
 
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
 
    parser = argparse.ArgumentParser(description="Pithy comment here.")
    parser.add_argument('dataRoot', help='Root of data directory')
 
    args = parser.parse_args()
 
    hello_world(args.dataRoot)
 
 
if __name__ == "__main__":
    main()
```

## JSON
- snake_case names for properties
- 4 space indent

## Git
- We use standard gitflow style
- We default to squash on merges
- When branches get confusing, prefer to rebase to a new branch with the suffix "-merge"
- Most tasks are features

# Design Notes

## Config files
Juneberry is a config-driven system. Different types of config files store different concerns, for example the structure 
of a model would be in one type of config, while the contents of a dataset would be defined in another. Regardless of 
config type, a similar process is used to load config data, so all configs follow a similar philosophy.

1) Files or blob are in common python-happy formats such as json, yaml or toml
1) Files are loaded into pure python data structures
1) Version conversions are performed
1) Data structures are validated using jsonschema regardless of file format
1) Validated structures are loaded into convenience data models such as Prodict. 

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
