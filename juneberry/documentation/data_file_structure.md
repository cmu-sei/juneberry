Data File Structure
==========

# Introduction
Juneberry will expect the data and datagroups to be in a specific structure.


# Schema
```
data
└───category_x
└───datagroup_x
│   └───category_x
│   └───version_x
│       └───category_x
└───cache
    └───datagroup_x
        └───version_x
            └───description_x
                └───category_x

```

### data
The parent directory of all data. Often the path to this directory will be set in juneberry.ini as DATA_ROOT

### category_x
Name of a group of files that fall into the same identification. Multiple groups are usually needed to train a model.

### datagroup_x
**(optional)** Name of a group of categories that make up a specific datagroup.

### version_x
**(optional)** Name of the version of the datagroup. When multiple versions are used within a data group,
the layout of the data within each version (folder wise) must be identical.

### cache
**(optional)** Directory where cached files will be preprocessed and saved. This is to reduce compute time.

### description_x
Directory which describes the preprocessing used on the caching. Directory will often take on a name 
with the formula {width}x{height}_{colorspace}. Ex. 64x64_gray

# Example
```
data
└───dir1
│   │   file01.png
│   │   file02.png
└───dir2
│   │   file01.png
│   │   file02.png
└───datagroup_1
│   └───version_1
│   │   └───dir1
│   │   │   │   file01.png
│   │   │   │   file02.png
│   │   └───dir2
│   │       │   file01.png
│   │       │   file02.png
│   └───version_2
│       └───dir1
│       │   │   file01.png
│       │   │   file02.png
│       └───dir2
│           │   file01.png
│           │   file02.png
└───datagroup_2
│   └───dir1
│       │   file01.png
│       │   file02.png
│   └───dir2
│       │   file01.png
│       │   file02.png
└───cache
    └───datagroup_1
    │   └───version_1
    │   │   └───24x24_gray
    │   │       └───dir1
    │   │       │   │   file01.png
    │   │       │   │   file02.png
    │   │       └───dir2
    │   │           │   file01.png
    │   │           │   file02.png
    │   └───version_2
    │       └───24x24_gray
    │           └───dir1
    │           │   │   file01.png
    │           │   │   file02.png
    │           └───dir2
    │               │   file01.png
    │               │   file02.png
    └───datagroup_2
        └───24x24_gray
        │   └───dir1
        │   │   │   file01.png
        │   │   │   file02.png
        │   └───dir2
        │       │   file01.png
        │       │   file02.png
        └───64x64_rgb
            └───dir1
            │   │   file01.png
            │   │   file02.png
            └───dir2
                │   file01.png
                │   file02.png
```

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
