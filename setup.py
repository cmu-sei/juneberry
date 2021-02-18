#!/usr/bin/env python

import setuptools

install_requires = [
    "numpy",
    "matplotlib",
    "pillow",
    "adversarial-robustness-toolbox",
    "sklearn",
    "tensorboard",
    "torch",
    "torchvision",
    "torch-summary"
]

bin_scripts = [
    'bin/jb_calculate_dataset_norms',
    'bin/jb_make_predictions',
    'bin/jb_ini_tool',
    'bin/jb_plot_roc',
    'bin/jb_preview_filelist',
    'bin/jb_run_experiment',
    'bin/jb_summary_report',
    'bin/jb_generate_experiments',
    'bin/jb_train'
]

setuptools.setup(
    name='Juneberry',
    version='0.3',
    description='Juneberry Machine Learning Pipeline',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    scripts=bin_scripts,
    python_requires='>=3.7',
)
