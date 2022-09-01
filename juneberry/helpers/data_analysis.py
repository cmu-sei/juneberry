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

import itertools
import logging
import math

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from juneberry.config.dataset import DatasetConfigBuilder
import juneberry.filesystem as jb_fs

logger = logging.getLogger(__name__)


class DatasetPathHelper:
    """
    This class organizes paths inside a dataset for quick access to annotation and image paths
    """
    def __init__(self, lab, ws_key='default', dr_key='default', dataset='', train_path='', val_path='', test_path='',
                 image_dir_paths=[], anno_file_paths=[]):
        """
        :param lab: Juneberry lab containing workspace and data root paths, keys are applied to lab so that user can
        access self.lab.data_root() or self.lab.workspace() without typing in keys
        :param ws_key: Key to workspace inside Juneberry lab
        :param dr_key: Key to data root inside juneberry lab
        :param dataset: Relative path of dataset to data root
        :param train_path: Relative path of training directory to dataset path
        :param val_path: Relative path of validation directory to dataset path
        :param test_path: Relative path of test directory to dataset path
        :param image_dir_paths: This parameter can be a list or a string. If list, entries are relative paths of image
        directory to train/val/test paths in that order. If string, relative path of image directory across
        train/val/test paths**
        :param anno_file_paths: This parameter can be a list or a string. If list, entries are relative paths of
        annotation files to train/val/test paths in that order. If string, relative path of annotation file across
        train/val/test paths**

        ** If train/val/test directories are missing, pad list with None for missing entries.
        """
        self.lab = lab.create_copy_from_keys(ws_key, dr_key)

        self.data_path = Path(lab.data_root()) / dataset
        self.dataset_name = str(dataset)
        self.sources = ['train', 'val', 'test']
        if not train_path:
            self.sources.remove('train')
        else: 
            self.train_path = self.data_path / train_path
        if not val_path:
            self.sources.remove('val')
        else:
            self.val_path = self.data_path / val_path
        if not test_path:
            self.sources.remove('test')
        else:
            self.test_path = self.data_path / test_path

        if isinstance(image_dir_paths, str):
            img_paths = self.get_paths_dict(image_dir_paths).values()
            self.train_img_path, self.val_img_path, self.test_img_path = img_paths
        elif isinstance(image_dir_paths, list):
            img_paths = [self.data_path / subpath for subpath in image_dir_paths]

        anno_file_paths = self._fix_file_paths(self.get_paths_dict(anno_file_paths).values())
        self.train_anno_path, self.val_anno_path, self.test_anno_path = anno_file_paths

        self.anno_sources = ['train', 'val', 'test']
        if not self.train_anno_path:
            self.anno_sources.remove('train')
        if not self.val_anno_path:
            self.anno_sources.remove('val')
        if not self.test_anno_path:
            self.anno_sources.remove('test')

    @staticmethod
    def _fix_file_paths(paths):
        return [path if (path is not None and path.exists() and path.is_file) else None for path in paths]

    def get_workspace(self):
        return self.lab.workspace(self.ws_key)

    def get_data_root(self):
        return self.lab.data_root(self.dr_key)

    def get_paths_dict(self, sub_dir=''):
        """
        Concatenates relative path to all source paths (train/val/test)

        :param sub_dir: sub directory relative to train/val/test paths
        :return: a dictionary with source (train/val/test) as the keys
            and absolute paths as the values
        """
        sources = ['train', 'val', 'test']
        paths = [self.train_path, self.val_path, self.test_path]
        if type(sub_dir) is list:
            return {src: (path / dir if dir is not None else None)
                    for src, path, dir in itertools.zip_longest(sources, paths, sub_dir)}
                   
        return {src: path / sub_dir for src, path in zip(sources, paths)}

    def get_image_dir_paths_dict(self):
        """
        Get image directories across sources

        :return: dictionary where keys are sources (train/val/test) and values are
            absolute paths to image directories
        """
        sources = ['train', 'val', 'test']
        paths = [self.train_img_path, self.val_img_path, self.test_img_path]
        return {src: path for src, path in zip(sources, paths)}

    def get_image_filepaths_dict(self):
        """
        Get image filepaths across sources

        :return: dictionary where keys are sources (train/val/test) and values are
            a list of absolute paths to all image files under that source
        """
        sources = ['train', 'val', 'test']
        paths = [self.get_train_image_filepaths(), 
                 self.get_val_image_filepaths(),
                 self.get_test_image_filepaths()]
        return {src: path for src, path in zip(sources, paths)}

    def get_anno_paths_dict(self):
        """
        Get paths to coco annotations files across sources

        :return: a dictionary where keys are sources (train/val/test) and values
            are absolute paths to coco annotations files
        """
        sources = ['train', 'val', 'test']
        paths = [self.train_anno_path, self.val_anno_path, self.test_anno_path]
        return {src: path for src, path in zip(sources, paths)}

    def get_train_image_filepaths(self):
        """
        Get list of image filepaths with .png or .jpg extensions under the
            training directory for this dataset

        :return: list of image filepaths with supported extensions
        """
        extensions = ['.png', '.jgp']
        filepaths = [fp for fp in self.train_img_path.glob('**/*') if fp.suffix in extensions]
        return filepaths

    def get_val_image_filepaths(self):
        """
        Get list of image filepaths with .png or .jpg extensions under the
            validation directory for this dataset

        :return: list of image filepaths with supported extensions
        """
        extensions = ['.png', '.jgp']
        filepaths = [fp for fp in self.val_img_path.glob('**/*') if fp.suffix in extensions]
        return filepaths

    def get_test_image_filepaths(self):
        """
        Get list of image filepaths with .png or .jpg extensions under the
            test directory for this dataset

        :return: list of image filepaths with supported extensions
        """
        extensions = ['.png', '.jgp']
        filepaths = [fp for fp in self.test_img_path.glob('**/*') if fp.suffix in extensions]
        return filepaths


class DatasetDataframe:
    """
    Builds pandas dataframes of images and annotations for a dataset using
        COCO format annotations.
    """
    def __init__(self, dataset_path_manager):
        """
        Initializes image and annotation dataframes for the dataset.
            Image dataframe contains fields for
            - width: width in pixels
            - height: height in pixels
            - file_name: file name of the image
            - img_file_path: absolute path to image
            - source: train/val/test
            - size_in_MB: size of image file in megabytes
            - id: id extracted from filename if COCO annotations
                not provided or id in COCO annotations
                Note: ids may be reused across sources, do not
                    expect to be unique

            Annotations dataframe contains fields for
            - image_id: from COCO annotations file
                *** Do not expect this to be unique for each image in whole dataset
            - area: segmentation area from COCO annotations file
            - bbox: LTWH from COCO annotations file
            - iscrowd: from COCO annotations file
            - category_id: from COCO annotations file
            - segmentation: from COCO annotations file
            - source: train/val/test
            - file_name: file name of the image
            - img_file_path: absolute file path of the image
            - label: category in human readable string
            - bbox_area: bbox width x bbox height
        """
        self.dataset_path_manager = dataset_path_manager
        self.image_df = None
        self.anno_df = None
        self._build_image_df()
        self._build_anno_df()

    @staticmethod
    def _load_json(path):
        return jb_fs.load_file(path)

    def _build_image_df(self):
        srcs = self.dataset_path_manager.sources
        anno_paths = self.dataset_path_manager.get_anno_paths_dict()
        image_dir_paths = self.dataset_path_manager.get_image_dir_paths_dict()
        image_filepaths = self.dataset_path_manager.get_image_filepaths_dict()
        dfs = []
        for src in srcs:
            images = []
            n_imgs = 0
            if anno_paths[src] is not None:
                content = self._load_json(anno_paths[src])
                images = content['images']
                for img in images:
                    img['source'] = src
                    img['size_in_MB'] = (image_dir_paths[src] / img['file_name']).stat().st_size / 1000000
                    img['img_file_path'] = image_dir_paths[src] / img['file_name']
            else:
                n_imgs = len(image_filepaths[src])
                for f in image_filepaths[src]:
                    with Image.open(f) as im:
                        id = int(f.stem[1:])
                        images.append({
                                        'id': id,
                                        'file_name': f.name, 
                                        'height': im.height,
                                        'width': im.width,
                                        'source': src,
                                        'size_in_MB': f.stat().st_size / 1000000,
                                        'img_file_path': str(f)
                                        })

            df = pd.DataFrame.from_dict(images)
            dfs.append(df)
        if len(dfs) > 1:
            self.image_df = pd.concat(dfs, axis=0)
        elif len(dfs) == 1:
            self.image_df = dfs[0]
        else:
            self.image_df = None

    def _build_anno_df(self):
        srcs = self.dataset_path_manager.anno_sources
        anno_paths = self.dataset_path_manager.get_anno_paths_dict()
        image_dir_paths = self.dataset_path_manager.get_image_dir_paths_dict()
        image_filepaths = self.dataset_path_manager.get_image_filepaths_dict()
        
        dfs = []
        for src in srcs:
            content = self._load_json(anno_paths[src])
            categories = {cat['id']: cat for cat in content['categories']}
            annotations = content['annotations']
            ann_data = []
            for a in annotations:
                a['source'] = src
                a['file_name'] = content['images'][a['image_id']]['file_name']
                a['img_file_path'] = image_dir_paths[src] / a['file_name']
                cat_id = a['category_id']
                cat = categories[cat_id]
                a['label'] = cat['name']
                bbox = a['bbox']
                a['bbox_area'] = bbox[2] * bbox[3]
                ann_data.append(a)

            df = pd.DataFrame.from_dict(ann_data)
            dfs.append(df)
        if len(dfs) > 1:
            self.anno_df = pd.concat(dfs, axis=0)
        elif len(dfs) == 1:
            self.anno_df = dfs[0]
        else:
            self.anno_df = None

    def get_annotations_df_for_images_with_label(self, label, anno_df=None):
        """
        Returns an annotations dataframe from only images that include an
            annotation with the specified label
        """
        if anno_df is None:
            anno_df = self.anno_df
        label_anno_df = anno_df.loc[anno_df['label'] == label]
        label_images = label_anno_df.groupby(['img_file_path'])
        label_img_annos = []
        for img in label_images:
            filepath = img[0]
            df = anno_df.loc[anno_df['img_file_path'] == filepath]
            label_img_annos.append(df)
        label_img_annos_df = pd.concat(label_img_annos, axis=0)
        return label_img_annos_df

    def generate_dataset_config_sources_from_df(self, image_df=None, description='', sampling_count=None,
                                                sampling_fraction=None, dataset_config_builder=None):
        """
        Creates source config dictionary for copy and paste into dataset config
        :param image_df: If None, the original image dataframe is used. If specified,
            finds image ids missing from the original image dataframe to pass to
            DatasetConfigBuilder
        :param description: Description of the source dataset. This string is appended
            with the number of images in the dataset, omitting the images to remove
        :param sampling_count: See juneberry.documentation.data_set_specification
        :param sampling_fraction: See juneberry.documentation.data_set_specification
        :param dataset_config_builder: If specified, sources are added to config in builder
        :returns: list of sources represented as dictionaries

        TO DO:
        1) Test in juneberry pipeline, are correct images removed?
        2) parameter image_df requires same fields as self.image_df
        """
        if image_df is None:
            image_df = self.image_df
        # Get annotations that are missing from self.anno_df
        merged_image_df = image_df.merge(self.image_df, how='outer', indicator=True)
        merged_image_df['remove_id'] = merged_image_df['_merge'] == 'right_only'
        
        sources_groups = merged_image_df.groupby('source')
        sources = []
        for grp in sources_groups:
            src = grp[0]
            ids = sources_groups['id', 'remove_id'].get_group(src)
            remove_all = ids.all().get('remove_id')
            if not remove_all:
                remove_image_ids = list(ids.loc[ids['remove_id']]['id'])
                keep_image_ids = ids.loc[ids['remove_id'] is False]

                anno_path = self.dataset_path_manager.get_anno_paths_dict()[src] \
                    .relative_to(self.dataset_path_manager.lab.data_root())
                l_description = f'{description}, {keep_image_ids.shape[0]} images in source'
                args = [self.dataset_path_manager.lab, anno_path,
                        f'{self.dataset_path_manager.dataset_name} {src}; {l_description}',
                        remove_image_ids, sampling_count, sampling_fraction, '']

                if dataset_config_builder:
                    source = dataset_config_builder.add_source(*args)
                else:
                    source = DatasetConfigBuilder.create_source(*args)
                sources.append(source)
        return sources


class FigureManager:
    """
    Organizes matplotlib figures into titled sections with titled plots. Figures are organized
        by keys. If a figure is too large and needs to be split up, the value at key is a
        dictionary of sub-figures and the keys are numbered indexes.
    """
    def __init__(self, results_dir='.', fig_size=(5, 5), dpi=100, section_title_fontsize=16, 
                 plot_title_fontsize=12, axis_label_fontsize=10, tick_fontsize=8):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.fig_size = fig_size
        self.dpi = dpi
        self.section_title_fontsize = section_title_fontsize
        self.plot_title_fontsize = plot_title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.tick_fontsize = tick_fontsize
        self.figures = {}
        self.sections = {}

    def _initialize_figure_subplots(self, fig_key, nrows, ncols, fig_key_index=None):
        if fig_key not in self.figures:
            logger.error(f"{fig_key} does not exist; cannot add subplot. Doing nothing.")
        axes = np.empty((nrows, ncols), dtype=type(Axes))
        if fig_key_index is not None:
            figure = self.figures[fig_key][fig_key_index]
        else:
            figure = self.figures[fig_key]

        for r in range(nrows):
            for c in range(ncols):
                ax = figure.add_subplot(nrows, ncols, r*ncols + c + 1)
                axes[r, c] = ax
        return axes

    @staticmethod
    def _add_line_to_plot(ax, x_pos, text):
        line = ax.axvline(x=x_pos)
        ax.annotate(text, xy=line.get_xydata(),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom')

    @staticmethod
    def _set_fig_constrained(fig):
        fig.set_tight_layout(True)
        fig.tight_layout(pad=1.1)

    def save_figures_as_pdf(self, filename='report.pdf'):
        with PdfPages(self.results_dir / filename) as pdf:
            for fig in self.figures.values():
                pdf.savefig(fig)

    def save_figures_as_pngs(self):
        for key, fig in self.figures.items():
            fig.savefig(self.results_dir/(key + '.png'))

    def save_plots_as_pngs(self):
        for fig in self.figures.values():
            for ax in fig.axes:
                tmp_fig = plt.Figure(self.fig_size, tigh_layout=True)
                tmp_fig.add_subplot(ax)
                tmp_fig.savefig(self.results_dir / (tmp_fig.title.replace('\s', '_') + '.png'))

    def add_figure(self, fig_key, figure, overwrite=False):
        if fig_key in self.figures and overwrite:
            logger.warning(f"Figure with fig_key {fig_key} already exists, and overwrite is False. Doing nothing.")
            return
        self.figures[fig_key] = figure

    def create_figure(self, fig_key, fig_size=None, dpi=None, overwrite=False, nsections=1, 
                      total_nrows=1, total_ncols=1, max_rows_per_figure=5):
        if fig_key in self.figures and overwrite:
            logger.warning(f"Figure with fig_key {fig_key} already exists, and overwrite is False. Doing nothing.")
            return
        if nsections > 0:
            if nsections <= max_rows_per_figure:
                figure = plt.figure(figsize=(self.fig_size if fig_size is None else fig_size), tight_layout=True,
                                    dpi=(self.dpi if dpi is None else dpi))
                self.figures[fig_key] = figure
                self.sections[fig_key] = figure.subplots(nrows=nsections, ncols=1)
                if not isinstance(self.sections[fig_key], np.ndarray):
                    self.sections[fig_key] = [self.sections[fig_key]]
                for s in self.sections[fig_key]:
                    s.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
                    s._frameon = False
                axes = self._initialize_figure_subplots(fig_key, total_nrows, total_ncols)
            else:
                n_figs = math.ceil(nsections / max_rows_per_figure)
                self.figures[fig_key] = {}
                self.sections[fig_key] = {}
                axes = np.empty((0, total_ncols))
                for idx in range(n_figs):
                    figure = plt.figure(figsize=(self.fig_size if fig_size is None else fig_size), tight_layout=True,
                                        dpi=(self.dpi if dpi is None else dpi))
                    self.figures[fig_key][idx] = figure
                    nsects = ((nsections % max_rows_per_figure) or max_rows_per_figure) if idx == n_figs-1 \
                        else max_rows_per_figure
                    self.sections[fig_key][idx] = figure.subplots(nrows=nsects, ncols=1)
                    if not isinstance(self.sections[fig_key][idx], np.ndarray):
                        self.sections[fig_key][idx] = [self.sections[fig_key][idx]]
                    for s in self.sections[fig_key][idx]:
                        s.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
                        s._frameon = False
                    l_axes = self._initialize_figure_subplots(fig_key, nsects, total_ncols, idx)
                    axes = np.concatenate([axes, l_axes])
        return axes

    def set_section_titles(self, fig_key, titles=[]):
        sections = self.sections[fig_key]
        if isinstance(sections, dict):
            sections = [sect for sects in self.sections[fig_key].values() for sect in sects]
        if len(sections) != len(titles):
            logger.error(f"Number of section titles {len(titles)} does not match number of sections {len(sections)}!")
            return
        for s, t in zip(sections, titles):
            s.set_title(t, fontsize=self.section_title_fontsize, pad=15)

    def set_section_title(self, fig_key, section_index, title):
        sections = self.sections[fig_key]
        if sections is dict:
            sections = [sect for sects in self.sections[fig_key].values() for sect in sects]
        self.sections[section_index].set_title(title, fontsize=self.section_title_fontsize, pad=20)

    def format_plot(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=self.plot_title_fontsize)
        ax.set_xlabel(xlabel, fontsize=self.axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.axis_label_fontsize)
        ax.tick_params(labelsize=self.tick_fontsize)
        ax.grid(b=True, alpha=0.5)

    def add_mean_and_median_lines_to_plot(self, ax, mean, median):
        self._add_line_to_plot(ax, mean, f'mean: {mean}')
        self._add_line_to_plot(ax, median, f'mean: {median}')

    @staticmethod
    def plot_x_turn_off_scientific_notation(ax):
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    def generate_hist(self, fig_key, df, ax, title, xlabel, ylabel, log=False):
        """
        Generates general histogram plot from dataframe, using all columns present
        """
        hist = df.hist(ax=ax, log=log)
        
        self.format_plot(ax, title, xlabel, ylabel)
        if isinstance(self.figures[fig_key], Figure):
            self._set_fig_constrained(self.figures[fig_key])
        else:
            [self._set_fig_constrained(fig) for fig in self.figures[fig_key].values()]

    def generate_bar(self, fig_key, df, ax, title, xlabel, ylabel, log=False):
        """
        Generates general plot from dataframe, using all columns present
        """
        bar = df.plot(kind='bar', ax=ax, logy=log)
        
        self.format_plot(ax, title, xlabel, ylabel)
        if isinstance(self.figures[fig_key], Figure):
            self._set_fig_constrained(self.figures[fig_key])
        else:
            [self._set_fig_constrained(fig) for fig in self.figures[fig_key].values()]
            