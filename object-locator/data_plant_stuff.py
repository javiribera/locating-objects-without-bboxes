__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import os
import random
from collections import OrderedDict

from PIL import Image
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import xmltodict
from parse import parse

from . import get_image_size

IMG_EXTENSIONS = ['.png', '.jpeg', '.jpg', '.tiff']

torch.set_default_dtype(torch.float32)


class XMLDataset(torch.utils.data.Dataset):
    def __init__(self,
                 directory,
                 transforms=None,
                 max_dataset_size=float('inf'),
                 ignore_gt=False,
                 seed=0):
        """XMLDataset.
        The sample images of this dataset must be all inside one directory.
         Inside the same directory, there must be one XML file as described by
         https://communityhub.purdue.edu/groups/phenosorg/wiki/APIspecs
         (minimum XML API version is v.0.4.0).
         If there is no XML file, metrics will not be computed,
         and only estimations will be provided.
        :param directory: Directory with all the images and the XML file.
        :param transform: Transform to be applied to each image.
        :param max_dataset_size: Only use the first N images in the directory.
        :param ignore_gt: Ignore the GT in the XML file,
                          i.e, provide samples without plant locations or counts.
        :param seed: Random seed.
        """

        self.root_dir = directory
        self.transforms = transforms

        # Get list of files in the dataset directory,
        # and the filename of the XML
        listfiles = os.listdir(directory)
        xml_filenames = [f for f in listfiles if f.endswith('.xml')]
        if len(xml_filenames) == 1:
            xml_filename = xml_filenames[0]
        elif len(xml_filenames) == 0:
            xml_filename = None
        else:
            print(f"E: there is more than one XML file in '{directory}'")
            exit(-1)

        # Ignore files that are not images
        listfiles = [f for f in listfiles
                     if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]

        # Shuffle list of files
        np.random.seed(seed)
        random.shuffle(listfiles)

        if len(listfiles) == 0:
            raise ValueError(f"There are no images in '{directory}'")

        if xml_filename is None:
            print('W: The dataset directory %s does not contain '
                  'a XML file with groundtruth. Metrics will not be evaluated.'
                  'Only estimations will be returned.' % directory)

        self.there_is_gt = (xml_filename is not None) and (not ignore_gt)

        # Read all XML as a string
        with open(os.path.join(directory, xml_filename), 'r') as fd:
            xml_str = fd.read()

        # Convert to dictionary
        # (some elements we expect to have multiple repetitions,
        #  so put them in a list)
        xml_dict = xmltodict.parse(xml_str,
                                   force_list=['field',
                                               'panel',
                                               'plot',
                                               'plant'])

        # Check API version number
        try:
            api_version = xml_dict['fields']['@apiversion']
        except:
            # An unknown version number means it's the very first one
            # when we did not have api version numbers
            api_version = '0.1.0'
        major_version, minor_version, _ = parse('{}.{}.{}', api_version)
        major_version = int(major_version)
        minor_version = int(minor_version)
        if not(major_version == 0 and minor_version == 4):
            raise ValueError('An XML with API v0.4 is required.')

        # Create the dictionary with the entire dataset
        dictt = {}
        for field in xml_dict['fields']['field']:
            for panel in field['panels']['panel']:
                for plot in panel['plots']['plot']:

                    if self.there_is_gt and \
                            not('plant_count' in plot and \
                                'plants' in plot):
                        # There is GT for some plots but not this one
                        continue

                    filename = plot['orthophoto_chop_filename']
                    if 'plot_number' in plot:
                        plot_number = plot['plot_number']
                    else:
                        plot_number = 'unknown'
                    if 'subrow_grid_location' in plot:
                        subrow_grid_x = \
                            int(plot['subrow_grid_location']['x']['#text'])
                        subrow_grid_y = \
                            int(plot['subrow_grid_location']['y']['#text'])
                    else:
                        subrow_grid_x = 'unknown'
                        subrow_grid_y = 'unknown'
                    if 'row_number' in plot:
                        row_number = plot['row_number']
                    else:
                        row_number = 'unknown'
                    if 'range_number' in plot:
                        range_number = plot['range_number']
                    else:
                        range_number = 'unknown'
                    img_abspath = os.path.join(self.root_dir, filename)
                    orig_width, orig_height = \
                        get_image_size.get_image_size(img_abspath)
                    with torch.no_grad():
                        orig_height = torch.tensor(
                            orig_height, dtype=torch.get_default_dtype())
                        orig_width = torch.tensor(
                            orig_width, dtype=torch.get_default_dtype())
                    dictt[filename] = {'filename': filename,
                                           'plot_number': plot_number,
                                           'subrow_grid_location_x': subrow_grid_x,
                                           'subrow_grid_location_y': subrow_grid_y,
                                           'row_number': row_number,
                                           'range_number': range_number,
                                           'orig_width': orig_width,
                                           'orig_height': orig_height}
                    if self.there_is_gt:
                        count = int(plot['plant_count'])
                        locations = []
                        for plant in plot['plants']['plant']:
                            for y in plant['location']['y']:
                                if y['@units'] == 'pixels' and \
                                        y['@wrt'] == 'plot':
                                    y = float(y['#text'])
                                    break
                            for x in plant['location']['x']:
                                if x['@units'] == 'pixels' and \
                                        x['@wrt'] == 'plot':
                                    x = float(x['#text'])
                                    break
                            locations.append([y, x])
                        dictt[filename]['count'] = count
                        dictt[filename]['locations'] = locations

            # Use an Ordered Dictionary to allow random access
            dictt = OrderedDict(dictt.items())
            self.dict_list = list(dictt.items())

            # Make dataset smaller
            new_dataset_length = min(len(dictt), max_dataset_size)
            dictt = {key: elem_dict
                         for key, elem_dict in
                         self.dict_list[:new_dataset_length]}
            self.dict_list = list(dictt.items())

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self, idx):
        """Get one element of the dataset.
        Returns a tuple. The first element is the image.
        The second element is a dictionary containing the labels of that image.
        The dictionary may not contain the location and count if the original
         XML did not include it.

        :param idx: Index of the image in the dataset to get.
        """

        filename, dictionary = self.dict_list[idx]
        img_abspath = os.path.join(self.root_dir, filename)

        if self.there_is_gt:
            # list --> Tensors
            with torch.no_grad():
                dictionary['locations'] = torch.tensor(
                    dictionary['locations'],
                    dtype=torch.get_default_dtype())
                dictionary['count'] = torch.tensor(
                    dictionary['count'],
                    dtype=torch.get_default_dtype())
        # else:
        #     filename = self.listfiles[idx]
        #     img_abspath = os.path.join(self.root_dir, filename)
        #     orig_width, orig_height = \
        #         get_image_size.get_image_size(img_abspath)
        #     with torch.no_grad():
        #         orig_height = torch.tensor(
        #             orig_height, dtype=torch.get_default_dtype())
        #         orig_width = torch.tensor(
        #             orig_width, dtype=torch.get_default_dtype())
        #     dictionary = {'filename': self.listfiles[idx],
        #                   'orig_width': orig_width,
        #                   'orig_height': orig_height}

        img = Image.open(img_abspath)

        img_transformed = img
        transformed_dictionary = dictionary

        # Apply all transformations provided
        if self.transforms is not None:
            for transform in self.transforms.transforms:
                if hasattr(transform, 'modifies_label'):
                    img_transformed, transformed_dictionary = \
                        transform(img_transformed, transformed_dictionary)
                else:
                    img_transformed = transform(img_transformed)

        # Prevents crash when making a batch out of an empty tensor
        if self.there_is_gt and dictionary['count'].item() == 0:
            with torch.no_grad():
                dictionary['locations'] = torch.tensor([-1, -1],
                                                       dtype=torch.get_default_dtype())

        return (img_transformed, transformed_dictionary)


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
