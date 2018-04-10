import os
import random
from collections import OrderedDict

from PIL import Image
import skimage
import pandas as pd
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import xmltodict
from parse import parse
from . import get_image_size

IMG_EXTENSIONS = ['.png', '.jpeg', '.jpg', '.tiff']

class CSVDataset(data.Dataset):
    def __init__(self,
                 directory,
                 transforms=None,
                 max_dataset_size=float('inf'),
                 tensortype=torch.FloatTensor):
        """CSVDataset.
        The sample images of this dataset must be all inside one directory.
        Inside the same directory, there must be one CSV file.
        This file must contain one row per image.
        It can contain as many columns as wanted, i.e, filename, count...

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.
        :param max_dataset_size: Only use the first N images in the directory.
        :param tensortype: The data and labels will be returned in this type format.
        """

        self.root_dir = directory
        self.transforms = transforms

        # Type of tensor the output will be
        self.tensortype = tensortype

        # Get groundtruth from CSV file
        listfiles = os.listdir(directory)
        csv_filename = None
        for filename in listfiles:
            if filename.endswith('.csv'):
                csv_filename = filename
                break

        self.there_is_gt = csv_filename is not None

        # CSV does not exist (no GT available)
        if not self.there_is_gt:
            print('W: The dataset directory %s does not contain a CSV file with groundtruth. \n'
                  '   Metrics will not be evaluated. Only estimations will be returned.' % directory)
            self.csv_df = None
            self.listfiles = listfiles

            # Make dataset smaller
            self.listfiles = self.listfiles[0:min(
                len(self.listfiles), max_dataset_size)]

        # CSV does exist (GT is available)
        else:
            self.csv_df = pd.read_csv(os.path.join(directory, csv_filename))

            # Make dataset smaller
            self.csv_df = self.csv_df[0:min(
                len(self.csv_df), max_dataset_size)]

    def __len__(self):
        if self.there_is_gt:
            return len(self.csv_df)
        else:
            return len(self.listfiles)

    def __getitem__(self, idx):
        """Get one element of the dataset.
        Returns a tuple. The first element is the image.
        The second element is a dictionary where the keys are the columns of the CSV.
        If the CSV did not exist in the dataset directory,
         the dictionary will only contain the filename of the image.

        :param idx: Index of the image in the dataset to get.
        """

        if self.there_is_gt:
            img_abspath = os.path.join(self.root_dir, self.csv_df.ix[idx, 0])
            dictionary = dict(self.csv_df.ix[idx])
        else:
            img_abspath = os.path.join(self.root_dir, self.listfiles[idx])
            dictionary = {'filename': self.listfiles[idx]}

        img = Image.open(img_abspath)

        # str -> lists
        dictionary['locations'] = eval(dictionary['locations'])
        dictionary['locations'] = [
            list(loc) for loc in dictionary['locations']]

        # list --> Tensors
        dictionary['locations'] = self.tensortype(
            dictionary['locations'])
        dictionary['count'] = self.tensortype(
            [dictionary['count']])

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
        if dictionary['count'][0] == 0:
            dictionary['locations'] = self.tensortype([-1, -1])

        return (img_transformed, transformed_dictionary)


def csv_collator(samples):
    """Merge a list of samples to form a batch.
    The batch is a 2-element tuple, being the first element
     the BxHxW tensor and the second element a list of dictionaries.

    :param samples: List of samples returned by CSVDataset as (img, dict) tuples.
    """

    imgs = []
    dicts = []

    for sample in samples:
        img = sample[0]
        dictt = sample[1]

        # # We cannot deal with images with 0 plants (WHD is not defined)
        # if dictt['count'][0] == 0:
        #     continue

        imgs.append(img)
        dicts.append(dictt)

    data = torch.stack(imgs)

    return data, dicts


class RandomHorizontalFlipImageAndLabel(object):
    """ Horizontally flip a numpy array image and the GT with probability p """

    def __init__(self, p):
        self.modifies_label = True
        self.p = p

    def __call__(self, img, dictionary):
        transformed_img = img
        transformed_dictionary = dictionary

        if random.random() < self.p:
            transformed_img = hflip(img)
            width = img.size[0]
            for l, loc in enumerate(dictionary['locations']):
                dictionary['locations'][l][1] = (width - 1) - loc[1]

        return transformed_img, transformed_dictionary


class RandomVerticalFlipImageAndLabel(object):
    """ Vertically flip a numpy array image and the GT with probability p """

    def __init__(self, p):
        self.modifies_label = True
        self.p = p

    def __call__(self, img, dictionary):
        transformed_img = img
        transformed_dictionary = dictionary

        if random.random() < self.p:
            transformed_img = vflip(img)
            height = img.size[1]
            for l, loc in enumerate(dictionary['locations']):
                dictionary['locations'][l][0] = (height - 1) - loc[0]

        return transformed_img, transformed_dictionary


class ScaleImageAndLabel(transforms.Scale):
    """
    Scale a PIL Image and the GT to a given size.
     If there is no GT, then only scale the PIL Image.

    Args:
        size: Desired output size (h, w).
        interpolation (int, optional): Desired interpolation.
                                       Default is ``PIL.Image.BILINEAR``.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.modifies_label = True
        self.size = size
        super(ScaleImageAndLabel, self).__init__(size, interpolation)

    def __call__(self, img, dictionary):

        old_width, old_height = img.size
        scale_h = self.size[0]/old_height
        scale_w = self.size[1]/old_width

        # Scale image to new size
        img = super(ScaleImageAndLabel, self).__call__(img)

        # Scale GT
        if 'locations' in dictionary:
            dictionary['locations'] *= torch.FloatTensor([scale_h, scale_w])
            dictionary['locations'] = torch.round(dictionary['locations'])
            ys = torch.clamp(dictionary['locations'][:, 0], 0, self.size[0])
            xs = torch.clamp(dictionary['locations'][:, 1], 0, self.size[1])
            dictionary['locations'] = torch.cat((ys.view(-1, 1),
                                                 xs.view(-1, 1)),
                                                1)

        # Indicate new size in dictionary
        dictionary['resized_height'] = self.size[0]
        dictionary['resized_width'] = self.size[1]

        return img, dictionary


def hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """Vertically flip the given PIL Image.
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def _is_pil_image(img):
    return isinstance(img, Image.Image)


class XMLDataset(data.Dataset):
    def __init__(self,
                 directory,
                 transforms=None,
                 max_dataset_size=float('inf'),
                 ignore_gt=False,
                 tensortype=torch.FloatTensor):
        """XMLDataset.
        The sample images of this dataset must be all inside one directory.
         Inside the same directory, there must be one XML file as described by
         https://communityhub.purdue.edu/groups/phenosorg/wiki/APIspecs
         (minimum xml api version is v.0.2.0)
         if the xml file is not present, then samples do not contain plant location
         or plant count information.
        :param directory: Directory with all the images and the XML file.
        :param transform: Transform to be applied to each image.
        :param max_dataset_size: Only use the first N images in the directory.
        :param ignore_gt: Ignore the GT in the XML file,
                          i.e, provide samples without plant locations or counts.
        :param tensortype: The data and labels will be returned in this type format.
        """

        self.root_dir = directory
        self.transforms = transforms

        # Type of tensor the output will be
        self.tensortype = tensortype

        # Get list of files in the dataset directory,
        # and the filename of the XML
        listfiles = os.listdir(directory)
        xml_filename = None
        for filename in listfiles:
            if filename.endswith('.xml'):
                xml_filename = filename
                break

        if xml_filename is None:
            print('W: The dataset directory %s does not contain '
                  'a XML file with groundtruth. Metrics will not be evaluated.'
                  'Only estimations will be returned.' % directory)

        self.there_is_gt = (xml_filename is not None) and (not ignore_gt)

        # Ignore files that are not images
        listfiles = [f for f in listfiles 
                     if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)] 

        # XML does not exist (no GT available)
        if not self.there_is_gt:
            self.dict = None
            self.listfiles = listfiles

            # Make dataset smaller
            self.listfiles = self.listfiles[0:min(len(self.listfiles),
                                                  max_dataset_size)]

        # XML does exist (GT is available)
        else:

            # Read all XML as a string
            with open(os.path.join(directory, xml_filename), 'r') as fd:
                xml_str = fd.read()

            # Convert to dictionary
            # (some elements we expect to have multiple repetitions, so put them in a list)
            xml_dict = xmltodict.parse(
                xml_str, force_list=['field', 'panel', 'plot', 'plant'])

            # Check API version number
            try:
                api_version = xml_dict['fields']['@apiversion']
            except:
                # An unknown version number means it's the very first one
                # when we did not have api version numbers
                api_version = '0.1.0'
            major_version, minor_version, addendum_version = \
                parse('{}.{}.{}', api_version)
            major_version = int(major_version)
            minor_version = int(minor_version)
            addendum_version = int(addendum_version)
            if not(major_version == 0 and
                   minor_version == 2 and
                   addendum_version >= 1):
                raise ValueError('An XML with API v0.2.1 is required.')

            # Create the dictionary with the entire dataset
            self.dict = {}
            for field in xml_dict['fields']['field']:
                for panel in field['panels']['panel']:
                    for plot in panel['plots']['plot']:
                        filename = plot['orthophoto_chop_filename']
                        count = int(plot['plant_count'])
                        if 'plot_number' in plot:
                            plot_number = plot['plot_number']
                        else:
                            plot_number = 'unknown'
                        if 'cigar_grid_location_yx' in plot:
                            cigar = plot['cigar_grid_location_yx']
                        else:
                            plot_number = 'unknown'
                        locations = []
                        for plant in plot['plants']['plant']:
                            locations.append(eval(plant['location_wrt_plot']))
                        img_abspath = os.path.join(self.root_dir, filename)
                        orig_width, orig_height = \
                            get_image_size.get_image_size(img_abspath)
                        self.dict[filename] = {'filename': filename,
                                               'count': count,
                                               'locations': locations,
                                               'orig_width': orig_width,
                                               'orig_height': orig_height}

            # Use an Ordered Dictionary to allow random access
            self.dict = OrderedDict(self.dict.items())
            self.dict_list = list(self.dict.items())

            # Make dataset smaller
            new_dataset_length = min(len(self.dict), max_dataset_size)
            self.dict = {key: elem_dict
                         for key, elem_dict in
                         self.dict_list[:new_dataset_length]}
            self.dict_list = list(self.dict.items())

    def __len__(self):
        if self.there_is_gt:
            return len(self.dict)
        else:
            return len(self.listfiles)

    def __getitem__(self, idx):
        """Get one element of the dataset.
        Returns a tuple. The first element is the image.
        The second element is a dictionary containing the labels of that image.
        If the XML did not exist in the dataset directory,
         the dictionary will only contain the filename and size of the image.

        :param idx: Index of the image in the dataset to get.
        """

        if self.there_is_gt:
            filename, dictionary = self.dict_list[idx]
            img_abspath = os.path.join(self.root_dir, filename)

            # list --> Tensors
            dictionary['locations'] = self.tensortype(
                dictionary['locations'])
            dictionary['count'] = self.tensortype(
                [dictionary['count']])
        else:
            filename = self.listfiles[idx]
            img_abspath = os.path.join(self.root_dir, filename)
            orig_width, orig_height = \
                get_image_size.get_image_size(img_abspath)
            dictionary = {'filename': self.listfiles[idx],
                          'orig_width': orig_width,
                          'orig_height': orig_height}

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
        if self.there_is_gt and dictionary['count'][0] == 0:
            dictionary['locations'] = self.tensortype([-1, -1])

        return (img_transformed, transformed_dictionary)
