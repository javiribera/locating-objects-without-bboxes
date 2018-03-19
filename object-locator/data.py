import os
# 
import inspect
import random

from PIL import Image
import skimage
import pandas as pd
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


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
        It can containas many columns as wanted, i.e, filename, count...

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
            width = img.size[1]
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
            height = img.size[0]
            for l, loc in enumerate(dictionary['locations']):
                dictionary['locations'][l][0] = (height - 1) - loc[0]

        return transformed_img, transformed_dictionary


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
