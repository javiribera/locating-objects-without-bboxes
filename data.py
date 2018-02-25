import os
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
    def __init__(self, root_dir, transform=None, max_dataset_size=float('inf')):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            max_dataset_size: If the dataset is bigger than this integer,
                              ignore additional samples.
        """

        # Get groundtruth from CSV file
        csv_filename = None
        for filename in os.listdir(root_dir):
            if filename.endswith('.csv'):
                csv_filename = filename
                break
        if csv_filename is None:
            raise ValueError(
                'The root directory %s does not have a CSV file with groundtruth' % root_dir)
        self.csv_df = pd.read_csv(os.path.join(root_dir, csv_filename))

        # Make the dataset smaller
        self.csv_df = self.csv_df[0:min(len(self.csv_df), max_dataset_size)]

        self.root_dir = root_dir
        self.transforms = transform

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv_df.ix[idx, 0])
        img = Image.open(img_path)
        dictionary = dict(self.csv_df.ix[idx])

        # str -> lists
        dictionary['plant_locations'] = eval(dictionary['plant_locations'])
        dictionary['plant_locations'] = [
            list(loc) for loc in dictionary['plant_locations']]

        # list --> Tensors
        dictionary['plant_locations'] = torch.FloatTensor(
            dictionary['plant_locations'])
        dictionary['plant_count'] = torch.FloatTensor(
            [dictionary['plant_count']])

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
        if dictionary['plant_count'][0] == 0:
            dictionary['plant_locations'] = torch.FloatTensor([-1, -1])

        return (img_transformed, transformed_dictionary)


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
            for l, loc in enumerate(dictionary['plant_locations']):
                dictionary['plant_locations'][l][1] = (width - 1) - loc[1]

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
            for l, loc in enumerate(dictionary['plant_locations']):
                dictionary['plant_locations'][l][0] = (height - 1) - loc[0]

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
