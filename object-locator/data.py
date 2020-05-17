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

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
from ballpark import ballpark

from . import get_image_size

IMG_EXTENSIONS = ['.png', '.jpeg', '.jpg', '.tiff', '.tif']

torch.set_default_dtype(torch.float32)


def build_dataset(directory,
                  transforms=None,
                  max_dataset_size=float('inf'),
                  ignore_gt=False,
                  seed=0):
    """
    Build a dataset from a directory.
     Depending if the directory contains a CSV or an XML dataset,
     it builds an XMLDataset or a CSVDataset, which are subclasses
     of torch.utils.data.Dataset.
    :param directory: Directory with all the images and the CSV file.
    :param transform: Transform to be applied to each image.
    :param max_dataset_size: Only use the first N images in the directory.
    :param ignore_gt: Ignore the GT of the dataset,
                      i.e, provide samples without locations or counts.
    :param seed: Random seed.
    :return: An XMLDataset or CSVDataset instance.
    """
    if any(fn.endswith('.csv') for fn in os.listdir(directory)) \
            or ignore_gt:
        dset = CSVDataset(directory=directory,
                          transforms=transforms,
                          max_dataset_size=max_dataset_size,
                          ignore_gt=ignore_gt,
                          seed=seed)
    else:
        from . import data_plant_stuff
        dset = data_plant_stuff.\
            XMLDataset(directory=directory,
                       transforms=transforms,
                       max_dataset_size=max_dataset_size,
                       ignore_gt=ignore_gt,
                       seed=seed)

    return dset
    

def get_train_val_loaders(train_dir,
                          collate_fn,
                          height,
                          width,
                          no_data_augmentation=False,
                          max_trainset_size=np.infty,
                          seed=0,
                          batch_size=1,
                          drop_last_batch=False,
                          shuffle=True,
                          num_workers=0,
                          val_dir=None,
                          max_valset_size=np.infty):
    """
    Create a training loader and a validation set.
    If the validation directory is 'auto',
    20% of the dataset is used for validation.

    :param train_dir: Directory with all the training images and the CSV file.
    :param train_transforms: Transform to be applied to each training image.
    :param max_trainset_size: Only use first N images for training.
    :param collate_fn: Function to assemble samples into batches.
    :param height: Resize the images to this height.
    :param width: Resize the images to this width.
    :param no_data_augmentation: Do not perform data augmentation.
    :param seed: Random seed.
    :param batch_size: Number of samples in a batch, for training.
    :param drop_last_batch: Drop the last incomplete batch during training
    :param shuffle: Randomly shuffle the dataset before each epoch.
    :param num_workers: Number of subprocesses dedicated for data loading.
    :param val_dir: Directory with all the training images and the CSV file.
    :param max_valset_size: Only use first N images for validation.
    """

    # Data augmentation for training
    training_transforms = []
    if not no_data_augmentation:
        training_transforms += [RandomHorizontalFlipImageAndLabel(p=0.5,
                                                                  seed=seed)]
        training_transforms += [RandomVerticalFlipImageAndLabel(p=0.5,
                                                                seed=seed)]
    training_transforms += [ScaleImageAndLabel(size=(height, width))]
    training_transforms += [torchvision.transforms.ToTensor()]
    training_transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))]
    training_transforms = torchvision.transforms.Compose(training_transforms)

    # Data augmentation for validation
    validation_transforms = torchvision.transforms.Compose([
                                ScaleImageAndLabel(size=(height, width)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.\
                                    Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)),
                            ])

    # Training dataset
    trainset = build_dataset(directory=train_dir,
                             transforms=training_transforms,
                             max_dataset_size=max_trainset_size,
                             seed=seed)

    # Validation dataset
    if val_dir is not None:
        if val_dir == 'auto':
            # Create a dataset just as in training
            valset = build_dataset(directory=train_dir,
                                   transforms=validation_transforms,
                                   max_dataset_size=max_trainset_size,
                                   seed=seed)

            # Split 80% for training, 20% for validation
            n_imgs_for_training = int(round(0.8*len(trainset)))
            if isinstance(trainset, CSVDataset):
                if trainset.there_is_gt:
                    trainset.csv_df = \
                        trainset.csv_df[:n_imgs_for_training]
                    valset.csv_df = \
                        valset.csv_df[n_imgs_for_training:].reset_index()
                else:
                    trainset.listfiles = \
                        trainset.listfiles[:n_imgs_for_training]
                    valset.listfiles = \
                        valset.listfiles[n_imgs_for_training:]
            else: # isinstance(trainset, XMLDataset):
                trainset.dict_list = trainset.dict_list[:n_imgs_for_training]
                valset.dict_list = valset.dict_list[n_imgs_for_training:]

        else:
            valset = build_dataset(val_dir,
                                   transforms=validation_transforms,
                                   max_dataset_size=max_valset_size,
                                   seed=seed)
            valset_loader = torch.utils.data.DataLoader(valset,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       collate_fn=csv_collator)
    else:
        valset, valset_loader = None, None

    print(f'# images for training: '
          f'{ballpark(len(trainset))}')
    if valset is not None:
        print(f'# images for validation: '
              f'{ballpark(len(valset))}')
    else:
        print('W: no validation set was selected!')

    # Build data loaders from the datasets
    trainset_loader = torch.utils.data.DataLoader(trainset,
                                 batch_size=batch_size,
                                 drop_last=drop_last_batch,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 collate_fn=csv_collator)
    if valset is not None:
        valset_loader = torch.utils.data.DataLoader(valset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   collate_fn=csv_collator)

    return trainset_loader, valset_loader


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self,
                 directory,
                 transforms=None,
                 max_dataset_size=float('inf'),
                 ignore_gt=False,
                 seed=0):
        """CSVDataset.
        The sample images of this dataset must be all inside one directory.
        Inside the same directory, there must be one CSV file.
        This file must contain one row per image.
        It can contain as many columns as wanted, i.e, filename, count...

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.
        :param max_dataset_size: Only use the first N images in the directory.
        :param ignore_gt: Ignore the GT of the dataset,
                          i.e, provide samples without locations or counts.
        :param seed: Random seed.
        """

        self.root_dir = directory
        self.transforms = transforms

        # Get groundtruth from CSV file
        listfiles = os.listdir(directory)
        csv_filename = None
        for filename in listfiles:
            if filename.endswith('.csv'):
                csv_filename = filename
                break

        # Ignore files that are not images
        listfiles = [f for f in listfiles
                     if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]

        # Shuffle list of files
        np.random.seed(seed)
        random.shuffle(listfiles)

        if len(listfiles) == 0:
            raise ValueError(f"There are no images in '{directory}'")

        self.there_is_gt = (csv_filename is not None) and (not ignore_gt)

        # CSV does not exist (no GT available)
        if not self.there_is_gt:
            print('W: The dataset directory %s does not contain a CSV file with groundtruth. \n'
                  '   Metrics will not be evaluated. Only estimations will be returned.' % directory)
            self.csv_df = None
            self.listfiles = listfiles

            # Make dataset smaller
            self.listfiles = self.listfiles[0:min(len(self.listfiles),
                                                  max_dataset_size)]

        # CSV does exist (GT is available)
        else:
            self.csv_df = pd.read_csv(os.path.join(directory, csv_filename))

            # Shuffle CSV dataframe
            self.csv_df = self.csv_df.sample(frac=1).reset_index(drop=True)

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
            img_abspath = os.path.join(self.root_dir, self.csv_df.ix[idx].filename)
            dictionary = dict(self.csv_df.ix[idx])
        else:
            img_abspath = os.path.join(self.root_dir, self.listfiles[idx])
            dictionary = {'filename': self.listfiles[idx]}

        img = Image.open(img_abspath)

        if self.there_is_gt:
            # str -> lists
            dictionary['locations'] = eval(dictionary['locations'])
            dictionary['locations'] = [
                list(loc) for loc in dictionary['locations']]

            # list --> Tensors
            with torch.no_grad():
                dictionary['locations'] = torch.tensor(
                    dictionary['locations'], dtype=torch.get_default_dtype())
                dictionary['count'] = torch.tensor(
                    [dictionary['count']], dtype=torch.get_default_dtype())

        # Record original size
        orig_width, orig_height = get_image_size.get_image_size(img_abspath)
        with torch.no_grad():
            orig_height = torch.tensor(orig_height,
                                       dtype=torch.get_default_dtype())
            orig_width = torch.tensor(orig_width,
                                      dtype=torch.get_default_dtype())
        dictionary['orig_width'] = orig_width
        dictionary['orig_height'] = orig_height

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
        if self.there_is_gt:
            if dictionary['count'][0] == 0:
                with torch.no_grad():
                    dictionary['locations'] = torch.tensor([-1, -1],
                                                           dtype=torch.get_default_dtype())

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

        # # We cannot deal with images with 0 objects (WHD is not defined)
        # if dictt['count'][0] == 0:
        #     continue

        imgs.append(img)
        dicts.append(dictt)

    data = torch.stack(imgs)

    return data, dicts


class RandomHorizontalFlipImageAndLabel(object):
    """ Horizontally flip a numpy array image and the GT with probability p """

    def __init__(self, p, seed=0):
        self.modifies_label = True
        self.p = p
        np.random.seed(seed)

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

    def __init__(self, p, seed=0):
        self.modifies_label = True
        self.p = p
        np.random.seed(seed)

    def __call__(self, img, dictionary):
        transformed_img = img
        transformed_dictionary = dictionary

        if random.random() < self.p:
            transformed_img = vflip(img)
            height = img.size[1]
            for l, loc in enumerate(dictionary['locations']):
                dictionary['locations'][l][0] = (height - 1) - loc[0]

        return transformed_img, transformed_dictionary


class ScaleImageAndLabel(torchvision.transforms.Resize):
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
        if 'locations' in dictionary and len(dictionary['locations']) > 0:
            # print(dictionary['locations'].type())
            # print(torch.tensor([scale_h, scale_w]).type())
            with torch.no_grad():
                dictionary['locations'] *= torch.tensor([scale_h, scale_w])
                dictionary['locations'] = torch.round(dictionary['locations'])
                ys = torch.clamp(
                    dictionary['locations'][:, 0], 0, self.size[0])
                xs = torch.clamp(
                    dictionary['locations'][:, 1], 0, self.size[1])
                dictionary['locations'] = torch.cat((ys.view(-1, 1),
                                                     xs.view(-1, 1)),
                                                    1)

        # Indicate new size in dictionary
        with torch.no_grad():
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


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
