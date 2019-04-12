# Copyright &copyright 2018 The Board of Trustees of Purdue University.
# All rights reserved.
# 
# This source code is not to be distributed or modified
# without the written permission of Edward J. Delp at Purdue University
# Contact information: ace@ecn.purdue.edu
# =====================================================================

from __future__ import print_function

import os
import sys

import cv2
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torch.utils import data

from .data import CSVDataset
from .data import csv_collator
from . import argparser
from . import utils


# Parse command line arguments
args = argparser.parse_command_args('testing')

# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Data loading code
try:
    testset = CSVDataset(args.dataset,
                         transforms=transforms.Compose([
                             transforms.ToTensor(),
                         ]),
                         max_dataset_size=args.max_testset_size)
except ValueError as e:
    print(f'E: {e}')
    exit(-1)
dataset_loader = data.DataLoader(testset,
                                 batch_size=1,
                                 num_workers=args.nThreads,
                                 collate_fn=csv_collator)

os.makedirs(os.path.join(args.out_dir), exist_ok=True)

for img, dictionary in tqdm(dataset_loader):

    # Move to device
    img = img.to(device_cpu)

    # One image at a time (BS=1)
    img = img[0]
    dictionary = dictionary[0]

    # Tensor -> float & numpy
    target_locs = dictionary['locations'].to(device_cpu).numpy().reshape(-1, 2)
    img = img.to(device_cpu).numpy()

    img *= 255

    # Paint circles on top of image
    img_with_x = utils.paint_circles(img=img,
                                     points=target_locs,
                                     color='white')
    img_with_x = np.moveaxis(img_with_x, 0, 2)
    img_with_x = img_with_x[:, :, ::-1]

    cv2.imwrite(os.path.join(args.out_dir, dictionary['filename']),
                img_with_x)

