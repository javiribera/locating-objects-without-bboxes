from __future__ import print_function

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


import math
import os
from itertools import chain
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler
import matplotlib
matplotlib.use('Agg')
import skimage.transform
from peterpy import peter
from ballpark import ballpark
from matplotlib import pyplot as plt

from . import losses
from .models import unet_model
from .data import CSVDataset
from .data import csv_collator
from .data import RandomHorizontalFlipImageAndLabel
from .data import RandomVerticalFlipImageAndLabel
from .data import ScaleImageAndLabel
from . import argparser


# Parse command line arguments
args = argparser.parse_command_args('training')

# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') if args.cuda else device_cpu

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Data loading code
training_transforms = []
if not args.no_data_augm:
    training_transforms += [RandomHorizontalFlipImageAndLabel(p=0.5)]
    training_transforms += [RandomVerticalFlipImageAndLabel(p=0.5)]
training_transforms += [ScaleImageAndLabel(size=(args.height, args.width))]
training_transforms += [transforms.ToTensor()]
training_transforms += [transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))]
trainset = CSVDataset(args.train_dir,
                      transforms=transforms.Compose(training_transforms),
                      max_dataset_size=args.max_trainset_size)
trainset_loader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             drop_last=args.drop_last_batch,
                             shuffle=True,
                             num_workers=args.nThreads,
                             collate_fn=csv_collator)

# Model
with peter('Building network'):
    model = unet_model.UNet(3, 1,
                            height=args.height,
                            width=args.width,
                            known_n_points=args.n_points)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" with {ballpark(num_params)} trainable parameters. ", end='')
model = nn.DataParallel(model)
model.to(device)


# Loss function
loss_regress = nn.SmoothL1Loss()
loss_loc = losses.WeightedHausdorffDistance(resized_height=args.height,
                                            resized_width=args.width,
                                            p=args.p,
                                            return_2_terms=True,
                                            device=device)
l1_loss = nn.L1Loss(size_average=False)
mse_loss = nn.MSELoss(reduce=False)

optimizer = optim.SGD(model.parameters(),
                      lr=999) # will be set later

 
def find_lr(init_value = 1e-6, final_value=1e-3, beta = 0.7):
    num = len(trainset_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for imgs, dicts in tqdm(trainset_loader):
        batch_num += 1

        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        imgs = Variable(imgs)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dicts]
        target_counts = [dictt['count'].to(device)
                         for dictt in dicts]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dicts]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dicts]

        # Lists -> Tensor batches
        target_counts = torch.stack(target_counts)
        target_orig_heights = torch.stack(target_orig_heights)
        target_orig_widths = torch.stack(target_orig_widths)
        target_orig_sizes = torch.stack((target_orig_heights,
                                         target_orig_widths)).transpose(0, 1)
        # As before, get the loss for this mini-batch of inputs/outputs
        optimizer.zero_grad()
        est_maps, est_counts = model.forward(imgs)
        term1, term2 = loss_loc.forward(est_maps,
                                        target_locations,
                                        target_orig_sizes)
        target_counts = target_counts.view(-1)
        est_counts = est_counts.view(-1)
        target_counts = target_counts.view(-1)
        term3 = loss_regress.forward(est_counts, target_counts)
        term3 *= args.lambdaa
        loss = term1 + term2 + term3

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Stop if the loss is exploding
        if (batch_num > 1 and smoothed_loss > 4 * best_loss):
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Do the SGD step
        loss.backward()
        optimizer.step()

        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

logs, losses = find_lr()
plt.plot(logs, losses)
plt.savefig('/data/jprat/plot_beta0.7.png')


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
