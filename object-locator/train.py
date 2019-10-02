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
import cv2
import os
import sys
import time
import shutil
from itertools import chain
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import skimage.transform
from peterpy import peter
from ballpark import ballpark

from . import losses
from .models import unet_model
from .metrics import Judge
from . import logger
from . import argparser
from . import utils
from . import data
from .data import csv_collator
from .data import RandomHorizontalFlipImageAndLabel
from .data import RandomVerticalFlipImageAndLabel
from .data import ScaleImageAndLabel


# Parse command line arguments
args = argparser.parse_command_args('training')

# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') if args.cuda else device_cpu

# Create directory for checkpoint to be saved
if args.save:
    os.makedirs(os.path.split(args.save)[0], exist_ok=True)

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Visdom setup
log = logger.Logger(server=args.visdom_server,
                    port=args.visdom_port,
                    env_name=args.visdom_env)


# Create data loaders (return data in batches)
trainset_loader, valset_loader = \
    data.get_train_val_loaders(train_dir=args.train_dir,
                               max_trainset_size=args.max_trainset_size,
                               collate_fn=csv_collator,
                               height=args.height,
                               width=args.width,
                               seed=args.seed,
                               batch_size=args.batch_size,
                               drop_last_batch=args.drop_last_batch,
                               num_workers=args.nThreads,
                               val_dir=args.val_dir,
                               max_valset_size=args.max_valset_size)

# Model
with peter('Building network'):
    model = unet_model.UNet(3, 1,
                            height=args.height,
                            width=args.width,
                            known_n_points=args.n_points,
                            device=device,
                            ultrasmall=args.ultrasmallnet)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" with {ballpark(num_params)} trainable parameters. ", end='')
model = nn.DataParallel(model)
model.to(device)

# Loss functions
loss_regress = nn.SmoothL1Loss()
loss_loc = losses.WeightedHausdorffDistance(resized_height=args.height,
                                            resized_width=args.width,
                                            p=args.p,
                                            return_2_terms=True,
                                            device=device)

# Optimization strategy
if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           amsgrad=True)

start_epoch = 0
lowest_mahd = np.infty

# Restore saved checkpoint (model weights + epoch + optimizer state)
if args.resume:
    with peter('Loading checkpoint'):
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            try:
                lowest_mahd = checkpoint['mahd']
            except KeyError:
                lowest_mahd = np.infty
                print('W: Loaded checkpoint has not been validated. ', end='')
            model.load_state_dict(checkpoint['model'])
            if not args.replace_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"\n\__ loaded checkpoint '{args.resume}'"
                  f"(now on epoch {checkpoint['epoch']})")
        else:
            print(f"\n\__ E: no checkpoint found at '{args.resume}'")
            exit(-1)

running_avg = utils.RunningAverage(len(trainset_loader))

normalzr = utils.Normalizer(args.height, args.width)

# Time at the last evaluation
tic_train = -np.infty
tic_val = -np.infty

epoch = start_epoch
it_num = 0
while epoch < args.epochs:

    loss_avg_this_epoch = 0
    iter_train = tqdm(trainset_loader,
                      desc=f'Epoch {epoch} ({len(trainset_loader.dataset)} images)')

    # === TRAIN ===

    # Set the module in training mode
    model.train()

    for batch_idx, (imgs, dictionaries) in enumerate(iter_train):

        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_counts = [dictt['count'].to(device)
                         for dictt in dictionaries]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dictionaries]

        # Lists -> Tensor batches
        target_counts = torch.stack(target_counts)
        target_orig_heights = torch.stack(target_orig_heights)
        target_orig_widths = torch.stack(target_orig_widths)
        target_orig_sizes = torch.stack((target_orig_heights,
                                         target_orig_widths)).transpose(0, 1)

        # One training step
        optimizer.zero_grad()
        est_maps, est_counts = model.forward(imgs)
        term1, term2 = loss_loc.forward(est_maps,
                                        target_locations,
                                        target_orig_sizes)
        est_counts = est_counts.view(-1)
        target_counts = target_counts.view(-1)
        term3 = loss_regress.forward(est_counts, target_counts)
        term3 *= args.lambdaa
        loss = term1 + term2 + term3
        loss.backward()
        optimizer.step()

        # Update progress bar
        running_avg.put(loss.item())
        iter_train.set_postfix(running_avg=f'{round(running_avg.avg/3, 1)}')

        # Log training error
        if time.time() > tic_train + args.log_interval:
            tic_train = time.time()

            # Log training losses
            log.train_losses(terms=[term1, term2, term3, loss / 3, running_avg.avg / 3],
                             iteration_number=epoch +
                             batch_idx/len(trainset_loader),
                             terms_legends=['Term1',
                                            'Term2',
                                            'Term3*%s' % args.lambdaa,
                                            'Sum/3',
                                            'Sum/3 runn avg'])

            # Resize images to original size
            orig_shape = target_orig_sizes[0].data.to(device_cpu).numpy().tolist()
            orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().to(device_cpu).numpy().transpose((1, 2, 0)),
                                                           output_shape=orig_shape,
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(est_maps[0].data.unsqueeze(0).to(device_cpu).numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1)).squeeze(0)

            # Overlay output on heatmap
            orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                map=est_map_origsize).\
                astype(np.float32)

            # Send heatmap with circles at the labeled points to Visdom
            target_locs_np = target_locations[0].\
                to(device_cpu).numpy().reshape(-1, 2)
            target_orig_size_np = target_orig_sizes[0].\
                to(device_cpu).numpy().reshape(2)
            target_locs_wrt_orig = normalzr.unnormalize(target_locs_np,
                                                        orig_img_size=target_orig_size_np)
            img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                               points=target_locs_wrt_orig,
                                               color='white')
            log.image(imgs=[img_with_x],
                      titles=['(Training) Image w/ output heatmap and labeled points'],
                      window_ids=[1])

            # # Read image with GT dots from disk
            # gt_img_numpy = skimage.io.imread(
            #     os.path.join('/home/jprat/projects/phenosorg/data/plant_counts_dots/20160613_F54_training_256x256_white_bigdots',
            #                  dictionary['filename'][0]))
            # # dots_img_tensor = torch.from_numpy(gt_img_numpy).permute(
            # # 2, 0, 1)[0, :, :].type(torch.FloatTensor) / 255
            # # Send GT image to Visdom
            # viz.image(np.moveaxis(gt_img_numpy, 2, 0),
            #           opts=dict(title='(Training) Ground Truth'),
            #           win=3)

        it_num += 1

    # Never do validation?
    if not args.val_dir or \
            not valset_loader or \
            len(valset_loader) == 0 or \
            args.val_freq == 0:

        # Time to save checkpoint?
        if args.save and (epoch + 1) % args.val_freq == 0:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'n_points': args.n_points,
                        }, args.save)
        epoch += 1
        continue

    # Time to do validation?
    if (epoch + 1) % args.val_freq != 0:
        epoch += 1
        continue

    # === VALIDATION ===

    # Set the module in evaluation mode
    model.eval()

    judge = Judge(r=args.radius)
    sum_term1 = 0
    sum_term2 = 0
    sum_term3 = 0
    sum_loss = 0
    iter_val = tqdm(valset_loader,
                    desc=f'Validating Epoch {epoch} ({len(valset_loader.dataset)} images)')
    for batch_idx, (imgs, dictionaries) in enumerate(iter_val):

        # Pull info from this batch and move to device
        imgs = imgs.to(device)
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_counts = [dictt['count'].to(device)
                        for dictt in dictionaries]
        target_orig_heights = [dictt['orig_height'].to(device)
                               for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                              for dictt in dictionaries]

        with torch.no_grad():
            target_counts = torch.stack(target_counts)
            target_orig_heights = torch.stack(target_orig_heights)
            target_orig_widths = torch.stack(target_orig_widths)
            target_orig_sizes = torch.stack((target_orig_heights,
                                             target_orig_widths)).transpose(0, 1)
        orig_shape = (dictionaries[0]['orig_height'].item(),
                      dictionaries[0]['orig_width'].item())

        # Tensor -> float & numpy
        target_count_int = int(round(target_counts.item()))
        target_locations_np = \
            target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
        target_orig_size_np = \
            target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

        normalzr = utils.Normalizer(args.height, args.width)

        if target_count_int == 0:
            continue

        # Feed-forward
        with torch.no_grad():
            est_maps, est_counts = model.forward(imgs)

        # Tensor -> int
        est_count_int = int(round(est_counts.item()))

        # The 3 terms
        with torch.no_grad():
            est_counts = est_counts.view(-1)
            target_counts = target_counts.view(-1)
            term1, term2 = loss_loc.forward(est_maps,
                                            target_locations,
                                            target_orig_sizes)
            term3 = loss_regress.forward(est_counts, target_counts)
            term3 *= args.lambdaa
        sum_term1 += term1.item()
        sum_term2 += term2.item()
        sum_term3 += term3.item()
        sum_loss += term1 + term2 + term3

        # Update progress bar
        loss_avg_this_epoch = sum_loss.item() / (batch_idx + 1)
        iter_val.set_postfix(
            avg_val_loss_this_epoch=f'{loss_avg_this_epoch:.1f}-----')

        # The estimated map must be thresholed to obtain estimated points
        # BMM thresholding
        est_map_numpy = est_maps[0, :, :].to(device_cpu).numpy()
        est_map_numpy_origsize = skimage.transform.resize(est_map_numpy,
                                                          output_shape=orig_shape,
                                                          mode='constant')
        mask, _ = utils.threshold(est_map_numpy_origsize, tau=-1)
        # Obtain centroids of the mask
        centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                           max_mask_pts=args.max_mask_pts)

        # Validation metrics
        target_locations_wrt_orig = normalzr.unnormalize(target_locations_np,
                                                         orig_img_size=target_orig_size_np)
        judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                          max_ahd=loss_loc.max_dist)
        judge.feed_count(est_count_int, target_count_int)

        if time.time() > tic_val + args.log_interval:
            tic_val = time.time()

            # Resize to original size
            orig_img_origsize = ((skimage.transform.resize(imgs[0].to(device_cpu).squeeze().numpy().transpose((1, 2, 0)),
                                                           output_shape=target_orig_size_np.tolist(),
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(est_maps[0].to(device_cpu).unsqueeze(0).numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1)).squeeze(0)

            # Overlay output on heatmap
            orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                map=est_map_origsize).\
                astype(np.float32)

            # # Read image with GT dots from disk
            # gt_img_numpy = skimage.io.imread(
            #     os.path.join('/home/jprat/projects/phenosorg/data/plant_counts_dots/20160613_F54_validation_256x256_white_bigdots',
            #                  dictionary['filename'][0]))
            # # dots_img_tensor = torch.from_numpy(gt_img_numpy).permute(
            #     # 2, 0, 1)[0, :, :].type(torch.FloatTensor) / 255
            # # Send GT image to Visdom
            # viz.image(np.moveaxis(gt_img_numpy, 2, 0),
            #           opts=dict(title='(Validation) Ground Truth'),
            #           win=7)
            if not args.paint:
                # Send input and output heatmap (first one in the batch)
                log.image(imgs=[orig_img_w_heatmap_origsize],
                          titles=['(Validation) Image w/ output heatmap'],
                          window_ids=[5])
            else:
                # Send heatmap with a cross at the estimated centroids to Visdom
                img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                 points=centroids_wrt_orig,
                                                 color='red',
                                                 crosshair=True )
                log.image(imgs=[img_with_x],
                          titles=['(Validation) Image w/ output heatmap '
                                  'and point estimations'],
                          window_ids=[8])

    avg_term1_val = sum_term1 / len(valset_loader)
    avg_term2_val = sum_term2 / len(valset_loader)
    avg_term3_val = sum_term3 / len(valset_loader)
    avg_loss_val = sum_loss / len(valset_loader)

    # Log validation metrics
    log.val_losses(terms=(avg_term1_val,
                          avg_term2_val,
                          avg_term3_val,
                          avg_loss_val / 3,
                          judge.mahd,
                          judge.mae,
                          judge.rmse,
                          judge.mape,
                          judge.coeff_of_determination,
                          judge.pearson_corr \
                              if not np.isnan(judge.pearson_corr) else 1,
                          judge.precision,
                          judge.recall),
                   iteration_number=epoch,
                   terms_legends=['Term 1',
                                  'Term 2',
                                  'Term3*%s' % args.lambdaa,
                                  'Sum/3',
                                  'AHD',
                                  'MAE',
                                  'RMSE',
                                  'MAPE (%)',
                                  'R^2',
                                  'r',
                                  f'r{args.radius}-Precision (%)',
                                  f'r{args.radius}-Recall (%)'])

    # If this is the best epoch (in terms of validation error)
    if judge.mahd < lowest_mahd:
        # Keep the best model
        lowest_mahd = judge.mahd
        if args.save:
            torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                        'model': model.state_dict(),
                        'mahd': lowest_mahd,
                        'optimizer': optimizer.state_dict(),
                        'n_points': args.n_points,
                        }, args.save)
            print("Saved best checkpoint so far in %s " % args.save)

    epoch += 1


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
