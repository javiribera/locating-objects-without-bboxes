from __future__ import print_function

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
from sklearn import mixture
import skimage.transform

from . import losses
from .models import unet_model
from .eval_precision_recall import Judge
from .data import CSVDataset
from .data import XMLDataset
from .data import csv_collator
from .data import RandomHorizontalFlipImageAndLabel
from .data import RandomVerticalFlipImageAndLabel
from .data import ScaleImageAndLabel
from . import logger
from . import argparser


# Parse command line arguments
args = argparser.parse_command_args('training')

# Tensor type to use, select CUDA or not
tensortype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
tensortype_cpu = torch.FloatTensor

# Create directory for checkpoint to be saved
if args.save:
    os.makedirs(os.path.split(args.save)[0], exist_ok=True)

# Set seeds
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Visdom setup
log = logger.Logger(env_name=args.env_name)

# Data loading code
training_transforms = []
if not args.no_data_augm:
    training_transforms += [RandomHorizontalFlipImageAndLabel(p=0.5)]
    training_transforms += [RandomVerticalFlipImageAndLabel(p=0.5)]
training_transforms += [ScaleImageAndLabel(size=(args.height, args.width))]
training_transforms += [transforms.ToTensor()]
training_transforms += [transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))]
trainset = XMLDataset(args.train_dir,
                      transforms=transforms.Compose(training_transforms),
                      max_dataset_size=args.max_trainset_size,
                      tensortype=tensortype_cpu)
trainset_loader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             drop_last=args.drop_last_batch,
                             shuffle=True,
                             num_workers=args.nThreads,
                             collate_fn=csv_collator)
if args.val_dir:
    valset = XMLDataset(args.val_dir,
                        transforms=transforms.Compose([
                            ScaleImageAndLabel(size=(args.height, args.width)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)),
                        ]),
                        max_dataset_size=args.max_valset_size,
                        tensortype=tensortype_cpu)
    valset_loader = DataLoader(valset,
                               batch_size=args.eval_batch_size,
                               shuffle=True,
                               num_workers=args.nThreads,
                               collate_fn=csv_collator)

# Model
print('Building network... ', end='')
model = unet_model.UNet(3, 1,
                        height=args.height,
                        width=args.width,
                        known_n_points=args.n_points,
                        tensortype=tensortype)
print('DONE')
print(model)
model = nn.DataParallel(model)
if args.cuda:
    model.cuda()

# Loss function
loss_regress = nn.SmoothL1Loss()
loss_loc = losses.WeightedHausdorffDistance(resized_height=args.height,
                                            resized_width=args.width,
                                            return_2_terms=True,
                                            tensortype=tensortype)
l1_loss = nn.L1Loss(size_average=False)
mse_loss = nn.MSELoss(reduce=False)

# Optimization strategy
if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)

start_epoch = 0
lowest_avg_ahd_val = np.infty

# Restore saved checkpoint (model weights + epoch + optimizer state)
if args.resume:
    print("Loading checkpoint '{}' ...".format(args.resume))
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        lowest_avg_ahd_val = checkpoint['lowest_avg_ahd_val']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\__ loaded checkpoint '{}' (now on epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("\__ E: no checkpoint found at '{}'".format(args.resume))
        exit(-1)

# Time at the last evaluation
tic_train = -np.infty
tic_val = -np.infty

epoch = start_epoch
it_num = 0
while epoch < args.epochs:

    loss_avg_this_epoch = 0
    iter_train = tqdm(trainset_loader,
                      desc=f'Epoch {epoch} ({len(trainset)} images)')

    for batch_idx, (imgs, dictionaries) in enumerate(iter_train):
        # === TRAIN ===

        # Set the module in training mode
        model.train()

        # Pull info from this batch
        target_locations = [dictt['locations'] for dictt in dictionaries]
        target_count = torch.stack([dictt['count']
                                    for dictt in dictionaries])
        target_orig_heights = [dictt['orig_height'] for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'] for dictt in dictionaries]

        imgs = Variable(imgs.type(tensortype))
        target_locations = [Variable(t.type(tensortype))
                            for t in target_locations]
        target_count = Variable(target_count.type(tensortype))
        target_orig_heights = Variable(tensortype(target_orig_heights))
        target_orig_widths = Variable(tensortype(target_orig_widths))
        target_orig_sizes = torch.stack((target_orig_heights,
                                         target_orig_widths)).transpose(0, 1)

        # One training step
        optimizer.zero_grad()
        est_map, est_count = model.forward(imgs)
        term1, term2 = loss_loc.forward(
            est_map, target_locations, target_orig_sizes)
        term3 = loss_regress.forward(est_count, target_count)
        term3 *= args.lambdaa
        loss = term1 + term2 + term3
        loss.backward()
        optimizer.step()

        # Update progress bar
        loss_avg_this_epoch = (1/(batch_idx + 1))*(batch_idx * loss_avg_this_epoch +
                                                   loss.data[0])
        iter_train.set_postfix(
            avg_train_loss_this_epoch=f'{loss_avg_this_epoch:.1f}')

        # Log training error
        if time.time() > tic_train + args.log_interval:
            tic_train = time.time()

            # Log training losses
            log.train_losses(terms=[term1, term2, term3, loss / 3],
                             iteration_number=epoch +
                             batch_idx/len(trainset_loader),
                             terms_legends=['Term1',
                                            'Term2',
                                            'Term3*%s' % args.lambdaa,
                                            'Sum/3'])

            # Send input and output images (first one in the batch).
            # Resize to original size
            orig_shape = target_orig_sizes[0].data.cpu().numpy().tolist()
            orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().cpu().numpy().transpose((1, 2, 0)),
                                                           output_shape=orig_shape,
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(est_map[0].data.unsqueeze(0).cpu().numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1))
            log.image(imgs=[orig_img_origsize, est_map_origsize],
                      titles=['(Training) Input',
                              '(Training) U-Net output'],
                      windows=[1, 2])

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

    # Always save checkpoint at end of epoch if there is no validation set
    if not args.val_dir or \
            not valset_loader or \
            len(valset_loader) == 0 or \
            args.val_freq == 0:
        epoch += 1
        if args.save:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'n_points': args.n_points,
                        }, args.save)
        continue

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
    sum_ahd = 0
    sum_ae = 0
    sum_se = 0
    sum_ape = 0
    iter_val = tqdm(valset_loader,
                    desc=f'Validating Epoch {epoch} ({len(valset)} images)')
    for batch_idx, (imgs, dictionaries) in enumerate(iter_val):

        # Pull info from this batch
        target_locations = [dictt['locations'] for dictt in dictionaries]
        target_count = torch.stack([dictt['count']
                                    for dictt in dictionaries])
        target_orig_heights = [dictt['orig_height'] for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'] for dictt in dictionaries]

        if bool((target_count == 0).cpu().numpy()[0]):
            continue

        imgs = Variable(imgs.type(tensortype), volatile=True)
        target_locations = [Variable(t.type(tensortype), volatile=True)
                            for t in target_locations]
        target_count = Variable(target_count.type(tensortype), volatile=True)
        target_orig_heights = Variable(tensortype(target_orig_heights))
        target_orig_widths = Variable(tensortype(target_orig_widths))
        target_orig_sizes = torch.stack((target_orig_heights,
                                         target_orig_widths)).transpose(0, 1)

        # Feed-forward
        est_map, est_count = model.forward(imgs)

        # The 3 terms
        term1, term2 = loss_loc.forward(
            est_map, target_locations, target_orig_sizes)
        term3 = loss_regress.forward(est_count, target_count)
        term3 *= args.lambdaa
        sum_term1 += term1
        sum_term2 += term2
        sum_term3 += term3
        sum_loss += term1 + term2 + term3

        # Update progress bar
        loss_avg_this_epoch = sum_loss.data[0] / (batch_idx + 1)
        iter_val.set_postfix(
            avg_val_loss_this_epoch=f'{loss_avg_this_epoch:.1f}-----')

        # Validation using the Averaged Hausdorff Distance
        # __on the first image of the batch__
        # The estimated map must be thresholed to obtain estimated points
        est_map_numpy = est_map[0, :, :].data.cpu().numpy()
        mask = cv2.inRange(est_map_numpy, 4 / 255, 1)
        coord = np.where(mask > 0)
        y = coord[0].reshape((-1, 1))
        x = coord[1].reshape((-1, 1))
        c = np.concatenate((y, x), axis=1)
        if len(c) == 0:
            ahd = loss_loc.max_dist
            centroids = []
            print('len(c) == 0')
        else:
            n_components = int(torch.round(est_count).data.cpu().numpy()[0])
            # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
            n_components = max(min(n_components, x.size), 1)
            centroids = mixture.GaussianMixture(n_components=n_components,
                                                n_init=1,
                                                covariance_type='full').\
                fit(c).means_.astype(np.int)

            target_locations = \
                target_locations[0].data.cpu().numpy().reshape(-1, 2)
            ahd = losses.averaged_hausdorff_distance(
                centroids, target_locations)
        ahd = Variable(tensortype([ahd]), volatile=True)
        sum_ahd += ahd

        # Validation using MAE, MSE, MAPE
        ae = l1_loss.forward(est_count, target_count)
        se = mse_loss.forward(est_count, target_count)
        ape = torch.abs(target_count - est_count)/target_count
        sum_ae += ae
        sum_se += se
        sum_ape += ape

        # Validation using Precision and Recall
        judge.evaluate_sample(centroids, target_locations)

        if time.time() > tic_val + args.log_interval:
            tic_val = time.time()

            # Send input and output images (first one in the batch).
            # Resize to original size
            orig_shape = target_orig_sizes[0].data.cpu().numpy().tolist()
            orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().cpu().numpy().transpose((1, 2, 0)),
                                                           output_shape=orig_shape,
                                                           mode='constant') + 1) / 2.0 * 255.0).\
                astype(np.float32).transpose((2, 0, 1))
            est_map_origsize = skimage.transform.resize(est_map[0].data.unsqueeze(0).cpu().numpy().transpose((1, 2, 0)),
                                                        output_shape=orig_shape,
                                                        mode='constant').\
                astype(np.float32).transpose((2, 0, 1))
            log.image(imgs=[orig_img_origsize, est_map_origsize],
                      titles=['(Validation) Input',
                              '(Validation) U-Net output'],
                      windows=[5, 6])

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
            if args.paint:
                # Send original image with a cross at the estimated centroids to Visdom
                image_with_x = tensortype(imgs.data[0, :, :].squeeze().size()).\
                    copy_(imgs.data[0, :, :].squeeze())
                image_with_x = ((image_with_x + 1) / 2.0 * 255.0)
                image_with_x = image_with_x.cpu().numpy()
                image_with_x = np.moveaxis(image_with_x, 0, 2).copy()
                for y, x in centroids:
                    image_with_x = cv2.circle(
                        image_with_x, (x, y), 3, [255, 0, 0], -1)

                log.image(imgs=[np.moveaxis(image_with_x, 2, 0)],
                          titles=[
                              '(Validation) Estimated centers @ crossings'],
                          windows=[8])

    avg_term1_val = sum_term1 / len(valset_loader)
    avg_term2_val = sum_term2 / len(valset_loader)
    avg_term3_val = sum_term3 / len(valset_loader)
    avg_loss_val = sum_loss / len(valset_loader)
    avg_ahd_val = sum_ahd / len(valset_loader)
    mae = sum_ae / len(valset_loader)
    rmse = torch.sqrt(sum_se / len(valset_loader))
    mape = sum_ape/len(valset_loader)
    mae = mae.squeeze()
    rmse = rmse.squeeze()
    mape = mape.squeeze()
    prec, rec = judge.get_p_n_r()
    prec = Variable(tensortype([prec]), volatile=True)
    rec = Variable(tensortype([rec]), volatile=True)

    # Log validation metrics
    log.val_losses(terms=(avg_term1_val,
                          avg_term2_val,
                          avg_term3_val,
                          avg_loss_val / 3,
                          avg_ahd_val,
                          mae,
                          rmse,
                          mape*100,
                          prec,
                          rec),
                   iteration_number=epoch,
                   terms_legends=['Term 1',
                                  'Term 2',
                                  'Term3*%s' % args.lambdaa,
                                  'Sum/3',
                                  'AHD',
                                  'MAE',
                                  'RMSE',
                                  'MAPE (%)',
                                  f'r{args.radius}-Precision (%)',
                                  f'r{args.radius}-Recall (%)'])

    # If this is the best epoch (in terms of validation error)
    avg_ahd_val_float = avg_ahd_val.data.cpu().numpy()[0]
    if avg_ahd_val_float < lowest_avg_ahd_val:
        # Keep the best model
        lowest_avg_ahd_val = avg_ahd_val_float
        if args.save:
            torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                        'model': model.state_dict(),
                        'lowest_avg_ahd_val': avg_ahd_val_float,
                        'optimizer': optimizer.state_dict(),
                        'n_points': args.n_points,
                        }, args.save)
            print("Saved best checkpoint so far in %s " % args.save)

    epoch += 1
