from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
from parse import parse
import math
from collections import OrderedDict

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
from torchvision.models import inception_v3
from sklearn import mixture
import skimage.transform
from .data import CSVDataset
from .data import csv_collator
from .data import ScaleImageAndLabel
from peterpy import peter

from . import losses
from . import argparser
from .models import unet_model
from .metrics import Judge


# Parse command line arguments
args = argparser.parse_command_args('testing')

# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
device_cpu = torch.device('cpu')
device = torch.device('cuda') if args.cuda else device_cpu

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Create output directories
os.makedirs(os.path.join(args.out_dir, 'est_map'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map_thresholded'), exist_ok=True)
if args.paint:
    os.makedirs(os.path.join(args.out_dir, 'painted'), exist_ok=True)

# Data loading code
try:
    testset = CSVDataset(args.dataset,
                         transforms=transforms.Compose([
                             # ScaleImageAndLabel(size=(args.height, args.width)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5),
                                                  (0.5, 0.5, 0.5)),
                         ]),
                         ignore_gt=not args.evaluate,
                         max_dataset_size=args.max_testset_size)
except ValueError as e:
    print(f'E: {e}')
    exit(-1)
testset_loader = data.DataLoader(testset,
                                 batch_size=1,
                                 num_workers=args.nThreads,
                                 collate_fn=csv_collator)

# Array with [height, width] of the new size
resized_size = np.array([args.height, args.width])

# Loss function
criterion_training = losses.WeightedHausdorffDistance(resized_height=args.height,
                                                      resized_width=args.width,
                                                      return_2_terms=True,
                                                      device=device)

# Restore saved checkpoint (model weights)
with peter("Loading checkpoint"):

    # Pretrained models that come with this package
    if args.model == 'unet_256x256_sorghum':
        args.model = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'checkpoints',
                                  'unet_256x256_sorghum.ckpt')
    if os.path.isfile(args.model):
        if args.cuda:
            checkpoint = torch.load(args.model)
        else:
            checkpoint = torch.load(
                args.model, map_location=lambda storage, loc: storage)
        # Model
        if args.n_points is None:
            if 'n_points' not in checkpoint:
                # Model will also estimate # of points
                model = unet_model.UNet(3, 1,
                                        known_n_points=None,
                                        height=args.height,
                                        width=args.width)
            else:
                # The checkpoint tells us the # of points to estimate
                model = unet_model.UNet(3, 1,
                                        known_n_points=checkpoint['n_points'],
                                        height=args.height,
                                        width=args.width)
        else:
            # The user tells us the # of points to estimate
            model = unet_model.UNet(3, 1,
                                    known_n_points=args.n_points,
                                    height=args.height,
                                    width=args.width)

        # Parallelize
        if args.cuda:
            model = nn.DataParallel(model)
        model = model.to(device)

        # Load model in checkpoint
        if args.cuda:
            state_dict = checkpoint['model']
        else:
            # remove 'module.' of DataParallel
            state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]
                state_dict[name] = v
        model.load_state_dict(state_dict)
        print(f"\n\__ loaded checkpoint '{args.model}'")
        # print(model)
    else:
        print(f"\n\__  E: no checkpoint found at '{args.model}'")
        exit(-1)

    tic = time.time()


# Empty output CSV
df_out = pd.DataFrame()
# df_out = pd.DataFrame(columns=['count', 'locations'])
# df_out.index.name = 'filename'

# Set the module in evaluation mode
model.eval()

if testset.there_is_gt:
    judges = [Judge(r) for r in range(0, 16)]

for batch_idx, (imgs, dictionaries) in tqdm(enumerate(testset_loader),
                                            total=len(testset_loader)):

    # Move to device
    imgs = imgs.to(device)

    # Pull info from this batch and move to device
    if testset.there_is_gt:
        target_locations = [dictt['locations'].to(device)
                            for dictt in dictionaries]
        target_count = [dictt['count'].to(device)
                        for dictt in dictionaries]
    target_orig_heights = [dictt['orig_height'].to(device)
                           for dictt in dictionaries]
    target_orig_widths = [dictt['orig_width'].to(device)
                          for dictt in dictionaries]

    # Lists -> Tensor batches
    if testset.there_is_gt:
        target_count = torch.stack(target_count)
    target_orig_heights = torch.stack(target_orig_heights)
    target_orig_widths = torch.stack(target_orig_widths)
    target_orig_sizes = torch.stack((target_orig_heights,
                                     target_orig_widths)).transpose(0, 1)
    origsize = (dictionaries[0]['orig_height'].item(),
                dictionaries[0]['orig_width'].item())

    # Feed forward
    with torch.no_grad():
        est_map, est_count = model.forward(imgs)

    # Save estimated map to disk
    est_map_numpy = est_map[0, :, :].to(device_cpu).numpy()
    est_map_numpy_origsize = \
        skimage.transform.resize(est_map_numpy,
                                 output_shape=origsize,
                                 mode='constant')
    cv2.imwrite(os.path.join(args.out_dir,
                             'est_map',
                             dictionaries[0]['filename']),
                est_map_numpy_origsize)

    # The estimated map must be thresholded to obtain estimated points
    # mask = cv2.inRange(est_map_numpy_origsize, 2 / 255, 1)
    minn, maxx = est_map_numpy_origsize.min(), est_map_numpy_origsize.max()
    est_map_origsize_scaled = ((est_map_numpy_origsize - minn)/(maxx - minn)*255).round().astype(np.uint8).squeeze()
    th, mask = cv2.threshold(est_map_origsize_scaled,
                             0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coord = np.where(mask > 0)
    y = coord[0].reshape((-1, 1))
    x = coord[1].reshape((-1, 1))
    c = np.concatenate((y, x), axis=1)
    if len(c) == 0:
        ahd = criterion_training.max_dist
        centroids = np.array([])
    else:
        n_components = int(torch.round(est_count[0]).to(device_cpu).numpy()[0])
        # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
        n_components = max(min(n_components, x.size), 1)
        centroids = mixture.GaussianMixture(n_components=n_components,
                                            n_init=1,
                                            covariance_type='full').\
            fit(c).means_.astype(np.int)

    # Save thresholded map to disk
    cv2.imwrite(os.path.join(args.out_dir,
                             'est_map_thresholded',
                             dictionaries[0]['filename']),
                mask)

    # Paint red dots if user asked for it
    if args.paint:
        # Paint a circle in the original image at the estimated location
        image_with_x = np.moveaxis(imgs[0, :, :].to(device_cpu).numpy(),
                                   0, 2).copy()
        image_with_x = \
            skimage.transform.resize(image_with_x,
                                     output_shape=origsize,
                                     mode='constant')
        image_with_x = ((image_with_x + 1) / 2.0 * 255.0)
        for y, x in centroids:
            image_with_x = cv2.circle(image_with_x, (x, y), 3, [255, 0, 0], -1)
        # Save original image with circle to disk
        image_with_x = image_with_x[:, :, ::-1]
        cv2.imwrite(os.path.join(args.out_dir,
                                 'painted',
                                 dictionaries[0]['filename']),
                    image_with_x)

    # Convert to numpy
    est_count = est_count.to(device_cpu).numpy()[0][0]

    if args.evaluate:
        # Convert to numpy
        target_count = target_count.item()
        target_locations = \
            target_locations[0].to(device_cpu).numpy().reshape(-1, 2)

        # Normalize to use locations in the original image
        norm_factor = target_orig_sizes[0].unsqueeze(0).cpu().numpy() \
            / resized_size
        norm_factor = norm_factor.repeat(len(target_locations), axis=0)
        target_locations_wrt_orig = norm_factor*target_locations

        # Compute metrics for each value of r (for each Judge)
        for judge in judges:
            judge.feed_points(centroids, target_locations_wrt_orig,
                              max_ahd=math.sqrt(origsize[0]**2 + origsize[1]**2))
            judge.feed_count(est_count, target_count)

    # Save a new line in the CSV corresonding to the resuls of this img
    res_dict = dictionaries[0]
    res_dict['count'] = est_count
    res_dict['locations'] = str(centroids.tolist())
    for key, val in res_dict.copy().items():
        if 'height' in key or 'width' in key:
            del res_dict[key]
    df = pd.DataFrame(data=res_dict,
                      index=[res_dict['filename']])
    df.index.name = 'filename'
    df_out = df_out.append(df)

# Write CSV to disk
df_out.to_csv(os.path.join(args.out_dir, 'estimations.csv'))

if args.evaluate:

    # Output CSV where we will put
    # the precision as a function of r
    df_prec_n_rec = pd.DataFrame(columns=['precision', 'recall', 'fscore'])
    df_prec_n_rec.index.name = 'r'

    print('\__  Location metrics for all the testing set, r=0, ..., 15')
    for judge in judges:
        print(f'r={judge.r} => Precision: {judge.precision:.3f}, '
              f'Recall: {judge.recall:.3f}, F-score: {judge.fscore:.3f}')

        # Accumulate precision and recall in the CSV dataframe
        df = pd.DataFrame(data=[[judge.precision, judge.recall, judge.fscore]],
                          index=[judge.r],
                          columns=['precision', 'recall', 'fscore'])
        df.index.name = 'r'
        df_prec_n_rec = df_prec_n_rec.append(df)
    print(f'\__ Average AHD for all the testing set: {judge.mahd:.3f}')

    # Regression metrics
    # (any judge will do as regression metrics don't depend on r)
    print(f'\__  MAPE for all the testing set: {judge.mape:.3f} %')
    print(f'\__  ME for all the testing set: {judge.me:+.3f}')
    print(f'\__  MPE for all the testing set: {judge.mpe:+.3f} %')
    print(f'\__  MAE for all the testing set: {judge.mae:.3f}')
    print(f'\__  MSE for all the testing set: {judge.mse:.3f}')
    print(f'\__  RMSE for all the testing set: {judge.rmse:.3f}')

    # Write CSV to disk
    df_prec_n_rec.to_csv(os.path.join(
        args.out_dir, 'precision_and_recall.csv'))

print('It took %s seconds to evaluate all the testing set.' %
      int(time.time() - tic))
