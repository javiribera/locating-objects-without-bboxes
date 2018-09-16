from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
from parse import parse
import math
from collections import OrderedDict
import itertools

import matplotlib
matplotlib.use('Agg')
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
from peterpy import peter
from ballpark import ballpark

from .data import CSVDataset
from .data import csv_collator
from .data import ScaleImageAndLabel
from . import losses
from . import argparser
from .models import unet_model
from .metrics import Judge
from .metrics import make_metric_plots
from . import utils


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

# Data loading code
try:
    testset = CSVDataset(args.dataset,
                         transforms=transforms.Compose([
                             ScaleImageAndLabel(size=(args.height, args.width)),
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
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n\__ loaded checkpoint '{args.model}' "
              f"with {ballpark(num_params)} trainable parameters")
        # print(model)
    else:
        print(f"\n\__  E: no checkpoint found at '{args.model}'")
        exit(-1)

    tic = time.time()


# Set the module in evaluation mode
model.eval()

if testset.there_is_gt:
    # Prepare Judges that will compute P/R as fct of r and th
    judges = []
    for r, th in itertools.product(args.radii, args.taus):
        judge = Judge(r=r)
        judge.th = th
        judges.append(judge)

# Empty output CSV (one per threshold)
df_outs = [pd.DataFrame() for _ in args.taus]

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

    # Tensor -> float & numpy
    target_count = target_count.item()
    target_locations = \
        target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
    target_orig_size = \
        target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

    normalzr = utils.Normalizer(args.height, args.width)

    # Feed forward
    with torch.no_grad():
        est_map, est_count = model.forward(imgs)

    # Save estimated map to disk
    est_map_numpy = est_map[0, :, :].to(device_cpu).numpy()
    est_map_numpy_origsize = \
        skimage.transform.resize(est_map_numpy,
                                 output_shape=origsize,
                                 mode='constant')
    os.makedirs(os.path.join(args.out_dir, 'estimated_map'), exist_ok=True)
    cv2.imwrite(os.path.join(args.out_dir,
                             'estimated_map',
                             dictionaries[0]['filename']),
                est_map_numpy_origsize)

    # Tensor -> int
    est_count_int = int(round(est_count.item()))
    
    # The estimated map must be thresholded to obtain estimated points
    for tau, df_out in zip(args.taus, df_outs):
        mask, _ = utils.threshold(est_map_numpy_origsize, tau)
        coord = np.where(mask > 0)
        y = coord[0].reshape((-1, 1))
        x = coord[1].reshape((-1, 1))
        c = np.concatenate((y, x), axis=1)
        if len(c) == 0:
            ahd = criterion_training.max_dist
            centroids_wrt_orig = np.array([])
        else:
            # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
            n_components = max(min(est_count_int, x.size), 1)
            centroids_wrt_orig = mixture.GaussianMixture(n_components=n_components,
                                                n_init=1,
                                                covariance_type='full').\
                fit(c).means_.astype(np.int)


        # Save thresholded map to disk
        os.makedirs(os.path.join(args.out_dir, 'estimated_map_thresholded', f'tau={tau}'),
                    exist_ok=True)
        cv2.imwrite(os.path.join(args.out_dir, 'estimated_map_thresholded', f'tau={tau}',
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
            for y, x in centroids_wrt_orig:
                image_with_x = cv2.circle(image_with_x, (x, y), 3, [255, 0, 0], -1)
            # Save original image with circle to disk
            image_with_x = image_with_x[:, :, ::-1]
            os.makedirs(os.path.join(args.out_dir, 'painted', f'tau={tau}'), exist_ok=True)
            cv2.imwrite(os.path.join(args.out_dir, 'painted', f'tau={tau}',
                                     dictionaries[0]['filename']),
                        image_with_x)


        if args.evaluate:
            target_locations_wrt_orig = normalzr.unnormalize(target_locations,
                                                             orig_img_size=target_orig_size)

            # Compute metrics for each value of r (for each Judge)
            for judge in judges:
                if judge.th != tau:
                    continue
                judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                                  max_ahd=math.sqrt(origsize[0]**2 + origsize[1]**2))
                judge.feed_count(est_count_int, target_count)

        # Save a new line in the CSV corresonding to the resuls of this img
        res_dict = dictionaries[0]
        res_dict['count'] = est_count
        res_dict['locations'] = str(centroids_wrt_orig.tolist())
        for key, val in res_dict.copy().items():
            if 'height' in key or 'width' in key:
                del res_dict[key]
        df = pd.DataFrame(data=res_dict,
                          index=[res_dict['filename']])
        df.index.name = 'filename'
        df_out = df_out.append(df)

# Write CSVs to disk
for df_out, tau in zip(df_outs, args.taus):
    df_out.to_csv(os.path.join(args.out_dir, f'estimations_tau={tau}.csv'))

if args.evaluate:

    with peter("Evauating metrics"):

        # Output CSV where we will put
        # all our metrics as a function of r and the threshold
        df_metrics = pd.DataFrame(columns=['r', 'th',
                                              'precision', 'recall', 'fscore', 'MAHD',
                                              'MAPE', 'ME', 'MPE', 'MAE',
                                              'MSE', 'RMSE', 'r', 'R2'])
        df_metrics.index.name = 'idx'

        for j, judge in enumerate(tqdm(judges)):
            # Accumulate precision and recall in the CSV dataframe
            df = pd.DataFrame(data=[[judge.r,
                                     judge.th,
                                     judge.precision,
                                     judge.recall,
                                     judge.fscore,
                                     judge.mahd,
                                     judge.mape,
                                     judge.me,
                                     judge.mpe,
                                     judge.mae,
                                     judge.mse,
                                     judge.rmse,
                                     judge.pearson_corr,
                                     judge.coeff_of_determination]],
                              columns=['r', 'th',
                                       'precision', 'recall', 'fscore', 'MAHD',
                                       'MAPE', 'ME', 'MPE', 'MAE',
                                       'MSE', 'RMSE', 'r', 'R2'],
                              index=[j])
            df.index.name = 'idx'
            df_metrics = df_metrics.append(df)

        # Write CSV of metrics to disk
        df_metrics.to_csv(os.path.join(args.out_dir, 'metrics.csv'))

        # Generate plots
        figs = make_metric_plots(csv_path=os.path.join(args.out_dir, 'metrics.csv'),
                                 taus=args.taus,
                                 radii=args.radii)
        os.makedirs(os.path.join(args.out_dir, 'metrics_plots'), exist_ok=True)
        for label, fig in figs.items():
            # Save to disk
            fig.savefig(os.path.join(args.out_dir, 'metrics_plots', f'{label}.png'))


elapsed_time = int(time.time() - tic)
print(f'It took {elapsed_time} seconds to evaluate all the testing set.')

