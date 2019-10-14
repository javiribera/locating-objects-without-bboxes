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
import skimage.transform
from peterpy import peter
from ballpark import ballpark

from .data import csv_collator
from .data import ScaleImageAndLabel
from .data import build_dataset
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
    testset = build_dataset(args.dataset,
                            transforms=transforms.Compose([
                                ScaleImageAndLabel(size=(args.height,
                                                         args.width)),
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
                                        width=args.width,
                                        ultrasmall=args.ultrasmallnet)

            else:
                # The checkpoint tells us the # of points to estimate
                model = unet_model.UNet(3, 1,
                                        known_n_points=checkpoint['n_points'],
                                        height=args.height,
                                        width=args.width,
                                        ultrasmall=args.ultrasmallnet)
        else:
            # The user tells us the # of points to estimate
            model = unet_model.UNet(3, 1,
                                    known_n_points=args.n_points,
                                    height=args.height,
                                    width=args.width,
                                    ultrasmall=args.ultrasmallnet)

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

# Accumulative histogram of estimated maps
bmm_tracker = utils.AccBetaMixtureModel()


if testset.there_is_gt:
    # Prepare Judges that will compute P/R as fct of r and th
    judges = []
    for r, th in itertools.product(args.radii, args.taus):
        judge = Judge(r=r)
        judge.th = th
        judges.append(judge)

# Empty output CSV (one per threshold)
df_outs = [pd.DataFrame() for _ in args.taus]

# --force will overwrite output directory
if args.force:
    shutil.rmtree(args.out)

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
    if testset.there_is_gt:
        target_count = target_count.item()
        target_locations = \
            target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
    target_orig_size = \
        target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

    normalzr = utils.Normalizer(args.height, args.width)

    # Feed forward
    with torch.no_grad():
        est_maps, est_count = model.forward(imgs)

    # Convert to original size
    est_map_np = est_maps[0, :, :].to(device_cpu).numpy()
    est_map_np_origsize = \
        skimage.transform.resize(est_map_np,
                                 output_shape=origsize,
                                 mode='constant')
    orig_img_np = imgs[0].to(device_cpu).squeeze().numpy()
    orig_img_np_origsize = ((skimage.transform.resize(orig_img_np.transpose((1, 2, 0)),
                                                   output_shape=origsize,
                                                   mode='constant') + 1) / 2.0 * 255.0).\
        astype(np.float32).transpose((2, 0, 1))

    # Overlay output on original image as a heatmap
    orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_np_origsize,
                                                        map=est_map_np_origsize).\
        astype(np.float32)

    # Save estimated map to disk
    os.makedirs(os.path.join(args.out, 'intermediate', 'estimated_map'),
                exist_ok=True)
    cv2.imwrite(os.path.join(args.out,
                             'intermediate',
                             'estimated_map',
                             dictionaries[0]['filename']),
                orig_img_w_heatmap_origsize.transpose((1, 2, 0))[:, :, ::-1])

    # Tensor -> int
    est_count_int = int(round(est_count.item()))

    # The estimated map must be thresholded to obtain estimated points
    for t, tau in enumerate(args.taus):
        if tau != -2:
            mask, _ = utils.threshold(est_map_np_origsize, tau)
        else:
            mask, _, mix = utils.threshold(est_map_np_origsize, tau)
            bmm_tracker.feed(mix)
        centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                           max_mask_pts=args.max_mask_pts)

        # Save thresholded map to disk
        os.makedirs(os.path.join(args.out,
                                 'intermediate',
                                 'estimated_map_thresholded',
                                 f'tau={round(tau, 4)}'),
                    exist_ok=True)
        cv2.imwrite(os.path.join(args.out,
                                 'intermediate',
                                 'estimated_map_thresholded',
                                 f'tau={round(tau, 4)}',
                                 dictionaries[0]['filename']),
                    mask)

        # Paint red dots if user asked for it
        if args.paint:
            # Paint a cross at the estimated centroids
            img_with_x_n_map = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                   points=centroids_wrt_orig,
                                                   color='red',
                                                   crosshair=True)
            # Save to disk
            os.makedirs(os.path.join(args.out,
                                     'intermediate',
                                     'painted_on_estimated_map',
                                     f'tau={round(tau, 4)}'), exist_ok=True)
            cv2.imwrite(os.path.join(args.out,
                                     'intermediate',
                                     'painted_on_estimated_map',
                                     f'tau={round(tau, 4)}',
                                     dictionaries[0]['filename']),
                        img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1])
            # Paint a cross at the estimated centroids
            img_with_x = utils.paint_circles(img=orig_img_np_origsize,
                                             points=centroids_wrt_orig,
                                             color='red',
                                             crosshair=True)
            # Save to disk
            os.makedirs(os.path.join(args.out,
                                     'intermediate',
                                     'painted_on_original',
                                     f'tau={round(tau, 4)}'), exist_ok=True)
            cv2.imwrite(os.path.join(args.out,
                                     'intermediate',
                                     'painted_on_original',
                                     f'tau={round(tau, 4)}',
                                     dictionaries[0]['filename']),
                        img_with_x.transpose((1, 2, 0))[:, :, ::-1])


        if args.evaluate:
            target_locations_wrt_orig = normalzr.unnormalize(target_locations,
                                                             orig_img_size=target_orig_size)

            # Compute metrics for each value of r (for each Judge)
            for judge in judges:
                if judge.th != tau:
                    continue
                judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                                  max_ahd=criterion_training.max_dist)
                judge.feed_count(est_count_int, target_count)

        # Save a new line in the CSV corresonding to the resuls of this img
        res_dict = dictionaries[0]
        res_dict['count'] = est_count_int
        res_dict['locations'] = str(centroids_wrt_orig.tolist())
        for key, val in res_dict.copy().items():
            if 'height' in key or 'width' in key:
                del res_dict[key]
        df = pd.DataFrame(data={idx: [val] for idx, val in res_dict.items()})
        df = df.set_index('filename')
        df_outs[t] = df_outs[t].append(df)

# Write CSVs to disk
os.makedirs(os.path.join(args.out, 'estimations'), exist_ok=True)
for df_out, tau in zip(df_outs, args.taus):
    df_out.to_csv(os.path.join(args.out,
                               'estimations',
                               f'estimations_tau={round(tau, 4)}.csv'))

os.makedirs(os.path.join(args.out, 'intermediate', 'metrics_plots'),
            exist_ok=True)

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
        df_metrics.to_csv(os.path.join(args.out, 'metrics.csv'))

        # Generate plots
        figs = make_metric_plots(csv_path=os.path.join(args.out, 'metrics.csv'),
                                 taus=args.taus,
                                 radii=args.radii)
        for label, fig in figs.items():
            # Save to disk
            fig.savefig(os.path.join(args.out,
                                     'intermediate',
                                     'metrics_plots',
                                     f'{label}.png'))


# Save plot figures of the statistics of the BMM-based threshold
if -2 in args.taus:
    for label, fig in bmm_tracker.plot().items():
        fig.savefig(os.path.join(args.out,
                                 'intermediate',
                                 'metrics_plots',
                                 f'{label}.png'))


elapsed_time = int(time.time() - tic)
print(f'It took {elapsed_time} seconds to evaluate all this dataset.')


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
