from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
from parse import parse
import math

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
from .data import CSVDataset
from .data import csv_collator

from . import losses
from . import argparser
from .models import unet_model
from .eval_precision_recall import Judge


if __name__ != '__main__':
    print("E: This script must be called from command line")
    exit(1)

# Parse command line arguments
args = argparser.parse_command_args('testing')

# Set seeds
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Create output directories
os.makedirs(os.path.join(args.out_dir, 'painted'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map_thresholded'), exist_ok=True)

# Tensor type to use, select CUDA or not
tensortype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
tensortype_cpu = torch.FloatTensor

# Data loading code
testset = CSVDataset(args.dataset,
                     transforms=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)),
                     ]),
                     max_dataset_size=args.max_testset_size,
                     tensortype=tensortype_cpu)
testset_loader = data.DataLoader(testset,
                                 batch_size=1,
                                 num_workers=args.nThreads,
                                 collate_fn=csv_collator)

# Loss function
l1_loss = nn.L1Loss(reduce=False)
mse_loss = nn.MSELoss(reduce=False)
criterion_training = losses.WeightedHausdorffDistance(height=args.height,
                                                      width=args.width,
                                                      return_2_terms=True,
                                                      tensortype=tensortype)

# Restore saved checkpoint (model weights)
print("Loading checkpoint '{}' ...".format(args.model))

# Pretrained models that come with this package
if args.model == 'unet_256x256_sorghum':
    args.model = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'models',
                              'unet_256x256_sorghum.ckpt')
if os.path.isfile(args.model):
    checkpoint = torch.load(args.model)
    # Model
    if args.n_points is None:
        if 'n_points' not in checkpoint:
            # Model will also estimate # of points
            model = unet_model.UNet(3, 1,
                                    known_n_points=None,
                                    height=args.height,
                                    width=args.width,
                                    tensortype=tensortype)
        else:
            # The checkpoint tells us the # of points to estimate
            model = unet_model.UNet(3, 1,
                                    known_n_points=checkpoint['n_points'],
                                    height=args.height,
                                    width=args.width,
                                    tensortype=tensortype)
    else:
        # The user tells us the # of points to estimate
        model = unet_model.UNet(3, 1,
                                known_n_points=args.n_points,
                                height=args.height,
                                width=args.width,
                                tensortype=tensortype)

    # Parallelize
    model = nn.DataParallel(model)
    if args.cuda:
        model.cuda()

    # Load model in checkpoint
    model.load_state_dict(checkpoint['model'])
    print("\__ loaded checkpoint '{}'".format(args.model))
    # print(model)
else:
    print("\__  E: no checkpoint found at '{}'".format(args.model))
    exit(-1)

tic = time.time()


# Empty output CSV
df_out = pd.DataFrame(columns=['count'])

# Set the module in evaluation mode
model.eval()

if testset.there_is_gt:
    judges = [Judge(r) for r in range(0, 16)]
    sum_ahd = 0
    sum_ae = 0
    sum_se = 0
    sum_ape = 0

for batch_idx, (imgs, dictionaries) in tqdm(enumerate(testset_loader),
                                            total=len(testset_loader)):

    imgs = Variable(imgs.type(tensortype), volatile=True)

    if testset.there_is_gt:
        # Pull info from this batch
        target_locations = [dictt['locations'] for dictt in dictionaries]
        target_count = torch.stack([dictt['count']
                                    for dictt in dictionaries])

        # Prepare targets
        target_locations = [Variable(t.type(tensortype), volatile=True)
                            for t in target_locations]
        target_count = Variable(target_count.type(tensortype), volatile=True)

    # Feed forward
    est_map, est_count = model.forward(imgs)

    # Save estimated map to disk
    tv.utils.save_image(est_map.data[0, :, :],
                        os.path.join(args.out_dir,
                                     'est_map',
                                     dictionaries[0]['filename']))

    # The estimated map must be thresholded to obtain estimated points
    est_map_numpy = est_map.data[0, :, :].cpu().numpy()
    mask = cv2.inRange(est_map_numpy, 2 / 255, 1)
    coord = np.where(mask > 0)
    y = coord[0].reshape((-1, 1))
    x = coord[1].reshape((-1, 1))
    c = np.concatenate((y, x), axis=1)
    if len(c) == 0:
        continue
        ahd = criterion_training.max_dist
        centroids = []
    else:
        n_components = int(torch.round(est_count[0]).data.cpu().numpy()[0])
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
        image_with_x = tensortype(imgs.data[0, :, :].squeeze().size()).\
            copy_(imgs.data[0, :, :].squeeze())
        image_with_x = ((image_with_x + 1) / 2.0 * 255.0)
        image_with_x = image_with_x.cpu().numpy()
        image_with_x = np.moveaxis(image_with_x, 0, 2).copy()
        for y, x in centroids:
            image_with_x = cv2.circle(image_with_x, (x, y), 3, [255, 0, 0], -1)
        # Save original image with circle to disk
        image_with_x = image_with_x[:, :, ::-1]
        cv2.imwrite(os.path.join(args.out_dir,
                                 'painted',
                                 dictionaries[0]['filename']),
                    image_with_x)

    if testset.there_is_gt:
        # Evaluate Average Percent Error for this image
        if bool((target_count == 0).data.cpu().numpy()[0][0]):
            ape = 100 * l1_loss.forward(est_count, target_count)
        else:
            ape = 100 * l1_loss.forward(est_count,
                                        target_count) / target_count
        ae = l1_loss.forward(est_count, target_count)
        se = mse_loss.forward(est_count, target_count)

        ape = ape.data.cpu().numpy()[0][0]
        ae = ae.data.cpu().numpy()[0][0]
        se = se.data.cpu().numpy()[0][0]

        sum_ae += ae
        sum_se += se
        sum_ape += ape

        # Evaluation using the Averaged Hausdorff Distance
        target_locations = \
            target_locations[0].data.cpu().numpy().reshape(-1, 2)
        ahd = losses.averaged_hausdorff_distance(centroids,
                                                 target_locations)

        sum_ahd += ahd

        # Validation using Precision and Recall
        for judge in judges:
            judge.evaluate_sample(centroids, target_locations)

    df = pd.DataFrame(data=[est_count.data[0, 0]],
                      index=[dictionaries[0]['filename']],
                      columns=['count'])
    df_out = df_out.append(df)

if testset.there_is_gt:
    avg_ahd = sum_ahd / len(testset_loader)
    mae = sum_ae / len(testset_loader)
    mse = sum_se / len(testset_loader)
    rmse = math.sqrt(sum_se / len(testset_loader))
    mape = sum_ape / len(testset_loader)

    # Output CSV where we will put
    # the precision as a function of r
    df_prec_n_rec = pd.DataFrame(columns=['precision', 'recall'])


    print(f'\__ Average AHD for all the testing set: {avg_ahd:.3f}')
    print('\__  Accuracy for all the testing set, r=0, ..., 15')
    for judge in judges:
        prec, rec = judge.get_p_n_r()
        print(f'r={judge.r} => Precision: {prec:.3f}, Recall: {rec:.3f}')

        # Accumulate precision and recall in the CSV dataframe
        df = pd.DataFrame(data=[[prec, rec]],
                          index=[judge.r],
                          columns=['precision', 'recall'])
        df_prec_n_rec = df_prec_n_rec.append(df)
    print(f'\__  MAPE for all the testing set: {mape:.3f} %')
    print(f'\__  MAE for all the testing set: {mae:.3f}')
    print(f'\__  MSE for all the testing set: {mse:.3f}')
    print(f'\__  RMSE for all the testing set: {rmse:.3f}')

print('It took %s seconds to evaluate all the testing set.' %
      int(time.time() - tic))

# Write CSV to disk
df_out.to_csv(os.path.join(args.out_dir, 'estimations.csv'))
df_prec_n_rec.to_csv(os.path.join(args.out_dir, 'precision_and_recall.csv'))
