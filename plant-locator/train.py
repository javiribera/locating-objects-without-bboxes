from __future__ import print_function

import cv2
import argparse
import os
import sys
import time
import shutil
from itertools import chain
from tqdm import tqdm

from parse import parse
import numpy as np
import torch
import torch.optim as optim
import visdom
import utils
from torch import nn
from torch.autograd import Variable
import torchvision as tv
from torchvision.models import inception_v3
from sklearn import mixture
import losses
from models import unet_model
from eval_precision_recall import Judge
from torchvision import transforms
from torch.utils.data import DataLoader
from data import CSVDataset
from data import RandomHorizontalFlipImageAndLabel
from data import RandomVerticalFlipImageAndLabel


# Training settings
parser = argparse.ArgumentParser(description='Plant Location with PyTorch',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', required=True,
                    help='Directory with training images.')
parser.add_argument('--val-dir',
                    help='Directory with validation images. If left blank no validation will be done.')
parser.add_argument('--imgsize', type=str, default='256x256', metavar='HxW',
                    help='Size of the input images (height x width).')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--eval-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=np.inf, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                    help='Number of data loading threads')
parser.add_argument('--lr', type=float, default=4e-5, metavar='LR',
                    help='learning rate (default: 1e-5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='where to save the model after each epoch')
parser.add_argument('--log-interval', type=float, default=3, metavar='N',
                    help='time to wait between logging training status (in seconds)')
parser.add_argument('--max-trainset-size', type=int, default=np.inf, metavar='N',
                    help='only use the first N images of the training dataset')
parser.add_argument('--max-valset-size', type=int, default=np.inf, metavar='N',
                    help='only use the first N images of the validation dataset')
parser.add_argument('--env-name', default='Pure U-Net', type=str, metavar='NAME',
                    help='Name of the environment in Visdom')
parser.add_argument('--paint', default=False, action="store_true",
                    help='Paint red circles at estimated locations in Validation? '
                    'It takes an enormous amount of time!')
parser.add_argument('--radius', type=int, default=5, metavar='R',
                    help='Detections at dist <= R to a GT pt are True Positives.')
parser.add_argument('--n-points', type=int, default=None, metavar='N',
                    help='If you know the number of points (e.g, just one pupil), set it.'
                    'Otherwise it will be estimated by adding a L1 cost term.')
parser.add_argument('--lambdaa', type=float, default=1, metavar='L',
                    help='Weight that will multiply the MAPE term in the loss function.')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

# Tensor type to use, select CUDA or not
tensortype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
tensortype_cpu = torch.FloatTensor


# Force batchsize == 1
args.eval_batch_size = 1
if args.eval_batch_size != 1:
    raise NotImplementedError('Only a batch size of 1 is implemented for now, got %s'
                              % args.eval_batch_size)

# Check we are not overwriting a checkpoint without resume from it
if args.save and os.path.isfile(args.save) and \
        not (args.resume and args.resume == args.save):
    print("E: Don't overwrite a checkpoint without resuming from it (if you want that, remove it manually).")
    exit(1)

# Create directory for checkpoint to be saved
if args.save:
    os.makedirs(os.path.split(args.save)[0], exist_ok=True)

try:
    height, width = parse('{}x{}', args.imgsize)
    height, width = int(height), int(width)
except TypeError as e:
    print("\__  E: The input --imgsize must be in format WxH, got '{}'".format(args.imgsize))
    exit(-1)

# Set seeds
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Visdom setup
viz = visdom.Visdom(env=args.env_name)
viz_train_input_win, viz_val_input_win = None, None
viz_train_loss_win, viz_val_loss_win = None, None
viz_train_gt_win, viz_val_gt_win = None, None
viz_train_est_win, viz_val_est_win = None, None

# Data loading code
trainset = CSVDataset(args.train_dir,
                      transforms=transforms.Compose([
                          RandomHorizontalFlipImageAndLabel(p=0.5),
                          RandomVerticalFlipImageAndLabel(p=0.5),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5)),
                      ]),
                      max_dataset_size=args.max_trainset_size,
                      tensortype=tensortype_cpu)
trainset_loader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.nThreads)
if args.val_dir:
    valset = CSVDataset(args.val_dir,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)),
                        ]),
                        max_dataset_size=args.max_valset_size,
                        tensortype=tensortype_cpu)
    valset_loader = DataLoader(valset,
                               batch_size=args.eval_batch_size,
                               shuffle=True,
                               num_workers=args.nThreads)

# Model
print('Building network... ', end='')
model = unet_model.UNet(3, 1,
                        height=height, width=width,
                        tensortype=tensortype)
print('DONE')
print(model)
model = nn.DataParallel(model)
if args.cuda:
    model.cuda()

# Loss function
l1_loss = nn.L1Loss()
criterion_training = losses.WeightedHausdorffDistance(height=height, width=width,
                                                      return_2_terms=True)

# Optimization strategy
optimizer = optim.SGD(model.parameters(),
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

    for batch_idx, (data, dictionary) in enumerate(trainset_loader):
        # === TRAIN ===

        # Set the module in training mode
        model.train()

        # Pull info from this sample image
        target_locations = dictionary['plant_locations']
        target_count = dictionary['plant_count']

        # We cannot deal with images with 0 plants (CD is not defined)
        if target_count[0][0] == 0:
            continue

        if args.cuda:
            data, target_locations, target_count = data.cuda(
            ), target_locations.cuda(), target_count.cuda()
        data, target_locations, target_count = Variable(
            data), Variable(target_locations), Variable(target_count)

        # One training step
        optimizer.zero_grad()
        est_map, est_count = model.forward(data)
        est_map = est_map.squeeze()
        term1, term2 = criterion_training.forward(est_map, target_locations)
        term3 = l1_loss.forward(est_count, target_count) / target_count
        term3 = term3.squeeze()
        loss = term1 + term2 + term3
        loss.backward()
        optimizer.step()

        # Log training error
        if time.time() > tic_train + args.log_interval:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                args.batch_size, len(trainset_loader.dataset),
                100. * batch_idx / len(trainset_loader), loss.data[0]))
            tic_train = time.time()

            # Send training loss to Visdom
            win_train_loss = viz.updateTrace(Y=torch.cat([term1, term2, term3, loss / 3]).view(1, -1).data.cpu(),
                                             X=torch.Tensor(
                                                 [it_num]).repeat(1, 4),
                                             opts=dict(title='Training',
                                                       legend=[
                                                           'Term 1', 'Term 2', 'Term3', 'Sum/3'],
                                                       ylabel='Loss', xlabel='Iteration'),
                                             append=True,
                                             win='0')
            if win_train_loss == 'win does not exist':
                win_train_loss = viz.line(Y=torch.cat([term1, term2, term3, loss / 3]).view(1, -1).data.cpu(),
                                          X=torch.Tensor(
                                              [it_num]).repeat(1, 4),
                                          opts=dict(title='Training',
                                                    legend=[
                                                        'Term 1', 'Term 2', 'Term3', 'Sum/3'],
                                                    ylabel='Loss', xlabel='Iteration'),
                                          win='0')

            # Send input image to Visdom
            viz.image(((data.data + 1) / 2.0 * 255.0).squeeze().cpu().numpy(),
                      opts=dict(title='(Training) Input'),
                      win=1)
            # Send estimated image to Visdom
            viz.image(est_map.data.unsqueeze(0).cpu().numpy(),
                      opts=dict(title='(Training) U-Net output'),
                      win=2)
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

    # At the end of each epoch, validate + save checkpoint if validation error decreased
    if not args.val_dir or not valset_loader or len(valset_loader) == 0:
        epoch += 1
        continue

    # === VALIDATION ===
    print("\nValidating... ")

    # Set the module in evaluation mode
    model.eval()

    judge = Judge(r=args.radius)
    sum_term1 = 0
    sum_term2 = 0
    sum_term3 = 0
    sum_loss = 0
    sum_ahd = 0
    for batch_idx, (data, dictionary) in tqdm(enumerate(valset_loader), total=len(valset_loader)):

        # Pull info from this sample image
        target_locations = dictionary['plant_locations']
        target_count = dictionary['plant_count']

        # We cannot deal with images with 0 plants (CD is not defined)
        if target_count[0][0] == 0:
            continue

        if args.cuda:
            data, target_locations, target_count = data.cuda(
            ), target_locations.cuda(), target_count.cuda()
        data, target_locations, target_count = Variable(data, volatile=True), Variable(
            target_locations, volatile=True), Variable(target_count, volatile=True)

        # Feed-forward
        est_map, est_count = model.forward(data)
        est_map = est_map.squeeze()
        est_count = est_count.squeeze()
        target_locations = target_locations.squeeze()

        # The 3 terms
        term1, term2 = criterion_training.forward(est_map, target_locations)
        term3 = l1_loss.forward(est_count, target_count) / target_count
        term3 = term3.squeeze()
        sum_term1 += term1
        sum_term2 += term2
        sum_term3 += term3
        sum_loss += term1 + term2 + term3

        # Validation using the Averaged Hausdorff Distance
        # The estimated map must be thresholed to obtain estimated points
        est_map_numpy = est_map.data.cpu().numpy()
        mask = cv2.inRange(est_map_numpy, 2 / 255, 1)
        coord = np.where(mask > 0)
        y = coord[0].reshape((-1, 1))
        x = coord[1].reshape((-1, 1))
        c = np.concatenate((y, x), axis=1)
        if len(c) == 0:
            ahd = criterion_training.max_dist
            centroids = []
        else:
            n_components = int(torch.round(est_count).data.cpu().numpy()[0])
            # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
            n_components = max(min(n_components, x.size), 1)
            centroids = mixture.GaussianMixture(n_components=n_components,
                                                n_init=1,
                                                covariance_type='full').\
                fit(c).means_.astype(np.int)

            target_locations = target_locations.data.cpu().numpy().reshape(-1, 2)
            ahd = losses.averaged_hausdorff_distance(
                centroids, target_locations)
        ahd = Variable(tensortype([ahd]))
        sum_ahd += ahd

        # Validation using Precision and Recall
        judge.evaluate_sample(centroids, target_locations)

        if time.time() > tic_val + args.log_interval:
            tic_val = time.time()

            # Send input image to Visdom
            viz.image(((data.data + 1) / 2.0 * 255.0).squeeze().cpu().numpy(),
                      opts=dict(title='(Validation) Input'),
                      win=5)
            # Send estimated image to Visdom
            viz.image(est_map.data.unsqueeze(0).cpu().numpy(),
                      opts=dict(title='(Validation) UNet output'),
                      win=6)
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
                image_with_x = tensortype(data.data.squeeze().size()).\
                    copy_(data.data.squeeze())
                image_with_x = ((image_with_x + 1) / 2.0 * 255.0)
                image_with_x = image_with_x.cpu().numpy()
                image_with_x = np.moveaxis(image_with_x, 0, 2).copy()
                for y, x in centroids:
                    image_with_x = cv2.circle(
                        image_with_x, (x, y), 3, [255, 0, 0], -1)
                viz.image(np.moveaxis(image_with_x, 2, 0),
                          opts=dict(
                              title='(Validation) Estimated centers @ crossings'),
                          win=8)

    avg_term1_val = sum_term1 / len(valset_loader)
    avg_term2_val = sum_term2 / len(valset_loader)
    avg_term3_val = sum_term3 / len(valset_loader)
    avg_loss_val = sum_loss / len(valset_loader)
    avg_ahd_val = sum_ahd / len(valset_loader)
    prec, rec = judge.get_p_n_r()
    prec, rec = Variable(tensortype([prec])), Variable(tensortype([rec]))

    # Send validation loss to Visdom
    y = torch.stack((avg_term1_val,
                     avg_term2_val,
                     avg_term3_val,
                     avg_loss_val / 3,
                     avg_ahd_val,
                     prec,
                     rec)).view(1, -1).cpu().data.numpy()
    x = tensortype([epoch]).repeat(1, 7).cpu().numpy()
    win_val_loss = viz.updateTrace(Y=y,
                                   X=x,
                                   opts=dict(title='Validation',
                                             legend=['Term 1', 'Term 2',
                                                     'Term 3', 'Sum/3',
                                                     'AHD', 'Precision', 'Recall'],
                                             ylabel='Loss',
                                             xlabel='Epoch'),
                                   append=True,
                                   win='4')
    if win_val_loss == 'win does not exist':
        win_val_loss = viz.line(Y=y,
                                X=x,
                                opts=dict(title='Validation',
                                          legend=['Term 1', 'Term 2',
                                                  'Term 3', 'Sum/3', 'AHD', 'Precision', 'Recall'],
                                          ylabel='Loss', xlabel='Epoch'),
                                win='4')

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