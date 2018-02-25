from __future__ import print_function

import cv2
import argparse
import os
import sys
import time
import shutil
from itertools import chain
from tqdm import tqdm

import numpy as np
import pandas as pd
import skimage.io
import torch
import torch.optim as optim
import visdom
import skimage.draw
import utils
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
from torchvision.models import inception_v3
from sklearn import mixture
import unet_pix2pix
import losses
import unet_model
from eval_precision_recall import Judge


# Training settings
parser = argparse.ArgumentParser(description='Plant Location with PyTorch')
parser.add_argument('--train-dir', required=True,
                    help='Directory with training images')
parser.add_argument('--val-dir', 
                    help='Directory with validation images. If left blank no validation will be done.')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--eval-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=np.inf, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                    help='number of data loading threads (default: 4)')
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
                    help='Paint red circles at estimated locations in Validation? '\
                            'It takes an enormous amount of time!')
parser.add_argument('--radius', type=int, default=5, metavar='R',
                    help='Default radius to consider a object detection as "match".')
parser.add_argument('--n-points', type=int, default=None, metavar='N',
                    help='If you know the number of points (e.g, just one pupil), set it.' \
                          'Otherwise it will be estimated by adding a L1 cost term.')
parser.add_argument('--lambdaa', type=float, default=1, metavar='L',
                    help='Weight that will multiply the MAPE term in the loss function.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Check we are not overwriting a checkpoint without resume from it
if args.save and os.path.isfile(args.save) and \
        not (args.resume and args.resume == args.save):
    print("E: Don't overwrite a checkpoint without resuming from it (if you want that, remove it manually).")
    exit(1)

# Create directory for checkpoint to be saved
if args.save:
    os.makedirs(os.path.split(args.save)[0], exist_ok=True)


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


class PlantDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, max_dataset_size=np.inf):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            max_dataset_size: If the dataset is bigger than this integer,
                              ignore additional samples.
        """

        # Get groundtruth from CSV file
        csv_filename = None
        for filename in os.listdir(root_dir):
            if filename.endswith('.csv'):
                csv_filename = filename
                break
        if csv_filename is None:
            raise ValueError(
                'The root directory %s does not have a CSV file with groundtruth' % root_dir)
        self.csv_df = pd.read_csv(os.path.join(root_dir, csv_filename))

        # Make the dataset smaller
        self.csv_df = self.csv_df[0:min(len(self.csv_df), max_dataset_size)]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv_df.ix[idx, 0])
        img = skimage.io.imread(img_path)
        dictionary = dict(self.csv_df.ix[idx])

        if self.transform:
            transformed = self.transform(img)
        else:
            transformed = img

        return (transformed, dictionary)


# Data loading code
trainset = PlantDataset(args.train_dir,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)),
                        ]),
                        max_dataset_size=args.max_trainset_size)
trainset_loader = data.DataLoader(trainset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.nThreads)
if args.val_dir:
    valset = PlantDataset(args.val_dir,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)),
                          ]),
                          max_dataset_size=args.max_valset_size)
    valset_loader = data.DataLoader(valset,
                                    batch_size=args.eval_batch_size,
                                    shuffle=True,
                                    num_workers=args.nThreads)

# Model
print('Building network... ', end='')
#model = unet.UnetGenerator(input_nc=3, output_nc=1, num_downs=8)
model = unet_model.UNet(3, 1, known_n_points=args.n_points)
print('DONE')
print(model)
model = nn.DataParallel(model)
if args.cuda:
    model.cuda()

# Loss function
l1_loss = nn.L1Loss()
chamfer_loss = losses.ModifiedChamferLoss(256, 256, return_2_terms=True)
criterion_training = chamfer_loss

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
        print("╰─ loaded checkpoint '{}' (now on epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("╰─ E: no checkpoint found at '{}'".format(args.resume))
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
        gt_plant_locations = [eval(el) for el in dictionary['plant_locations']]
        target_n_plants = dictionary['plant_count']
        # We cannot deal with images with 0 plants (CD is not defined)
        if any(len(target_one_img) == 0 for target_one_img in gt_plant_locations):
            continue


        if criterion_training is chamfer_loss:
            target = gt_plant_locations
        else:
            target = dots_img_tensor

        # Prepare data and target
        data, target, target_n_plants = data.type(
            torch.FloatTensor), torch.FloatTensor(target), target_n_plants.type(torch.FloatTensor)
        if args.cuda:
            data, target, target_n_plants = data.cuda(), target.cuda(), target_n_plants.cuda()
        data, target, target_n_plants = Variable(
            data), Variable(target), Variable(target_n_plants)

        # One training step
        optimizer.zero_grad()
        est_map, est_n_plants = model.forward(data)
        est_map = est_map.squeeze()
        term1, term2 = criterion_training.forward(est_map, target)
        term3 = l1_loss.forward(est_n_plants, target_n_plants) / \
            target_n_plants.type(torch.cuda.FloatTensor)
        term3 *= args.lambdaa
        loss = term1 + term2 + term3
        loss.backward()
        optimizer.step()

        # Log training error
        if time.time() > tic_train + args.log_interval:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset_loader.dataset),
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
        gt_plant_locations = [eval(el) for el in dictionary['plant_locations']]
        target_n_plants = dictionary['plant_count']
        # We cannot deal with images with 0 plants (CD is not defined)
        if any(len(target_one_img) == 0 for target_one_img in gt_plant_locations):
            continue

        if criterion_training is chamfer_loss:
            target = gt_plant_locations
        else:
            target = dots_img_tensor

        # Prepare data and target
        data, target, target_n_plants = data.type(
            torch.FloatTensor), torch.FloatTensor(target), target_n_plants.type(torch.FloatTensor)
        if args.cuda:
            data, target, target_n_plants = data.cuda(), target.cuda(), target_n_plants.cuda()
        data, target, target_n_plants = Variable(data, volatile=True), Variable(
            target, volatile=True), Variable(target_n_plants, volatile=True)

        # Feed-forward
        est_map, est_n_plants = model.forward(data)
        est_map = est_map.squeeze()
        est_n_plants = est_n_plants.squeeze()
        target = target.squeeze()

        # The 3 terms
        term1, term2 = criterion_training.forward(est_map, target)
        term3 = l1_loss.forward(est_n_plants, target_n_plants) / \
            target_n_plants.type(torch.cuda.FloatTensor)
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
            n_components = int(torch.round(est_n_plants).data.cpu().numpy()[0])
            # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
            n_components = max(min(n_components, x.size), 1)
            centroids = mixture.GaussianMixture(n_components=n_components,
                                                n_init=1,
                                                covariance_type='full').\
                fit(c).means_.astype(np.int)

            target = target.data.cpu().numpy().reshape(-1, 2)
            ahd = losses.averaged_hausdorff_distance(centroids, target)
        ahd = torch.cuda.FloatTensor([ahd])
        sum_ahd += ahd

        # Validation using Precision and Recall
        judge.evaluate_sample(centroids, target)

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
                image_with_x = torch.cuda.FloatTensor(data.data.squeeze().size()).\
                        copy_(data.data.squeeze())
                image_with_x = ((image_with_x + 1) / 2.0 * 255.0)
                image_with_x = image_with_x.cpu().numpy()
                image_with_x = np.moveaxis(image_with_x, 0, 2).copy()
                for y, x in centroids:
                    image_with_x = cv2.circle(image_with_x, (x, y), 3, [255, 0, 0], -1)
                viz.image(np.moveaxis(image_with_x, 2, 0),
                          opts=dict(title='(Validation) Estimated centers @ crossings'),
                          win=8)

    avg_term1_val = sum_term1 / len(valset_loader)
    avg_term2_val = sum_term2 / len(valset_loader)
    avg_term3_val = sum_term3 / len(valset_loader)
    avg_loss_val = sum_loss / len(valset_loader)
    avg_ahd_val = sum_ahd / len(valset_loader)
    prec, rec = judge.get_p_n_r()
    prec, rec = torch.cuda.FloatTensor([prec]), torch.cuda.FloatTensor([rec])

    # Send validation loss to Visdom
    win_val_loss = viz.updateTrace(Y=torch.stack((avg_term1_val, avg_term2_val, avg_term3_val, avg_loss_val/3, avg_ahd_val, prec, rec)).view(1, -1).data.cpu(),
                                   X=torch.Tensor([epoch]).repeat(1, 7),
                                   opts=dict(title='Validation',
                                             legend=['Term 1', 'Term 2',
                                                 'Term 3', 'Sum/3', 'AHD', 'Precision', 'Recall'],
                                             ylabel='Loss', xlabel='Epoch'),
                                   append=True,
                                   win='4')
    if win_val_loss == 'win does not exist':
        win_val_loss = viz.line(Y=torch.stack((avg_term1_val, avg_term2_val, avg_term3_val, avg_loss_val/3, avg_ahd_val, prec, rec)).view(1, -1).data.cpu(),
                                X=torch.Tensor([epoch]).repeat(1, 7),
                                opts=dict(title='Validation',
                                          legend=['Term 1', 'Term 2',
                                                 'Term 3', 'Sum/3', 'AHD', 'Precision', 'Recall'],
                                          ylabel='Loss', xlabel='Epoch'),
                                win='4')

    # If this is the best epoch (in terms of validation error)
    avg_ahd_val_float = avg_ahd_val.cpu().numpy()[0]
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
