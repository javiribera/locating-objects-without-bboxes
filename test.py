from __future__ import print_function

import argparse
import os
import sys
import time
import shutil

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
import unet
import losses
import unet_model

# Testing settings
parser = argparse.ArgumentParser(description='Plant Location with PyTorch')
parser.add_argument('--test-dir', required=True,
                    help='Directory with testing images')
parser.add_argument('--eval-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for validation and testing')
parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                    help='number of data loading threads (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--checkpoint', default='', type=str, required=True, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--max-testset-size', type=int, default=np.inf, metavar='N',
                    help='only use the first N images of the testing dataset')
parser.add_argument('--out-dir', type=str,
                    help='path where to store the results of analyzing the test set \
                            (images and CSV file)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set seeds
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

os.makedirs(args.out_dir, exist_ok=True)

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
testset = PlantDataset(args.test_dir,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                       ]),
                       max_dataset_size=args.max_testset_size)
testset_loader = data.DataLoader(testset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.nThreads)

# Model
print('Building network... ', end='')
#model = unet.UnetGenerator(input_nc=3, output_nc=1, num_downs=8)
model = unet_model.UNet(3, 1)
print('DONE')
print(model)
model = nn.DataParallel(model)
if args.cuda:
    model.cuda()

# Loss function
l1_loss = nn.L1Loss()
criterion_training = losses.ModifiedChamferLoss(256, 256, return_2_terms=True)

# Restore saved checkpoint (model weights + epoch + optimizer state)
print("Loading checkpoint '{}' ...".format(args.checkpoint))
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch']
    lowest_avg_loss_val = checkpoint['lowest_avg_loss_val']
    model.load_state_dict(checkpoint['model'])
    print("╰─ loaded checkpoint '{}' (now on epoch {})"
          .format(args.checkpoint, checkpoint['epoch']))
else:
    print("╰─ E: no checkpoint found at '{}'".format(args.checkpoint))
    exit(-1)

tic = time.time()


# === Testing ===
print("\Testing... ")

# Empty output CSV
df_out = pd.DataFrame(columns=['plant_count'])

# Set the module in evaluation mode
model.eval()

sum_loss = 0
for batch_idx, (data, dictionary) in tqdm(enumerate(testset_loader), total=len(testset_loader)):

    # Pull info from this sample image
    gt_plant_locations = [eval(el) for el in dictionary['plant_locations']]
    target_n_plants = dictionary['plant_count']
    # We cannot deal with images with 0 plants (CD is not defined)
    if any(len(target_one_img) == 0 for target_one_img in gt_plant_locations):
        continue

    target = gt_plant_locations

    # Prepare data and target
    data, target, target_n_plants = data.type(
        torch.FloatTensor), torch.FloatTensor(target), target_n_plants.type(torch.FloatTensor)
    if args.cuda:
        data, target, target_n_plants = data.cuda(), target.cuda(), target_n_plants.cuda()
    data, target, target_n_plants = Variable(data, volatile=True), Variable(
        target, volatile=True), Variable(target_n_plants, volatile=True)

    # One forward
    est_map, est_n_plants = model.forward(data)
    est_map = est_map.squeeze()
    term1, term2 = criterion_training.forward(est_map, target)
    term3 = l1_loss.forward(est_n_plants,
                            target_n_plants.type(torch.cuda.FloatTensor))/ \
        target_n_plants.type(torch.cuda.FloatTensor)
    loss = term1 + term2 + term3

    sum_loss += loss

    # Save estimation to disk and append to CSV
    tv.utils.save_image(est_map.data, os.path.join(args.out_dir, dictionary['filename'][0]))
    df = pd.DataFrame(data=[est_n_plants.data.cpu().numpy()[0]],
                      index=[dictionary['filename'][0]],
                      columns=['plant_count'])
    df_out = df_out.append(df)
    
avg_loss_test = sum_loss / len(testset_loader)
avg_loss_test_float = avg_loss_test.data.cpu().numpy()[0]

# Write CSV to disk
df_out.to_csv(os.path.join(args.out_dir, 'estimations.csv'))

print('╰─ Average Loss for all the testing set: {:.4f}'.format(avg_loss_test_float))
print('It took %s seconds to evaluate all the testing set.' % int(time.time() - tic))
