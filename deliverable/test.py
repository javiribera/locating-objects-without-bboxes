import argparse
import os
import sys
import time
import shutil

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage.io
import torch
import utils
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
from torchvision.models import inception_v3
from sklearn import mixture
import losses
import unet_model
from eval_precision_recall import Judge

# Testing settings
parser = argparse.ArgumentParser(description='Plant Location with PyTorch',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test-dir', required=True,
                    help='REQUIRED. Directory with test images.\n')
# parser.add_argument('--eval-batch-size', type=int, default=1, metavar='N',
# help='Input batch size.')
parser.add_argument('--checkpoint', type=str, required=True, metavar='PATH',
                    help='REQUIRED. Checkpoint with the CNN model.\n')
parser.add_argument('--out-dir', type=str, required=True,
                    help='REQUIRED. Directory where results will be stored (images+CSV).')
parser.add_argument('--radius', type=int, default=5, metavar='R',
                    help='Detections at dist <= R to a GT pt are True Positives.')
parser.add_argument('--paint', default=True, action="store_true",
                    help='Paint a red circle at each of the estimated locations.')
parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                    help='Number of data loading threads.')
parser.add_argument('--no-gpu', '--no-cuda', action='store_true', default=False,
                    help='Use CPU only, no GPU.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed.')
parser.add_argument('--max-testset-size', type=int, default=np.inf, metavar='N',
                    help='Only use the first N images of the testing dataset.')
parser.add_argument('--n-points', type=int, default=None, metavar='N',
                    help='If you know the exact number of points in the image, then set it. '
                    'Otherwise it will be estimated by adding a L1 cost term.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set seeds
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# Create output directories
os.makedirs(os.path.join(args.out_dir, 'painted'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map_thresholded'), exist_ok=True)


class CSVDataset(data.Dataset):
    def __init__(self, directory, transform=None, max_dataset_size=np.inf):
        """CSVDataset.
        The sample images of this dataset must be all inside one directory.
        Inside the same directory, there must be one CSV file.
        This file must contain one row per image.
        It can containas many columns as wanted, i.e, filename, count...

        :param directory: Directory with all the images and the CSV file.
        :param transform: Transform to be applied to each image.
        :param max_dataset_size: Only use the first N images in the directory.
        """

        # Get groundtruth from CSV file
        csv_filename = None
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                csv_filename = filename
                break
        if csv_filename is None:
            raise ValueError(
                'The root directory %s does not have a CSV file with groundtruth' % directory)
        self.csv_df = pd.read_csv(os.path.join(directory, csv_filename))

        # Make the dataset smaller
        self.csv_df = self.csv_df[0:min(len(self.csv_df), max_dataset_size)]

        self.root_dir = directory
        self.transform = transform

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        """Get one element of the dataset.
        Returns a tuple. The first element is the image.
        The second element is a dictionary where the keys are the columns of the CSV.

        :param idx: Index of the image in the dataset to get.
        """
        img_path = os.path.join(self.root_dir, self.csv_df.ix[idx, 0])
        img = skimage.io.imread(img_path)
        dictionary = dict(self.csv_df.ix[idx])

        if self.transform:
            transformed = self.transform(img)
        else:
            transformed = img

        return (transformed, dictionary)


# Force batchsize == 1
args.eval_batch_size = 1
if args.eval_batch_size != 1:
    raise NotImplementedError('Only a batch size of 1 is implemented for now, got %s'
                              % args.eval_batch_size)

# Data loading code
testset = CSVDataset(args.test_dir,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)),
                     ]),
                     max_dataset_size=args.max_testset_size)
testset_loader = data.DataLoader(testset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.nThreads)

# Loss function
l1_loss = nn.L1Loss()
criterion_training = losses.ModifiedChamferLoss(height=256, width=256,
                                                return_2_terms=True)

# Restore saved checkpoint (model weights + epoch)
print("Loading checkpoint '{}' ...".format(args.checkpoint))
if os.path.isfile(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch']
    # Model
    if args.n_points is None:
        if 'n_points' not in checkpoint:
            # Model will also estimate # of points
            model = unet_model.UNet(3, 1, None)
        else:
            # The checkpoint tells us the # of points to estimate
            model = unet_model.UNet(3, 1, checkpoint['n_points'])
    else:
        # The user tells us the # of points to estimate
        model = unet_model.UNet(3, 1, known_n_points=args.n_points)

    # Parallelize
    model = nn.DataParallel(model)
    if args.cuda:
        model.cuda()

    # Load model in checkpoint
    model.load_state_dict(checkpoint['model'])
    print("╰─ loaded checkpoint '{}' (now on epoch {})"
          .format(args.checkpoint, checkpoint['epoch']))
    print(model)
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

judges = [Judge(r) for r in range(0, 16)]
sum_ahd = 0
sum_ape = 0
for batch_idx, (data, dictionary) in tqdm(enumerate(testset_loader),
                                          total=len(testset_loader)):

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
    target = target.squeeze()

    ape = 100 * l1_loss.forward(est_n_plants, target_n_plants) / \
        target_n_plants.type(torch.cuda.FloatTensor)
    ape = ape.data.cpu().numpy()[0]
    sum_ape += ape

    # Save estimated map to disk
    tv.utils.save_image(est_map.data,
                        os.path.join(args.out_dir, 'est_map', dictionary['filename'][0]))

    # Evaluation using the Averaged Hausdorff Distance
    # The estimated map must be thresholded to obtain estimated points
    est_map_numpy = est_map.data.cpu().numpy()
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
        n_components = int(torch.round(est_n_plants).data.cpu().numpy()[0])
        # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
        n_components = max(min(n_components, x.size), 1)
        centroids = mixture.GaussianMixture(n_components=n_components,
                                            n_init=1,
                                            covariance_type='full').\
            fit(c).means_.astype(np.int)

        target = target.data.cpu().numpy().reshape(-1, 2)
        ahd = losses.averaged_hausdorff_distance(centroids, target)

    sum_ahd += ahd

    # Validation using Precision and Recall
    for judge in judges:
        judge.evaluate_sample(centroids, target)

    # Save thresholded map to disk
    cv2.imwrite(os.path.join(args.out_dir, 'est_map_thresholded', dictionary['filename'][0]),
                mask)

    if args.paint:
        # Paint a circle in the original image at the estimated location
        image_with_x = torch.cuda.FloatTensor(data.data.squeeze().size()).\
            copy_(data.data.squeeze())
        image_with_x = ((image_with_x + 1) / 2.0 * 255.0)
        image_with_x = image_with_x.cpu().numpy()
        image_with_x = np.moveaxis(image_with_x, 0, 2).copy()
        for y, x in centroids:
            image_with_x = cv2.circle(image_with_x, (x, y), 3, [255, 0, 0], -1)
        # Save original image with circle to disk
        image_with_x = image_with_x[:, :, ::-1]
        cv2.imwrite(os.path.join(args.out_dir, 'painted', dictionary['filename'][0]),
                    image_with_x)

    df = pd.DataFrame(data=[est_n_plants.data.cpu().numpy()[0]],
                      index=[dictionary['filename'][0]],
                      columns=['plant_count'])
    df_out = df_out.append(df)

avg_ahd = sum_ahd / len(testset_loader)
mape = sum_ape / len(testset_loader)

# Write CSV to disk
df_out.to_csv(os.path.join(args.out_dir, 'estimations.csv'))

print('╰─ Average AHD for all the testing set: {:.4f}'.format(avg_ahd))
print('╰─ Accuracy for all the testing set, r=0, ..., 15')
for judge in judges:
    acc, _ = judge.get_p_n_r()
    print(acc)
print('╰─ MAPE for all the testing set: {:.4f} %'.format(mape))
print('It took %s seconds to evaluate all the testing set.' %
      int(time.time() - tic))
