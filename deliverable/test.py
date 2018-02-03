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
parser.add_argument('--paint', default=False, action="store_true",
                    help='Paint red circles at estimated locations? '
                            'It takes an enormous amount of time!')
parser.add_argument('--radius', type=int, default=5, metavar='R',
                    help='Default radius to consider a object detection as "match".')
parser.add_argument('--n-points', type=int, default=None, metavar='N',
                    help='If you know the number of points (e.g, just one pupil), set it.' \
                          'Otherwise it will be estimated by adding a L1 cost term.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set seeds
np.random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

os.makedirs(os.path.join(args.out_dir, 'painted'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map'), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'est_map_thresholded'), exist_ok=True)


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

# Loss function
l1_loss = nn.L1Loss()
criterion_training = losses.ModifiedChamferLoss(height=288, width=384,
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
            model=unet_model.UNet(3, 1, None)
        else:
            # The checkpoint tells us the # of points to estimate
            model=unet_model.UNet(3, 1, checkpoint['n_points'])
    else:
        # The user tells us the # of points to estimate
        model=unet_model.UNet(3, 1, known_n_points=args.n_points)

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

    ape = 100*l1_loss.forward(est_n_plants, target_n_plants) / \
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
        image_with_x = image_with_x[:,:,::-1]
        cv2.imwrite(os.path.join(args.out_dir, 'painted', dictionary['filename'][0]),
                    image_with_x)

    df=pd.DataFrame(data=[est_n_plants.data.cpu().numpy()[0]],
                      index=[dictionary['filename'][0]],
                      columns=['plant_count'])
    df_out=df_out.append(df)

avg_ahd = sum_ahd/len(testset_loader)
mape = sum_ape/len(testset_loader)

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
