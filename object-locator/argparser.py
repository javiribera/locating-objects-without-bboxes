# Copyright &copyright 2018 The Board of Trustees of Purdue University.
# All rights reserved.
# 
# This source code is not to be distributed or modified
# without the written permission of Edward J. Delp at Purdue University
# Contact information: ace@ecn.purdue.edu
# =====================================================================

import numpy as np
import os
import argparse

from parse import parse
import torch


def parse_command_args(training_or_testing):
    """
    Parse the arguments passed by the user from the command line.
    Also performs some sanity checks.

    :param training_or_testing: 'training' or 'testing' parameters.
    Returns: args object containing the arguments as properties (args.argument_name) 
    """

    if training_or_testing == 'training':

        # Training settings
        parser = argparse.ArgumentParser(
            description='BoundingBox-less Location with PyTorch',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--train-dir',
                            required=True,
                            help='Directory with training images.')
        parser.add_argument('--val-dir',
                            help='Directory with validation images. '
                                 'If left blank no validation will be done. '
                                 'If not provided, will not do validation')
        parser.add_argument('--imgsize',
                            type=str,
                            default='256x256',
                            metavar='HxW',
                            help='Size of the input images (height x width).')
        parser.add_argument('--batch-size',
                            type=int,
                            default=1,
                            metavar='N',
                            help='input batch size for training')
        parser.add_argument('--epochs',
                            type=int,
                            default=np.inf,
                            metavar='N',
                            help='number of epochs to train')
        parser.add_argument('--nThreads', '-j',
                            default=4,
                            type=int,
                            metavar='N',
                            help='Number of data loading threads')
        parser.add_argument('--lr',
                            type=float,
                            default=4e-5,
                            metavar='LR',
                            help='learning rate (default: 1e-5)')
        parser.add_argument('--no-cuda',
                            action='store_true',
                            default=False,
                            help='disables CUDA training')
        parser.add_argument('--no-data-augm',
                            action='store_true',
                            default=False,
                            help='Disables Data Augmentation (random vert+horiz flip)')
        parser.add_argument('--drop-last-batch',
                            action='store_true',
                            default=False,
                            help='drop the last incomplete batch during training. '
                                 'If the dataset size is not divisible by '
                                 'the batch size, then the last batch will be smaller.')
        parser.add_argument('--seed',
                            type=int,
                            default=1,
                            metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--resume',
                            default='',
                            type=str,
                            metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--save',
                            default='',
                            type=str,
                            metavar='PATH',
                            help='where to save the model after each epoch')
        parser.add_argument('--log-interval',
                            type=float,
                            default=3,
                            metavar='N',
                            help='time to wait between logs of training status '
                                 '(in seconds)')
        parser.add_argument('--max-trainset-size',
                            type=int,
                            default=np.inf,
                            metavar='N',
                            help='only use the first N images of the training dataset')
        parser.add_argument('--max-valset-size',
                            type=int,
                            default=np.inf,
                            metavar='N',
                            help='only use the first N images of the validation dataset')
        parser.add_argument('--val-freq',
                            default=1,
                            type=int,
                            metavar='F',
                            help='run validation every F epochs. '
                                 'If 0, no validation will be done. '
                                 'If no validation is done, a checkpoint will be saved '
                                 'every F epochs.')
        parser.add_argument('--visdom-env',
                            default='default_environment',
                            type=str,
                            metavar='NAME',
                            help='name of the environment in Visdom')
        parser.add_argument('--visdom-server',
                            default='http://localhost',
                            type=str,
                            metavar='SRV',
                            help='Hostname of the Visdom server')
        parser.add_argument('--optimizer', '--optim',
                            default='sgd',
                            type=str.lower,
                            metavar='OPTIM',
                            choices=['sgd', 'adam'],
                            help='SGD or Adam')
        parser.add_argument('--paint',
                            default=False,
                            action="store_true",
                            help='paint red circles at estimated locations in validation. '
                                 'It takes an enormous amount of time!')
        parser.add_argument('--radius', type=int, default=5, metavar='R',
                            help='Detections at dist <= R to a GT pt are True Positives.')
        parser.add_argument('--n-points', type=int, default=None, metavar='N',
                            help='If you know the number of points (e.g, just one pupil), set it.'
                                 'Otherwise it will be estimated by adding a L1 cost term.')
        parser.add_argument('--lambdaa', type=float, default=1, metavar='L',
                            help='Weight that will multiply the regression term '
                                 'in the loss function.')

        args = parser.parse_args()

        # Force batchsize == 1 for validation
        args.eval_batch_size = 1
        if args.eval_batch_size != 1:
            raise NotImplementedError('Only a batch size of 1 is implemented for now, got %s'
                                      % args.eval_batch_size)

        # Convert to full path
        if args.save != '':
            args.save = os.path.abspath(args.save)
        if args.resume != '':
            args.resume = os.path.abspath(args.resume)

        # Check we are not overwriting a checkpoint without resuming from it
        if args.save != '' and os.path.isfile(args.save) and \
                not (args.resume and args.resume == args.save):
            print("E: Don't overwrite a checkpoint without resuming from it. "
                  "Are you sure you want to do that? "
                  "(if you do, remove it manually).")
            exit(1)

        args.cuda = not args.no_cuda and torch.cuda.is_available()

    elif training_or_testing == 'testing':

        # Testing settings
        parser = argparse.ArgumentParser(
            description='BoundingBox-less Location with PyTorch (inference/test only)',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        optional_args = parser._action_groups.pop()
        required_args = parser.add_argument_group('MANDATORY arguments')
        required_args.add_argument('--dataset',
                                   required=True,
                                   help='Directory with test images.\n')
        required_args.add_argument('--out-dir',
                                   type=str,
                                   required=True,
                                   help='Directory where results will be stored (images+CSV).')
        optional_args.add_argument('--model',
                                   type=str,
                                   metavar='PATH',
                                   default='unet_256x256_sorghum',
                                   help='Checkpoint with the CNN model.\n')
        optional_args.add_argument('--evaluate',
                                   action='store_true',
                                   default=False,
                                   help='Evaluate metrics (Precision/Recall, RMSE, MAPE, etc.)')
        optional_args.add_argument('--no-cuda', '--no-gpu',
                                   action='store_true',
                                   default=False,
                                   help='Use CPU only, no GPU.')
        optional_args.add_argument('--imgsize',
                                   type=str,
                                   default='256x256',
                                   metavar='HxW',
                                   help='Size of the input images (heightxwidth).')
        optional_args.add_argument('--radii',
                                   type=str,
                                   default=range(0, 15 + 1),
                                   metavar='Rs',
                                   help='Detections at dist <= R to a GT pt are True Positives.'
                                        'If not selected, R=0, ..., 15 will be tested.')
        optional_args.add_argument('--taus',
                                   type=str,
                                   default=np.linspace(0, 1, 100).tolist() + [-1],
                                   metavar='Ts',
                                   help='Detection threshold. '
                                        'If not selected, 100 thresholds in [0, 1] will be tested. '
                                        'tau=-1 means dynamic Otsu thresholding.')
        optional_args.add_argument('--n-points',
                                   type=int,
                                   default=None,
                                   metavar='N',
                                   help='If you know the exact number of points in the image, then set it. '
                                   'Otherwise it will be estimated by adding a L1 cost term.')
        optional_args.add_argument('--no-paint',
                                   default=False,
                                   action="store_true",
                                   help='Don\'t paint a red circle at each estimated location.')
        optional_args.add_argument('--seed',
                                   type=int,
                                   default=0,
                                   metavar='S',
                                   help='Random seed.')
        optional_args.add_argument('--max-testset-size',
                                   type=int,
                                   default=np.inf,
                                   metavar='N',
                                   help='Only use the first N images of the testing dataset.')
        optional_args.add_argument('--nThreads', '-j',
                                   default=4,
                                   type=int,
                                   metavar='N',
                                   help='Number of data loading threads.')
        parser._action_groups.append(optional_args)
        args = parser.parse_args()

        if not args.no_cuda and not torch.cuda.is_available():
            print(
                'W: No GPU (CUDA) devices detected in your system, running with --no-gpu option...')
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        args.paint = not args.no_paint

        # String/Int -> List
        if isinstance(args.taus, list):
            pass
        elif isinstance(args.taus, str) and ',' in args.taus:
            args.taus = [float(tau) for tau in args.taus.split(',')]
        else:
            args.taus = [int(args.taus)]

        if isinstance(args.radii, list):
            pass
        elif isinstance(args.radii, str) and ',' in args.radii:
            args.radii = [int(r) for r in args.radii.split(',')]
        else:
            args.radii = [int(args.radii)]


    else:
        raise ValueError('Only \'training\' or \'testing\' allowed, got %s'
                         % training_or_testing)

    # imgsize -> height x width
    try:
        args.height, args.width = parse('{}x{}', args.imgsize)
        args.height, args.width = int(args.height), int(args.width)
    except TypeError as e:
        print("\__  E: The input --imgsize must be in format WxH, got '{}'".format(args.imgsize))
        exit(-1)

    return args
