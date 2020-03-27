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

import numpy as np
import os
import argparse

from parse import parse
import torch


def parse_command_args(training_or_testing):
    """
    Parse the arguments passed by the user from the command line.
    Also performs some sanity checks.

    :param training_or_testing: 'training' or 'testing'.
    Returns: args object containing the arguments
             as properties (args.argument_name)
    """

    if training_or_testing == 'training':

        # Training settings
        parser = argparse.ArgumentParser(
            description='BoundingBox-less Location with PyTorch.',
            formatter_class=CustomFormatter)
        optional_args = parser._action_groups.pop()
        required_args = parser.add_argument_group('MANDATORY arguments')
        required_args.add_argument('--train-dir',
                                   required=True,
                                   help='Directory with training images. '
                                        'Must contain image files (any format), and '
                                        'a CSV or XML file containing groundtruth, '
                                        'as described in the README.')
        optional_args.add_argument('--val-dir',
                                   help="Directory with validation images and GT. "
                                   "If 'auto', 20%% of the training samples "
                                   "will be removed from training "
                                   "and used for validation. "
                                   "If left blank, no validation "
                                   "will be done.")
        optional_args.add_argument('--imgsize',
                                   type=str,
                                   default='256x256',
                                   metavar='HxW',
                                   help="Size of the input images "
                                        "(height x width).")
        optional_args.add_argument('--batch-size',
                                   type=strictly_positive_int,
                                   default=1,
                                   metavar='N',
                                   help="Input batch size for training.")
        optional_args.add_argument('--epochs',
                                   type=strictly_positive_int,
                                   default=np.inf,
                                   metavar='N',
                                   help="Number of epochs to train.")
        optional_args.add_argument('--nThreads', '-j',
                                   default=4,
                                   type=strictly_positive_int,
                                   metavar='N',
                                   help="Number of threads to create "
					"for data loading. "
					"Must be a striclty positive int")
        optional_args.add_argument('--lr',
                                   type=strictly_positive,
                                   default=4e-5,
                                   metavar='LR',
                                   help="Learning rate (step size).")
        optional_args.add_argument('-p',
                                   type=float,
                                   default=-1,
                                   metavar='P',
                                   help="alpha in the generalized mean "
                                        "(-inf => minimum)")
        optional_args.add_argument('--no-cuda',
                                   action='store_true',
                                   default=False,
                                   help="Disables CUDA training")
        optional_args.add_argument('--no-data-augm',
                                   action='store_true',
                                   default=False,
                                   help="Disables data augmentation "
                                        "(random vert+horiz flip)")
        optional_args.add_argument('--drop-last-batch',
                                   action='store_true',
                                   default=False,
                                   help="Drop the last batch during training, "
                                        "which may be incomplete. "
                                        "If the dataset size is not "
                                        "divisible by the batch size, "
                                        "then the last batch will be smaller.")
        optional_args.add_argument('--seed',
                                   type=int,
                                   default=1,
                                   metavar='S',
                                   help="Random seed.")
        optional_args.add_argument('--resume',
                                   default='',
                                   type=str,
                                   metavar='PATH',
                                   help="Path to latest checkpoint.")
        optional_args.add_argument('--save',
                                   default='',
                                   type=str,
                                   metavar='PATH',
                                   help="Where to save the model "
                                        "after each epoch.")
        optional_args.add_argument('--log-interval',
                                   type=strictly_positive,
                                   default=3,
                                   metavar='N',
                                   help="Time to wait between every "
                                        " time the losses are printed "
                                        "(in seconds).")
        optional_args.add_argument('--max-trainset-size',
                                   type=strictly_positive_int,
                                   default=np.inf,
                                   metavar='N',
                                   help="Only use the first N "
                                        "images of the training dataset.")
        optional_args.add_argument('--max-valset-size',
                                   type=strictly_positive_int,
                                   default=np.inf,
                                   metavar='N',
                                   help="Only use the first N images "
                                        "of the validation dataset.")
        optional_args.add_argument('--val-freq',
                                   default=1,
                                   type=int,
                                   metavar='F',
                                   help="Run validation every F epochs. "
                                        "If 0, no validation will be done. "
                                        "If no validation is done, "
                                        "a checkpoint will be saved "
                                        "every F epochs.")
        optional_args.add_argument('--visdom-env',
                                   default='default_environment',
                                   type=str,
                                   metavar='NAME',
                                   help="Name of the environment in Visdom.")
        optional_args.add_argument('--visdom-server',
                                   default=None,
                                   metavar='SRV',
                                   help="Hostname of the Visdom server. "
                                        "If not provided, nothing will "
                                        "be sent to Visdom.")
        optional_args.add_argument('--visdom-port',
                                   default=8989,
                                   metavar='PRT',
                                   help="Port of the Visdom server.")
        optional_args.add_argument('--optimizer', '--optim',
                                   default='sgd',
                                   type=str.lower,
                                   metavar='OPTIM',
                                   choices=['sgd', 'adam'],
                                   help="SGD or Adam.")
        optional_args.add_argument('--replace-optimizer',
                                   action='store_true',
                                   default=False,
                                   help="Replace optimizer state "
                                        "when resuming from checkpoint. "
                                        "If True, the optimizer "
                                        "will be replaced using the "
                                        "arguments of this scripts. "
                                        "If not resuming, it has no effect.")
        optional_args.add_argument('--max-mask-pts',
                                   type=strictly_positive_int,
                                   default=np.infty,
                                   metavar='M',
                                   help="Subsample this number of points "
                                        "from the mask, so that GMM fitting "
                                        "runs faster.")
        optional_args.add_argument('--paint',
                                   default=False,
                                   action="store_true",
                                   help="Paint red circles at the "
                                        "estimated locations in validation. "
                                        "This maskes it run much slower!")
        optional_args.add_argument('--radius',
                                   type=strictly_positive,
                                   default=5,
                                   metavar='R',
                                   help="Detections at dist <= R to a GT point"
                                        "are considered True Positives.")
        optional_args.add_argument('--n-points',
                                   type=strictly_positive_int,
                                   default=None,
                                   metavar='N',
                                   help="If you know the number of points "
                                        "(e.g, just one pupil), then set it. "
                                        "Otherwise it will be estimated.")
        optional_args.add_argument('--ultrasmallnet',
                                   default=False,
                                   action="store_true",
                                   help="If True, the 5 central layers are removed,"
                                         "resulting in a much smaller UNet. "
                                         "This is used for example for the pupil dataset."
                                         "Make sure to enable this if your are restoring "
                                         "a checkpoint that was trained using this option enabled.")

        optional_args.add_argument('--lambdaa',
                                   type=strictly_positive,
                                   default=1,
                                   metavar='L',
                                   help="Weight that will increase the "
                                        "importance of estimating the "
                                        "right number of points.")
        parser._action_groups.append(optional_args)
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
                                   help='Directory with test images. '
                                        'Must contain image files (any format), and '
                                        '(optionally) a CSV or XML file containing '
                                        'groundtruth, as described in the README.')
        required_args.add_argument('--out',
                                   type=str,
                                   required=True,
                                   help='Directory where results will be stored (images+CSV).')
        optional_args.add_argument('--model',
                                   type=str,
                                   metavar='PATH',
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
                                   # default=np.linspace(0, 1, 25).tolist() + [-1, -2],
                                   default=-2,
                                   metavar='Ts',
                                   help='Detection threshold between 0 and 1. '
                                   # 'If not selected, 25 thresholds in [0, 1] will be tested. '
                                   'tau=-1 means dynamic Otsu thresholding. '
                                   'tau=-2 means Beta Mixture Model-based thresholding.')
        optional_args.add_argument('--n-points',
                                   type=int,
                                   default=None,
                                   metavar='N',
                                   help='If you know the exact number of points in the image, then set it. '
                                   'Otherwise it will be estimated by adding a L1 cost term.')
        optional_args.add_argument('--max-mask-pts',
                                   type=int,
                                   default=np.infty,
                                   metavar='M',
                                   help='Subsample this number of points from the mask, '
                                   'so GMM fitting runs faster.')
        optional_args.add_argument('--no-paint',
                                   default=False,
                                   action="store_true",
                                   help='Don\'t paint a red circle at each estimated location.')
        optional_args.add_argument('--force', '-f',
                                   default=False,
                                   action="store_true",
                                   help='Overwrite output files if they exist. '
                                   'In fact, it removes the output directory first')
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
        optional_args.add_argument('--ultrasmallnet',
                                   default=False,
                                   action="store_true",
                                   help="If True, the 5 central layers are removed,"
                                         "resulting in a much smaller UNet. "
                                         "This is used for example for the pupil dataset."
                                         "Make sure to enable this if your are restoring "
                                         "a checkpoint that was trained using this option enabled.")
        parser._action_groups.append(optional_args)
        args = parser.parse_args()

        if not args.no_cuda and not torch.cuda.is_available():
            print(
                'W: No GPU (CUDA) devices detected in your system, running with --no-gpu option...')
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        args.paint = not args.no_paint

        # String/Int -> List
        if isinstance(args.taus, (list, range)):
            pass
        elif isinstance(args.taus, str) and ',' in args.taus:
            args.taus = [float(tau)
                         for tau in args.taus.replace('[', '').replace(']', '').split(',')]
        else:
            args.taus = [float(args.taus)]

        if isinstance(args.radii, (list, range)):
            pass
        elif isinstance(args.radii, str) and ',' in args.radii:
            args.radii = [int(r) for r in args.radii.replace('[', '').replace(']', '').split(',')]
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


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    if action.default is not None and action.default != '':
                        help += ' (default: ' + str(action.default) + ')'
        help += '\n\n'

        return help


def strictly_positive_int(val):
    """Convert to a strictly positive integer."""
    val = float(val)
    if not val > 0:
        raise argparse.ArgumentTypeError("Should be strictly positive.")
    return int(val) 


def strictly_positive(val):
    """Convert to a strictly positive float."""
    val = float(val) 
    if not val > 0:
        raise argparse.ArgumentTypeError("Should be strictly positive.")
    return val

"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
