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


import os
import numpy as np
import pandas as pd
import argparse

from . import metrics

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Create a bunch of plot from the metrics in a CSV.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('csv',
                    help='CSV file with the precision and recall results.')
parser.add_argument('out',
                    help='Output directory.')
parser.add_argument('--title',
                    default='',
                    help='Title of the plot in the figure.')
parser.add_argument('--taus',
                    type=str,
                    required=True,
                    help='Detection threshold taus. '
                         'For each of these taus, a precision(r) and recall(r) will be created.'
                         'The closest to these values will be used.')
parser.add_argument('--radii',
                    type=str,
                    required=True,
                    help='List of values, each with different colors in the scatter plot. '
                         'Maximum distance to consider a True Positive. '
                         'The closest to this value will be used.')
args = parser.parse_args()


os.makedirs(args.out, exist_ok=True)

taus = [float(tau) for tau in args.taus.replace('[', '').replace(']', '').split(',')]
radii = [int(r) for r in args.radii.replace('[', '').replace(']', '').split(',')]

figs = metrics.make_metric_plots(csv_path=args.csv,
                                 taus=taus,
                                 radii=radii,
                                 title=args.title)

for label, fig in figs.items():
    # Save to disk
    fig.savefig(os.path.join(args.out, f'{label}.png'))


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
