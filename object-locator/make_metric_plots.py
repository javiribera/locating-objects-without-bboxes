# Copyright &copyright 2018 The Board of Trustees of Purdue University.
# All rights reserved.
# 
# This source code is not to be distributed or modified
# without the written permission of Edward J. Delp at Purdue University
# Contact information: ace@ecn.purdue.edu
# =====================================================================

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

