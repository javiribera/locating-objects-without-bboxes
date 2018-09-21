import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import argparse

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
parser.add_argument('--th',
                    type=float,
                    required=True,
                    help='Detection threshold tau. The closest to this value will be used.')
parser.add_argument('--rs',
                    type=str,
                    required=True,
                    help='List of values, each with different colors in the scatter plot. '
                         'Maximum distance to consider a True Positive. '
                         'The closest to this value will be used.')
args = parser.parse_args()


figs = metrics.make_metric_plots(csv_path=csv,
                                 th=args.th,
                                 rs=args.rs,
                                 title=args.title)

for key, fig in figs:
    # Save to disk
    fig.savefig(os.path.join(args.out, f'{fig}.png'))

