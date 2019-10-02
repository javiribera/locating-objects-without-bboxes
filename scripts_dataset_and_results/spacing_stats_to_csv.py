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


import argparse
import os
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean as distance
import statistics
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute intra-row spacing stats of a CSV. '
                    'Add mean, median, and stdev of each row. '
                    'Optional: plot histograms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_csv',
                        help='Input CSV with plant location info.')
    parser.add_argument('out_csv',
                        help='Output CSV with the added stats.')
    parser.add_argument('--hist',
                        metavar='DIR',
                        help='Directory with histograms.')
    parser.add_argument('--res',
                        metavar='DIR',
                        type=float,
                        default=1,
                        help='Resolution in centimeters.')
    args = parser.parse_args()

    # Import GT from CSV
    df = pd.read_csv(args.in_csv)

    # Store stats of each single-row plot
    means, medians, stds = [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df.index)):
        if row['locations_wrt_orthophoto'] is np.nan:
            continue
        locs = eval(row['locations_wrt_orthophoto'])

        # 1. Sort by row coordinate
        locs = sorted(locs, key=lambda x: x[0])

        # 2. Compute distances (chain-like) between plants
        dists = list(map(distance, locs[:-1], locs[1:]))

        # 3. pixels -> centimeters
        dists = [d * args.res for d in dists]

        # 4. Statistics!
        mean = statistics.mean(dists)
        median = statistics.median(dists)
        std = statistics.stdev(dists)
        means.append(mean)
        medians.append(median)
        stds.append(std)

        # 5. Put in CSV
        df.loc[idx, 'mean_intrarow_spacing_in_cm'] = mean
        df.loc[idx, 'median_intrarow_spacing_in_cm'] = median
        df.loc[idx, 'stdev_intrarow_spacing_in_cm'] = std

    # Save to disk as CSV
    df.to_csv(args.out_csv)

    if args.hist is not None:
        os.makedirs(args.hist, exist_ok=True)

        # 6. Generate nice graphs for presentation
        # Means
        fig = plt.figure()
        n, bins, patches = plt.hist(
            means, 30, normed=1, facecolor='green', alpha=0.75, label='Histogram')
        # add a 'best fit' norm line
        y = mlab.normpdf(bins, statistics.mean(means), statistics.stdev(means))
        l = plt.plot(bins, y, 'r--', linewidth=1, label='Fitted Gaussian')
        plt.xlabel('Average intra-row spacing [cm]')
        plt.ylabel('Probability')
        plt.title('Histogram of average intra-row spacing')
        plt.axis([5, 30, 0, 0.3])
        plt.grid(True)
        plt.legend()
        fig.savefig(os.path.join(
            args.hist, 'histogram_averages.png'), dpi=fig.dpi)

        # Medians
        fig = plt.figure()
        n, bins, patches = plt.hist(
            medians, 30, normed=1, facecolor='green', alpha=0.75, label='Histogram')
        # add a 'best fit' norm line
        y = mlab.normpdf(bins, statistics.mean(
            medians), statistics.stdev(medians))
        l = plt.plot(bins, y, 'r--', linewidth=1, label='Fitted Gaussian')
        plt.xlabel('Median of intra-row spacing [cm]')
        plt.ylabel('Probability')
        plt.title('Histogram of medians intra-row spacing')
        plt.axis([5, 30, 0, 0.3])
        plt.grid(True)
        plt.legend()
        fig.savefig(os.path.join(
            args.hist, 'histogram_medians.png'), dpi=fig.dpi)

        # Standard deviations
        fig = plt.figure()
        n, bins, patches = plt.hist(
            stds, 30, normed=1, facecolor='green', alpha=0.75, label='Histogram')
        # add a 'best fit' norm line
        y = mlab.normpdf(bins, statistics.mean(stds), statistics.stdev(stds))
        l = plt.plot(bins, y, 'r--', linewidth=1, label='Fitted Gaussian')
        plt.xlabel('Standard deviation of intra-row spacing [cm]')
        plt.ylabel('Probability')
        plt.title('Histogram of standard deviations of intra-row spacing')
        plt.axis([0, 25, 0, 0.3])
        plt.grid(True)
        plt.legend()
        fig.savefig(os.path.join(
            args.hist, 'histogram_stdevs.png'), dpi=fig.dpi)


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
