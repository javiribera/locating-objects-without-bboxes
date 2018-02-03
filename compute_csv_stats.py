import os
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean as distance
import statistics
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

CSV_PATH = '/data/jprat/projects/phenosorg/gt/plant_locations/20170616_F41/GxE/manual20170616_F41_GNSSINS_1cm_vertices_GxEPMZ_withplants_and_plotnum.csv'
OUTPUT_CSV = '/data/jprat/projects/phenosorg/gt/plant_locations/20170616_F41/GxE/manual20170616_F41_GNSSINS_1cm_vertices_GxEPMZ_withplants_and_plotnum_withstats.csv'
OUTPUT_FIGURES_DIR = '/data/jprat/projects/phenosorg/gt/plant_locations/20170616_F41/GxE/histograms'

os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)

# Import GT from CSV
df = pd.read_csv(CSV_PATH)

# Store stats of each single-row plot
means, medians, stds = [], [], []

for idx, row in tqdm(df.iterrows(), total=len(df.index)):
    locs = eval(row['plant_locations_wrt_orthophoto'])

    # 1. Sort by row coordinate
    locs = sorted(locs, key=lambda x: x[0])
    
    # 2. Compute distances (chain-like) between plants
    dists = list(map(distance, locs[:-1], locs[1:]))

    # 3. Statistics!
    mean = statistics.mean(dists)
    median = statistics.median(dists)
    std = statistics.stdev(dists)
    means.append(mean)
    medians.append(median)
    stds.append(std)

    # 4. Put in CSV
    df.loc[idx, 'mean_intrarow_spacing'] = mean
    df.loc[idx, 'median_intrarow_spacing'] = median
    df.loc[idx, 'stdev_intrarow_spacing'] = std


# Save to disk as CSV
df.to_csv(OUTPUT_CSV)

# 5. Generate nice graphs for presentation
# Means
fig = plt.figure()
n, bins, patches = plt.hist(means, 30, normed=1, facecolor='green', alpha=0.75, label='Histogram')
# add a 'best fit' norm line
y = mlab.normpdf( bins, statistics.mean(means), statistics.stdev(means))
l = plt.plot(bins, y, 'r--', linewidth=1, label='Fitted Gaussian')
plt.xlabel('Average intra-row spacing [cm]')
plt.ylabel('Probability')
plt.title('Histogram of average intra-row spacing')
plt.axis([10, 30, 0, 0.3])
plt.grid(True)
plt.legend()
fig.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'histogram_averages.png'), dpi=fig.dpi)

# Medians
fig = plt.figure()
n, bins, patches = plt.hist(medians, 30, normed=1, facecolor='green', alpha=0.75, label='Histogram')
# add a 'best fit' norm line
y = mlab.normpdf( bins, statistics.mean(medians), statistics.stdev(medians))
l = plt.plot(bins, y, 'r--', linewidth=1, label='Fitted Gaussian')
plt.xlabel('Median of intra-row spacing [cm]')
plt.ylabel('Probability')
plt.title('Histogram of medians intra-row spacing')
plt.axis([10, 30, 0, 0.3])
plt.grid(True)
plt.legend()
fig.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'histogram_medians.png'), dpi=fig.dpi)

# Standard deviations
fig = plt.figure()
n, bins, patches = plt.hist(stds, 30, normed=1, facecolor='green', alpha=0.75, label='Histogram')
# add a 'best fit' norm line
y = mlab.normpdf( bins, statistics.mean(stds), statistics.stdev(stds))
l = plt.plot(bins, y, 'r--', linewidth=1, label='Fitted Gaussian')
plt.xlabel('Standard deviation of intra-row spacing [cm]')
plt.ylabel('Probability')
plt.title('Histogram of standard deviations of intra-row spacing')
plt.axis([0, 25, 0, 0.3])
plt.grid(True)
plt.legend()
fig.savefig(os.path.join(OUTPUT_FIGURES_DIR, 'histogram_stdevs.png'), dpi=fig.dpi)
