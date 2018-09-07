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


# Data extraction
df = pd.read_csv(args.csv)

# ==== Precision and Recall as a function of R for the fixed th ====
# Find closest threshold
th_selected = df.th.values[np.argmin(np.abs(df.th.values - args.th))]
print(f'Using th={th_selected}')

# Use only a particular r
precision = df.precision.values[df.th.values == th_selected]
recall = df.recall.values[df.th.values == th_selected]
r = df.r.values[df.th.values == th_selected]

# Create the figure for "Crowd" Dataset
plt.ioff()
fig, ax = plt.subplots()
precision = ax.plot(r, precision, 'r--',label='Precision')
recall = ax.plot(r, recall,  'b:',label='Recall')
ax.legend()
ax.set_ylabel('%')
ax.set_xlabel(r'$r$ (in pixels)')
ax.grid(True)
plt.title(args.title + f' th={th_selected:4f}')

# Save to disk
fig.savefig(os.path.join(args.out, f'precision_and_recall_vs_r_th={th_selected}.png'))


# ==== Precision vs Recall ====
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
rs = [int(item) for item in args.rs.split(',')]
if len(rs) > len(colors):
    raise ValueError(f'Too many radii provided, maximum {len(colors)}')

# Create figure
fig, ax = plt.subplots()
plt.ioff()
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.grid(True)
plt.title(args.title)

for r, c in zip(rs, colors):
    # Find closest R
    r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

    # Use only a particular r
    precision = df.precision.values[df.r.values == r_selected]
    recall = df.recall.values[df.r.values == r_selected]
    th = df.th.values[df.r.values == r_selected]

    # Sort by ascending recall
    idxs = np.argsort(recall)
    recall = recall[idxs]
    precision = precision[idxs]
    th = th[idxs]

    # Plot precision vs. recall for this r
    ax.scatter(recall, precision, c=c, s=2, label=f'r={r}')

ax.legend()

# Save to disk
fig.savefig(os.path.join(args.out, 'precision_vs_recall.png'))


# ==== Precision as a function of TH for all provided R ====
# Create figure
fig, ax = plt.subplots()
plt.ioff()
ax.set_ylabel('Precision')
ax.set_xlabel(r'$\tau$')
ax.grid(True)
plt.title(args.title)

list_of_precisions = []

for r, c in zip(rs, colors):
    # Find closest R
    r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

    # Use only a particular r
    precision = df.precision.values[df.r.values == r_selected]
    list_of_precisions.append(precision)
    th = df.th.values[df.r.values == r_selected]

    # Plot precision vs th for this r
    ax.scatter(th, precision, c=c, s=2, label=f'r={r}')

ax.plot(th, np.average(np.stack(list_of_precisions), axis=0),
        'k-', label='avg')

# Invert legend order
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]

# Put legend outside the plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles, labels, bbox_to_anchor=(1, 0.5))

# Save to disk
fig.savefig(os.path.join(args.out, 'precision_vs_th.png'))


# ==== Recall as a function of TH for all provided R ====
# Create figure
fig, ax = plt.subplots()
plt.ioff()
ax.set_ylabel('Recall')
ax.set_xlabel(r'$\tau$')
ax.grid(True)
plt.title(args.title)

list_of_recalls = []

for r, c in zip(rs, colors):
    # Find closest R
    r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

    # Use only a particular r
    recall = df.recall.values[df.r.values == r_selected]
    list_of_recalls.append(recall)
    th = df.th.values[df.r.values == r_selected]

    # Plot precision vs th for this r
    ax.scatter(th, recall, c=c, s=2, label=f'r={r}')

ax.plot(th, np.average(np.stack(list_of_recalls), axis=0),
        'k-', label='avg')

# Invert legend order
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]

# Put legend outside the plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles, labels, bbox_to_anchor=(1, 0.5))

# Save to disk
fig.savefig(os.path.join(args.out, 'recall_vs_th.png'))


# ==== F-score as a function of TH for all provided R ====
# Create figure
fig, ax = plt.subplots()
plt.ioff()
ax.set_ylabel('F-score')
ax.set_xlabel(r'$\tau$')
ax.grid(True)
plt.title(args.title)

list_of_fscores = []

for r, c in zip(rs, colors):
    # Find closest R
    r_selected = df.r.values[np.argmin(np.abs(df.r.values - r))]

    # Use only a particular r
    fscore = df.fscore.values[df.r.values == r_selected]
    list_of_fscores.append(fscore)
    th = df.th.values[df.r.values == r_selected]

    # Plot precision vs th for this r
    ax.scatter(th, fscore, c=c, s=2, label=f'r={r}')

ax.plot(th, np.average(np.stack(list_of_fscores), axis=0),
        'k-', label='avg')

# Invert legend order
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]

# Put legend outside the plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(handles, labels, bbox_to_anchor=(1, 0.5))

# Save to disk
fig.savefig(os.path.join(args.out, 'fscore_vs_th.png'))

