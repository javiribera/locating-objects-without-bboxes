import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Make a Precision and Recall plot from the CSV result',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('csv',
                    help='CSV file with the precision and recall results.')
parser.add_argument('out',
                    help='Output plot image.')
parser.add_argument('--title',
                    default='',
                    help='Title of the plot in the figure.')
args = parser.parse_args()

# Data extraction
df = pd.read_csv(args.csv)
precision = df.precision.values
recall = df.recall.values
r = df.r.values

# Create the figure for "Crowd" Dataset
plt.ioff()
fig, ax = plt.subplots()
precision = ax.plot(r, precision, 'r--',label='Precision')
recall = ax.plot(r, recall,  'b:',label='Recall')
ax.legend()
ax.set_ylabel('%')
ax.set_xlabel(r'$r$ (in pixels)')
ax.grid(True)
plt.title(args.title)

# Save to disk
fig.savefig(args.out)

