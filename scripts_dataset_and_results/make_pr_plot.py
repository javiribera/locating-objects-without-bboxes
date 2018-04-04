import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Data extraction
data = pd.read_excel("/Users/dguera/Desktop/test.xlsx")
pr_crowd = data["Precision (crowd)"].as_matrix(range(len(data["Precision (crowd)"])))
r_crowd = data["Recall (crowd)"].as_matrix(range(len(data["Recall (crowd)"])))
r = data["r"].as_matrix(range(len(data["r"])))

#  Create the figure for "Crowd" Dataset
fig, ax = plt.subplots()
precision = ax.plot(r, pr_crowd, 'r--',label='Precision')
recall = ax.plot(r, r_crowd,  'b:',label='Recall')
ax.legend()
ax.set_ylabel('Percentatge')
ax.set_xlabel(r'$r$ (in pixels)')
ax.grid(True)
fig.savefig('/Users/dguera/Desktop/plot.eps')
#plt.show()
