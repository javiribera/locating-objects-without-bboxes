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


import pandas as pd
import numpy as np
import sys
import ast
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import mixture

CSV_FILE = "estimations.csv"

def eval_plant_locations(estimated, gt):
    """
    Distance function between the estimated plant locations and the ground
    truth.
    This function is a symmetric function which parameter is the estimated
    plant locations and which is the ground truth should not matter.
    The returned value is guaranteed to be always positive,
    and is only zero if both lists are exactly equal.

    :param estimated: List of (x, y) or (y,x) plant locations.
    :param gt: List of (x, y) or (y, x) plant locations.
    :return: Distance between two sets.
    """

    estimated = np.array(estimated)
    gt = np.array(gt)

    # Check dimension 
    assert estimated.ndim == gt.ndim == 2, \
    'Both estimated and GT plant locations must be 2D, i.e, (x, y) or (y, x)'
    
    d2_matrix = pairwise_distances(estimated, gt, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
          np.average(np.min(d2_matrix, axis=1))

    return res

def processImg(image, n, GMM=False):
    #extract mask from the image
    mask = cv2.inRange(image, (5,5,5), (255,255,255))
    coord = np.where(mask > 0)
    y = coord[0].reshape((-1, 1))
    x = coord[1].reshape((-1, 1))
    
    c = np.concatenate((y, x), axis=1)

    if GMM:
        gmm = mixture.GaussianMixture(n_components=n, n_init=1, covariance_type='full').fit(c)
        return gmm.means_.astype(np.int)
    
    else:
        
        #find kmean cluster
        kmeans = KMeans(n_clusters=n, random_state=0).fit(c)
        return kmeans.cluster_centers_

def processCSV(csvfile):

    df = pd.read_csv(csvfile)
    res_array = []
    for i in range(len(df.iloc[:])):
        filename = df.iloc[:, 1][i]

        plant_count = df.iloc[:, 2][i]
        plant_count = float(plant_count.split('\n')[1].strip())

        gt = df.iloc[:, 3][i]
        gt = ast.literal_eval(gt)

        image = cv2.imread(filename)
        detected = processImg(image, int(plant_count), GMM=True)

        res = eval_plant_locations(detected, gt)
        res_array.append(res)
        print(res)
        break
    return res_array


#Note the script needs to be put into the data directory with the CSV file
res = processCSV(CSV_FILE)


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
