import numpy as np
from sklearn.neighbors import NearestNeighbors


#evaluate the output by comparing the distance of each point to gt points
def evaluate(pts, gt, r):
    
    nbr = NearestNeighbors(n_neighbors = 1, metric='euclidean').fit(gt)
    dis, idx = nbr.kneighbors(pts)
    detected_pts = (dis[:, 0] <= r).astype(np.uint8)

    nbr = NearestNeighbors(n_neighbors = 1, metric='euclidean').fit(pts)
    dis, idx = nbr.kneighbors(gt)
    detected_gt = (dis[:, 0] <= r).astype(np.uint8)

    tp = np.sum(detected_pts)
    fp = len(pts) - tp
    fn = len(gt) - np.sum(detected_gt)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    #print("detected: ", len(pts), " gt: ", len(gt))
    #print("tp: ", tp, " fn: ", fn, " fp: ", fp)
    #print("precision", precision, "recall", recall)
    return precision, recall


#tests
'''
a = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [100, 100]]
b = [[3, 4], [5, 6], [7, 0], [3, 5], [1, 1]]
evaluate(b, a, 2)
'''
