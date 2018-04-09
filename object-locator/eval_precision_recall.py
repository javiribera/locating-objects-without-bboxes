import numpy as np
from sklearn.neighbors import NearestNeighbors

class Judge():
    def __init__(self, r):
        self.r = r
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n_evaluations = 0


    #evaluate the output by comparing the distance of each point to gt points
    def evaluate_sample(self, pts, gt):
        
        if len(pts) == 0:
            tp = 0
            fp = 0
            fn = len(gt)
        else:
            nbr = NearestNeighbors(n_neighbors = 1, metric='euclidean').fit(gt)
            dis, idx = nbr.kneighbors(pts)
            detected_pts = (dis[:, 0] <= self.r).astype(np.uint8)

            nbr = NearestNeighbors(n_neighbors = 1, metric='euclidean').fit(pts)
            dis, idx = nbr.kneighbors(gt)
            detected_gt = (dis[:, 0] <= self.r).astype(np.uint8)

            tp = np.sum(detected_pts)
            fp = len(pts) - tp
            fn = len(gt) - np.sum(detected_gt)

        self.tp += tp
        self.fp += fp
        self.fn += fn
        
        #print("detected: ", len(pts), " gt: ", len(gt))
        #print("tp: ", tp, " fn: ", fn, " fp: ", fp)
        #print("precision", precision, "recall", recall)

    def get_p_n_r(self):
        precision = 100*self.tp / (self.tp + self.fp)
        recall = 100*self.tp / (self.tp + self.fn)
        fscore = 2 * (precision*recall/(precision+recall))

        return precision, recall, fscore


#tests
'''
a = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [100, 100]]
b = [[3, 4], [5, 6], [7, 0], [3, 5], [1, 1]]
evaluate(b, a, 2)
'''
