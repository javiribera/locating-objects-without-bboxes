import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import losses


class Judge():
    """
    A Judge computes the following metrics:
        (Location metrics)
        - Precision
        - Recall
        - Fscore
        - Mean Average Hausdorff Distance
        (Count metrics)
        - Mean Error
        - Mean Absolute Error
        - Mean Percent Error
        - Mean Absolute Percent Error
        - Mean Squared Error
        - Root Mean Squared Error
    """

    def __init__(self, r):
        """
        Create a Judge that will compute metrics with a particular r
         (r is only used to compute Precision, Recall, and Fscore).

        :param r: If an estimated point and a ground truth point 
                  are at a distance <= r, then a True Positive is counted.
        """
        # Location metrics
        self.r = r
        self.tp = 0
        self.fp = 0
        self.fn = 0

        # Internal variables
        self._sum_ahd = 0
        self._sum_e = 0
        self._sum_pe = 0
        self._sum_ae = 0
        self._sum_se = 0
        self._sum_ape = 0
        self._n_calls_to_feed_points = 0
        self._n_calls_to_feed_count = 0

    def feed_points(self, pts, gt, max_ahd=np.inf):
        """
        Evaluate the location metrics of one set of estimations.
         This set can correspond to the estimated points and
         the groundtruthed points of one image.
         The TP, FP, FN, Precision, Recall, Fscore, and AHD will be
         accumulated into this Judge.

        :param pts: List of estmated points.
        :param gt: List of ground truth points.
        :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
        """

        if len(pts) == 0:
            tp = 0
            fp = 0
            fn = len(gt)
        else:
            nbr = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(gt)
            dis, idx = nbr.kneighbors(pts)
            detected_pts = (dis[:, 0] <= self.r).astype(np.uint8)

            nbr = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(pts)
            dis, idx = nbr.kneighbors(gt)
            detected_gt = (dis[:, 0] <= self.r).astype(np.uint8)

            tp = np.sum(detected_pts)
            fp = len(pts) - tp
            fn = len(gt) - np.sum(detected_gt)

        self.tp += tp
        self.fp += fp
        self.fn += fn

        # Evaluation using the Averaged Hausdorff Distance
        ahd = losses.averaged_hausdorff_distance(pts, gt,
                                                 max_ahd=max_ahd)
        self._sum_ahd += ahd
        self._n_calls_to_feed_points += 1

    def feed_count(self, estim_count, gt_count):
        """
        Evaluate count metrics for a count estimation.
         This count can correspond to the estimated and groundtruthed count
         of one image. The ME, MAE, MPE, MAPE, MSE, and RMSE will be updated
         accordignly.

        :param estim_count: (positive number) Estimated count.
        :param gt_count: (positive number) Groundtruthed count.
        """

        if estim_count < 0:
            raise ValueError(f'estim_count < 0, got {estim_count}')
        if gt_count < 0:
            raise ValueError(f'gt_count < 0, got {gt_count}')

        e = estim_count - gt_count
        ae = abs(e)
        if gt_count == 0:
            ape = 100*ae
            pe = 100*e
        else:
            ape = 100 * ae / gt_count
            pe = 100 * e / gt_count
        se = e**2

        self._sum_e += e
        self._sum_pe += pe
        self._sum_ae += ae
        self._sum_se += se
        self._sum_ape += ape

        self._n_calls_to_feed_count += 1

    @property
    def me(self):
        """ Mean Error (float) """
        return float(self._sum_e / self._n_calls_to_feed_count)

    @property
    def mae(self):
        """ Mean Absolute Error (positive float) """
        return float(self._sum_ae / self._n_calls_to_feed_count)

    @property
    def mpe(self):
        """ Mean Percent Error (float) """
        return float(self._sum_pe / self._n_calls_to_feed_count)

    @property
    def mape(self):
        """ Mean Absolute Percent Error (positive float) """
        return float(self._sum_ape / self._n_calls_to_feed_count)

    @property
    def mse(self):
        """ Mean Squared Error (positive float)"""
        return float(self._sum_se / self._n_calls_to_feed_count)

    @property
    def rmse(self):
        """ Root Mean Squared Error (positive float)"""
        return float(math.sqrt(self.mse))

    @property
    def mahd(self):
        """ Mean Average Hausdorff Distance (positive float)"""
        return float(self._sum_ahd / self._n_calls_to_feed_points)

    @property
    def precision(self):
        """ Precision (positive float) """
        return float(100*self.tp / (self.tp + self.fp)) \
            if self.tp > 0 else 0

    @property
    def recall(self):
        """ Recall (positive float) """
        return float(100*self.tp / (self.tp + self.fn)) \
            if self.tp > 0 else 0

    @property
    def fscore(self):
        """ F-score (positive float) """
        return float(2 * (self.precision*self.recall /
                          (self.precision+self.recall))) \
            if self.tp > 0 else 0
