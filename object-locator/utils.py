# Copyright &copyright 2018 The Board of Trustees of Purdue University.
# All rights reserved.
# 
# This source code is not to be distributed or modified
# without the written permission of Edward J. Delp at Purdue University
# Contact information: ace@ecn.purdue.edu
# =====================================================================

import torch
import numpy as np
import sklearn.mixture
import scipy.stats
import cv2
from . import bmm
from matplotlib import pyplot as plt
import scipy.stats

class Normalizer():
    def __init__(self, new_size_height, new_size_width):
        """
        Normalizer.
        Converts coordinates in an original image size
        to a new image size (resized/normalized).

        :param new_size_height: (int) Height of the new (resized) image size.
        :param new_size_width: (int) Width of the new (resized) image size.
        """
        new_size_height = int(new_size_height)
        new_size_width = int(new_size_width)

        self.new_size = np.array([new_size_height, new_size_width])

    def unnormalize(self, coordinates_yx_normalized, orig_img_size):
        """
        Unnormalize coordinates,
        i.e, make them with respect to the original image.

        :param coordinates_yx_normalized:
        :param orig_size: Original image size ([height, width]).
        :return: Unnormalized coordinates
        """

        orig_img_size = np.array(orig_img_size)
        assert orig_img_size.ndim == 1
        assert len(orig_img_size) == 2

        norm_factor = orig_img_size / self.new_size
        norm_factor = np.tile(norm_factor, (len(coordinates_yx_normalized),1))
        coordinates_yx_unnormalized = norm_factor*coordinates_yx_normalized

        return coordinates_yx_unnormalized

def threshold(array, tau):
    """
    Threshold an array using either hard thresholding, Otsu thresholding or beta-fitting.

    If the threshold value is fixed, this function returns
    the mask and the threshold used to obtain the mask.
    When using tau=-1, the threshold is obtained as described in the Otsu method.
    When using tau=-2, it also returns the fitted 2-beta Mixture Model.


    :param array: Array to threshold.
    :param tau: (float) Threshold to use.
                Values above tau become 1, and values below tau become 0.
                If -1, use Otsu thresholding.
		If -2, fit a mixture of 2 beta distributions, and use
		the average of the two means.
    :return: The tuple (mask, threshold).
             If tau==-2, returns the tuple (mask, otsu_tau, ((rv1, rv2), (pi1, pi2))).
             
    """
    if tau == -1:
        # Otsu thresholding
        minn, maxx = array.min(), array.max()
        array_scaled = ((array - minn)/(maxx - minn)*255) \
            .round().astype(np.uint8).squeeze()
        tau, mask = cv2.threshold(array_scaled,
                                  0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tau = minn + (tau/255)*(maxx - minn)
        # print(f'Otsu selected tau={tau_otsu}')
    elif tau == -2:
        array_flat = array.flatten()
        ((a1, b1), (a2, b2)), (pi1, pi2), niter = bmm.estimate(array_flat, list(range(2)))
        rv1 = scipy.stats.beta(a1, b1)
        rv2 = scipy.stats.beta(a2, b2)
        
        tau = rv2.mean()
        mask = cv2.inRange(array, tau, 1)

        return mask, tau, ((rv1, pi1), (rv2, pi2))
    else:
        # Thresholding with a fixed threshold tau
        mask = cv2.inRange(array, tau, 1)

    return mask, tau


class AccBetaMixtureModel():

    def __init__(self, n_components=2, n_pts=1000):
        """
        Accumulator that tracks multiple Mixture Models based on Beta distributions.
        Each mixture is a tuple (scipy.RV, weight).

        :param n_components: (int) Number of components in the mixtures.
        :param n_pts: Number of points in the x axis (values the RV can take in [0, 1]) 
        """
        self.n_components = n_components
        self.mixtures = []
        self.x = np.linspace(0, 1, n_pts)

    def feed(self, mixture):
        """
        Accumulate another mixture so that this AccBetaMixtureModel can track it.

        :param mixture: List/Tuple of mixtures, i.e, ((RV, weight), (RV, weight), ...)
        """
        assert len(mixture) == self.n_components
        
        self.mixtures.append(mixture)

    def plot(self):
        """
        Create and return plots showing a variety of stats
        of the mixtures feeded into this object.
        """
        assert len(self.mixtures) > 0

        figs = {}

        # Compute the mean of the pdf of each component
        pdf_means = [(1/len(self.mixtures))*np.clip(rv.pdf(self.x), a_min=0, a_max=50)\
                     for rv, w in self.mixtures[0]]
        for mix in self.mixtures[1:]:
            for c, (rv, w) in enumerate(mix):
                pdf_means[c] += (1/len(self.mixtures))*np.clip(rv.pdf(self.x), a_min=0, a_max=50)

        # Compute the stdev of the pdf of each component
        if len(self.mixtures) > 1:
            pdfs_sq_err_sum = [(np.clip(rv.pdf(self.x), a_min=0, a_max=50) - pdf_means[c])**2 \
                               for c, (rv, w) in enumerate(self.mixtures[0])]
            for mix in self.mixtures[1:]:
                for c, (rv, w) in enumerate(mix):
                    pdfs_sq_err_sum[c] += (np.clip(rv.pdf(self.x), a_min=0, a_max=50) - pdf_means[c])**2
            pdf_stdevs = [np.sqrt(pdf_sq_err_sum)/(len(self.mixtures) - 1) \
                          for pdf_sq_err_sum in pdfs_sq_err_sum]

        # Plot the means of the pdfs
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for c, (pdf_mean, color) in enumerate(zip(pdf_means, colors)):
            ax.plot(self.x, pdf_mean, c=color, label=f'Component #{c}')
        ax.set_title('Mean Probability Density Function\nof the fitted bimodal Beta Mixture Model')
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Probability Density')
        figs['mean_bmm'] = fig
        plt.close(fig)

        if len(self.mixtures) > 1:
            # Plot the means of the pdfs
            fig, ax = plt.subplots()
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            max_stdev = 0
            for c, (pdf_stdev, color) in enumerate(zip(pdf_stdevs, colors)):
                ax.plot(self.x, pdf_stdev, c=color, label=f'Component #{c}')
                max_stdev = max(max_stdev, max(pdf_stdev))
            ax.set_title('Standard Deviation of the\nProbability Density Functions\n'
                         'of the fitted bimodal Beta Mixture Model')
            ax.set_xlabel('Pixel value')
            ax.set_ylabel('Standard Deviation')
            ax.set_ylim([0, max_stdev])
            figs['std_bmm'] = fig
            plt.close(fig)

            # Plot the KDE of the histogram of the threshold (the mean of last RV)
            thresholds = [mix[-1][0].mean() for mix in self.mixtures]
            kde = scipy.stats.gaussian_kde(np.array(thresholds).reshape(1, -1))
            fig, ax = plt.subplots()
            ax.plot(self.x, kde.pdf(self.x))
            ax.set_title('KDE of the threshold used by method #3')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Probability Density')
            figs['kde_bmm_threshold'] = fig
            plt.close(fig)

        return figs

def cluster(array, n_clusters, max_mask_pts=np.infty):
    """
    Cluster a 2-D binary array.
    Applies a Gaussian Mixture Model on the positive elements of the array,
    and returns the number of clusters.
    
    :param array: Binary array.
    :param n_clusters: Number of clusters (Gaussians) to fit,
    :param max_mask_pts: Randomly subsample "max_pts" points
                         from the array before fitting.
    :return: Centroids in the input array.
    """

    array = np.array(array)
    
    assert array.ndim == 2

    coord = np.where(array > 0)
    y = coord[0].reshape((-1, 1))
    x = coord[1].reshape((-1, 1))
    c = np.concatenate((y, x), axis=1)
    if len(c) == 0:
        centroids = np.array([])
    else:
        # Subsample our points randomly so it is faster
        if max_mask_pts != np.infty:
            n_pts = min(len(c), max_mask_pts)
            np.random.shuffle(c)
            c = c[:n_pts]

        # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
        n_components = max(min(n_clusters, x.size), 1)
        centroids = sklearn.mixture.GaussianMixture(n_components=n_components,
                                                    n_init=1,
                                                    covariance_type='spherical').\
            fit(c).means_.astype(np.int)

    return centroids


class RunningAverage():

    def __init__(self, size):
        self.list = []
        self.size = size

    def put(self, elem):
        if len(self.list) >= self.size:
            self.list.pop(0)
        self.list.append(elem)

    def pop(self):
        self.list.pop(0)

    @property
    def avg(self):
        return np.average(self.list)

