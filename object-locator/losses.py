import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch.nn import functional as F
import os
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.kde import KernelDensity
import skimage.io
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances


def averaged_hausdorff_distance(set1, set2):
    """
    Compute the Averaged Hausdorff Distance function
     between two unordered sets of points (the function is symmetric).
     Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res


class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 height, width,
                 return_2_terms=False,
                 tensortype=torch.FloatTensor):
        """
        :param height: Number of rows in the image.
        :param width: Number of columns in the image.
        :param return_2_terms: Whether to return the 2 terms of the CD instead of their sum. Default: False.
        :param tensortype: The result will be in this Tensor type.
        """
        super(nn.Module, self).__init__()

        self.tensortype = tensortype

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = height, width
        self.max_dist = math.sqrt(height**2 + width**2)
        self.n_pixels = height * width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(height),
                                                             np.arange(width)]))
        self.all_img_locations = self.all_img_locations.type(tensortype)
        self.all_img_locations = Variable(self.all_img_locations)

        self.return_2_terms = return_2_terms

    def forward(self, prob_map, gt):
        """
        Compute the Weighted Hausdorff Distance function
         between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance. 
        """

        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []#Variable(self.tensortype(batch_size))
        terms_2 = []#Variable(self.tensortype(batch_size))
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b] 

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            d_matrix = cdist(self.all_img_locations, gt_b)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            eps = 1e-6
            alpha = 4

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + eps)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            d_div_p = torch.min((d_matrix + eps) /
                                (p_replicated**alpha + eps / self.max_dist), 0)[0]
            d_div_p = torch.clamp(d_div_p, 0, self.max_dist)
            term_2 = torch.mean(d_div_p, 0)[0]

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()

        return res
