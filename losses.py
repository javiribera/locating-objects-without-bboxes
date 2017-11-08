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

"""
We recommend copying this file to any project you need.
"""


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
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


class ModifiedChamferLoss(nn.Module):
    def __init__(self, height, width, return_2_terms=False):
        """
        :param height: Number of rows in the image.
        :param width: Number of columns in the image.
        :param return_2_terms: Whether to return the 2 terms of the CD instead of their sum. Default: False.
        """
        super(nn.Module, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = height, width
        self.max_dist = math.sqrt(height**2 + width**2)
        self.n_pixels = height * width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(height),
                                                             np.arange(width)]))
        self.all_img_locations = self.all_img_locations.type(torch.FloatTensor)
        self.all_img_locations = self.all_img_locations.cuda()
        self.all_img_locations = Variable(self.all_img_locations)

        self.return_2_terms = return_2_terms

    def forward(self, prob_map, gt):
        """
        Compute the Modified Chamfer Distance function
         between the estimated probability map and ground truth points.

        :param prob_map: Tensor of the probability map of the estimation, must be between 0 and 1.
        :param gt: Tensor where each row is the (y, x), i.e, (row, col) of GT points.
        :return: Value of the Modified Chamfer Distance, or their 2 terms as a tuples.
        """
        _assert_no_grad(gt)

        assert prob_map.size()[0:2] == (self.height, self.width), \
            'You must configure the ModifiedChamferLoss with the height and width of the ' \
            'probability map that you are using, got a probability map of size (%s, %s)'\
            % prob_map.size()

        # Pairwise distances between all possible locations and the GTed locations
        gt = gt.squeeze()
        n_gt_pts = gt.size()[0]
        d2_matrix = cdist(self.all_img_locations, gt)

        # Reshape probability map as a long column vector,
        # and prepare it for multiplication
        p = prob_map.view(prob_map.nelement())
        # Think of the next line as a regular threshold at 0.5 to {0,1} (damn pytorch!)
        # Hard threshold
        # p_thresh = F.threshold(p,0.1,0)/p
        n_est_pts = p.sum()
        p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
        # p_thresh_replicated = p_thresh.view(-1, 1).repeat(1, n_gt_pts)

        eps = 1e-6

        # Modified Chamfer Loss
        term_1 = (1 / (n_est_pts + eps)) * \
            torch.sum(p * torch.min(d2_matrix, 1)[0])
        d_div_p = torch.min((d2_matrix + eps) /
                            (p_replicated**4 + eps / self.max_dist), 0)[0]
        d_div_p = torch.clamp(d_div_p, 0, self.max_dist)
        term_2 = 1 * torch.mean(d_div_p, 0)[0]

        if self.return_2_terms:
            res = (term_1, term_2)
        else:
            res = term_1 + term_2

        return res
