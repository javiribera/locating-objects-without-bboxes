import torch
import numpy as np
import sklearn.mixture
import cv2

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
    Threshold an array using either hard thresholding or Otsu thresholding.

    :param array: Array to threshold.
    :param tau: (float) Threshold to use.
                Values above tau become 1, and values below tau become 0.
                If -1, use Otsu thresholding.
    :return: Tuple, where first element is the binary mask, and the second one
             is the threshold used. When using Otsu thresholding, this threshold will be
             is obtained adaptively according to the values of the input array.
             
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
    else:
        # Thresholding with a fixed threshold tau
        mask = cv2.inRange(array, tau, 1)

    return mask, tau


def cluster(array, n_clusters):
    """
    Cluster a 2-D binary array.
    Applies a Gaussian Mixture Model on the positive elements of the array,
    and returns the number of clusters.
    
    :param array: Binary array.
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
        # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
        n_components = max(min(n_clusters, x.size), 1)
        centroids = sklearn.mixture.GaussianMixture(n_components=n_components,
                                                    n_init=1,
                                                    covariance_type='full').\
            fit(c).means_.astype(np.int)

    return centroids


