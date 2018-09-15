import torch
import numpy as np

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

