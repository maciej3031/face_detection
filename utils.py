import os

import cv2
import numpy as np

from settings import IMG_SIZE


def jj(*args):
    return os.path.join(*args)


def open_image(full_image_name):
    return cv2.imread(full_image_name, cv2.IMREAD_GRAYSCALE)  # grey scale


def convert_to_array(img):
    return np.array(img, ndmin=3)


def resize(img, n):
    return cv2.resize(img, (n, n))


def resize_image_to_nxn_square(img, n):
    longer_side = max(img.shape)
    horizontal_padding = int((longer_side - img.shape[0]) / 2)
    vertical_padding = int((longer_side - img.shape[1]) / 2)

    new_image = cv2.copyMakeBorder(img,
                                   horizontal_padding,
                                   horizontal_padding,
                                   vertical_padding,
                                   vertical_padding,
                                   cv2.BORDER_CONSTANT)
    resized_img = resize(new_image, n)
    return resized_img


def reshape_to_on_dim_vector(array):
    return np.reshape(array, (array.shape[0], IMG_SIZE * IMG_SIZE))
