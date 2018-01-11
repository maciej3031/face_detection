import math

import numpy as np
# from matplotlib import pyplot as plt
from convolve import convolve


class Canny(object):
    def __init__(self, img, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal
        self.gaussian_filter = np.asarray([[1, 4, 7, 4, 1],
                                           [4, 16, 26, 16, 4],
                                           [7, 26, 41, 26, 7],
                                           [4, 16, 26, 16, 4],
                                           [4, 16, 26, 16, 4],
                                           [1, 4, 7, 4, 1]], dtype=np.int)

        self.sobel_Gx_filter = np.asarray([[-1, 0, 1],
                                           [-2, 0, 2],
                                           [-1, 0, 1]], dtype=np.int)

        self.sobel_Gy_filter = np.asarray([[-1, -2, -1],
                                           [0, 0, 0],
                                           [1, 2, 1]], dtype=np.int)

        self.array = np.asarray(img, dtype=np.int)
        self.height, self.width = self.array.shape[0], self.array.shape[1]
        self.canny_edges = np.zeros(self.array.shape)

        self.run()

    def run(self):
        array_after_gaussian_filtering = self.set_gaussian_filtered_array()
        array_after_Gx = self.set_array_after_Gx(array_after_gaussian_filtering)
        array_after_Gy = self.set_array_after_Gy(array_after_gaussian_filtering)
        array_G = self.set_array_after_G(array_after_Gx, array_after_Gy)
        array_theta = self.set_array_after_theta(array_after_Gx, array_after_Gy)
        array_nmax_suppress = self.set_array_after_suppression(array_G, array_theta)

        self.set_array_after_thresholding(array_nmax_suppress)

    def get_features(self):
        # plt.imshow(self.canny_edges)
        # plt.show()
        return self.canny_edges

    def set_gaussian_filtered_array(self):
        zero_padded_array = np.pad(self.array, 2, mode='constant')
        array_after_gaussian_filtering = convolve(zero_padded_array, self.gaussian_filter)
        return array_after_gaussian_filtering // 273

    def set_array_after_Gx(self, array_after_gaussian_filtering):
        zero_padded_array = np.pad(array_after_gaussian_filtering, 1, mode='constant')
        return convolve(zero_padded_array, self.sobel_Gx_filter)

    def set_array_after_Gy(self, array_after_gaussian_filtering):
        zero_padded_array = np.pad(array_after_gaussian_filtering, 1, mode='constant')
        return convolve(zero_padded_array, self.sobel_Gy_filter)

    def set_array_after_G(self, array_after_Gx, array_after_Gy):
        return np.sqrt(np.square(array_after_Gx) + np.square(array_after_Gy))

    def set_array_after_theta(self, array_after_Gx, array_after_Gy):
        return np.arctan2(array_after_Gy, array_after_Gx) * 180 / math.pi

    def set_array_after_suppression(self, array_G, array_theta):
        array_nmax_suppress = np.zeros(self.array.shape)
        for row in range(self.height):
            for col in range(self.width):
                if self.width - 1 > col > 0 and self.height - 1 > row > 0:
                    # 0 degrees
                    if (22.5 > array_theta[row][col] >= -22.5) or (
                            array_theta[row][col] < -157.5 or array_theta[row][col] >= 157.5):
                        if array_G[row][col] >= array_G[row][col + 1] and array_G[row][col] >= \
                                array_G[row][col - 1]:
                            array_nmax_suppress[row][col] = array_G[row][col]

                    # 45 degrees
                    if (67.5 > array_theta[row][col] >= 22.5) or (-112.5 > array_theta[row][col] >= -157.5):
                        if array_G[row][col] >= array_G[row - 1][col - 1] and array_G[row][col] >= \
                                array_G[row + 1][col + 1]:
                            array_nmax_suppress[row][col] = array_G[row][col]

                    # 90 degrees
                    if (112.5 > array_theta[row][col] >= 67.5) or (-67.5 > array_theta[row][col] >= -112.5):
                        if array_G[row][col] >= array_G[row - 1][col] and array_G[row][col] >= \
                                array_G[row + 1][col]:
                            array_nmax_suppress[row][col] = array_G[row][col]

                    # 135 degrees
                    if (157.5 > array_theta[row][col] >= 112.5) or (-22.5 > array_theta[row][col] >= -67.5):
                        if array_G[row][col] >= array_G[row - 1][col + 1] and array_G[row][col] >= \
                                array_G[row + 1][col - 1]:
                            array_nmax_suppress[row][col] = array_G[row][col]
        return array_nmax_suppress

    def set_array_after_thresholding(self, array_nmax_suppress):
        self._set_strong_edges(array_nmax_suppress)
        self._set_weak_edges(array_nmax_suppress)

    def _set_strong_edges(self, array_nmax_suppress):
        for row in range(self.height):
            for col in range(self.width):
                if self.width - 1 > col > 0 and self.height - 1 > row > 0:
                    if array_nmax_suppress[row][col] > self.maxVal:
                        self.canny_edges[row][col] = 1

    def _set_weak_edges(self, array_nmax_suppress):
        for row in range(self.height):
            for col in range(self.width):
                if self.width - 1 > col > 0 and self.height - 1 > row > 0:
                    if self.maxVal > array_nmax_suppress[row][col] > self.minVal and self._is_connected_with_strong_edge(row, col):
                        self.canny_edges[row][col] = 1

    def _is_connected_with_strong_edge(self, row, col):
        deltas = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1)]
        for i, j in deltas:
            if self.canny_edges[row + i][col + j] == 1:
                self.canny_edges[row][col] = 1

    def convolve(self, array, filter_array, result_array):
        offset = filter_array.shape[0] // 2
        for row in range(self.height):
            for col in range(self.width):
                if (self.height - 2 * offset >= row >= offset) and (self.width - 2 * offset >= col >= offset):
                    new_value = 0
                    for f_row in range(filter_array.shape[0]):
                        for f_col in range(filter_array.shape[1]):
                            new_value += array[row - offset + f_row][col - offset + f_col] * filter_array[f_row][f_col]
                    result_array[row - offset, col - offset] = new_value
