import math

import numpy as np
# from matplotlib import pyplot as plt


class Canny(object):
    def __init__(self, img, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal
        self.gaussian_filter = np.asarray([[1, 4, 7, 4, 1],
                                           [4, 16, 26, 16, 4],
                                           [7, 26, 41, 26, 7],
                                           [4, 16, 26, 16, 4],
                                           [1, 4, 7, 4, 1]]) / 273

        self.sobel_Gx_filter = np.asarray([[-1, 0, 1],
                                           [-2, 0, 2],
                                           [-1, 0, 1]])

        self.sobel_Gy_filter = np.asarray([[-1, -2, -1],
                                           [0, 0, 0],
                                           [1, 2, 1]])

        self.array = np.asarray(img)
        self.height, self.width = self.array.shape[0], self.array.shape[1]

        self.array_after_gaussian_filtering = np.zeros(self.array.shape)
        self.array_after_Gx = np.zeros(self.array.shape)
        self.array_after_Gy = np.zeros(self.array.shape)
        self.array_G = np.zeros(self.array.shape)
        self.array_theta = np.zeros(self.array.shape)
        self.array_nmax_suppress = np.zeros(self.array.shape)
        self.array_after_thresholding = np.zeros(self.array.shape)

        self.run()

    def run(self):
        self.set_gaussian_filtered_array()
        self.set_array_after_Gx()
        self.set_array_after_Gy()
        self.set_array_after_G()
        self.set_array_after_theta()
        self.set_array_after_suppression()
        self.set_array_after_thresholding()

    def get_features(self):
        # plt.imshow(self.array_after_thresholding)
        # plt.show()
        return self.array_after_thresholding

    def set_gaussian_filtered_array(self):
        zero_padded_array = np.pad(self.array, 2, mode='constant')
        self.convolve(zero_padded_array, self.gaussian_filter, self.array_after_gaussian_filtering)

    def set_array_after_Gx(self):
        zero_padded_array = np.pad(self.array_after_gaussian_filtering, 1, mode='constant')
        self.convolve(zero_padded_array, self.sobel_Gx_filter, self.array_after_Gx)

    def set_array_after_Gy(self):
        zero_padded_array = np.pad(self.array_after_gaussian_filtering, 1, mode='constant')
        self.convolve(zero_padded_array, self.sobel_Gy_filter, self.array_after_Gy)

    def set_array_after_G(self):
        self.array_G = np.sqrt(np.square(self.array_after_Gx) + np.square(self.array_after_Gy))

    def set_array_after_theta(self):
        self.array_theta = np.arctan2(self.array_after_Gy, self.array_after_Gx) * 180 / math.pi

    def set_array_after_suppression(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.width - 1 > col > 0 and self.height - 1 > row > 0:
                    # 0 degrees
                    if (22.5 > self.array_theta[row][col] >= -22.5) or (self.array_theta[row][col] < -157.5 or self.array_theta[row][col] >= 157.5):
                        if self.array_G[row][col] >= self.array_G[row][col + 1] and self.array_G[row][col] >= self.array_G[row][col - 1]:
                            self.array_nmax_suppress[row][col] = self.array_G[row][col]

                    # 45 degrees
                    if (67.5 > self.array_theta[row][col] >= 22.5) or (-112.5 > self.array_theta[row][col] >= -157.5):
                        if self.array_G[row][col] >= self.array_G[row - 1][col - 1] and self.array_G[row][col] >= self.array_G[row + 1][col + 1]:
                            self.array_nmax_suppress[row][col] = self.array_G[row][col]

                    # 90 degrees
                    if (112.5 > self.array_theta[row][col] >= 67.5) or (-67.5 > self.array_theta[row][col] >= -112.5):
                        if self.array_G[row][col] >= self.array_G[row - 1][col] and self.array_G[row][col] >= self.array_G[row + 1][col]:
                            self.array_nmax_suppress[row][col] = self.array_G[row][col]

                    # 135 degrees
                    if (157.5 > self.array_theta[row][col] >= 112.5) or (-22.5 > self.array_theta[row][col] >= -67.5):
                        if self.array_G[row][col] >= self.array_G[row - 1][col + 1] and self.array_G[row][col] >= self.array_G[row + 1][col - 1]:
                            self.array_nmax_suppress[row][col] = self.array_G[row][col]

    def set_array_after_thresholding(self):
        self._set_strong_edges()
        self._set_weak_edges()

    def _set_strong_edges(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.width - 1 > col > 0 and self.height - 1 > row > 0:
                    if self.array_nmax_suppress[row][col] > self.maxVal:
                        self.array_after_thresholding[row][col] = 1

    def _set_weak_edges(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.width - 1 > col > 0 and self.height - 1 > row > 0:
                    if self.maxVal > self.array_nmax_suppress[row][col] > self.minVal and self._is_connected_with_strong_edge(row, col):
                        self.array_after_thresholding[row][col] = 1

    def _is_connected_with_strong_edge(self, row, col):
        deltas = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1)]
        for i, j in deltas:
            if self.array_after_thresholding[row + i][col + j] == 1:
                self.array_after_thresholding[row][col] = 1

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
