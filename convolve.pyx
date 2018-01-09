from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t

def convolve(np.ndarray[DTYPE_t, ndim=2] array, np.ndarray[DTYPE_t, ndim=2] filter_array):
    cdef int offset = filter_array.shape[0] // 2
    cdef int height = array.shape[0]
    cdef int width = array.shape[1]
    cdef int f_height = filter_array.shape[0]
    cdef int f_width = filter_array.shape[1]

    cdef int row, col, f_row, f_col

    cdef np.ndarray[DTYPE_t, ndim=2] result_array = np.zeros([height, width], dtype=DTYPE)
    cdef DTYPE_t new_value

    for row in range(height):
        for col in range(width):
            if (height - 2 * offset >= row >= offset) and (width - 2 * offset >= col >= offset):
                new_value = 0
                for f_row in range(f_height):
                    for f_col in range(f_width):
                        new_value += array[row - offset + f_row][col - offset + f_col] * filter_array[f_row][f_col]
                result_array[row - offset, col - offset] = new_value

    return result_array