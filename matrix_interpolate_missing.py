import numpy as np
import scipy
import scipy.interpolate

def interpolate_miss(matrix, method='nearest'):
    matrix_copy = np.copy(matrix)
    x = np.arange(0, matrix_copy.shape[1])
    y = np.arange(0, matrix_copy.shape[0])
    #mask invalid values
    matrix_copy = np.ma.masked_invalid(matrix_copy)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~matrix_copy.mask]
    y1 = yy[~matrix_copy.mask]
    newarr = matrix_copy[~matrix_copy.mask]

    GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method=method)
    return GD1