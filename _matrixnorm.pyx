import numpy as np
import cython
cimport numpy as np
from libc.math cimport pow, sqrt

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def matrixnorm(np.ndarray[DTYPE_t,ndim=2] cmap, int a):
        cdef int i,j
        cdef int N = cmap.shape[0]
        # a is normalization factor
        cdef int b = N/a

        cdef np.ndarray[DTYPE_t,ndim=2] output = np.zeros((b,b),dtype=DTYPE)
        cdef DTYPE_t tmp
        for i in xrange(b):
                for j in xrange(i):
                        output[i,j] = np.sum(cmap[i*a:(i+1)*a,j*a:(j+1)*a])

        return output

@cython.boundscheck(False)
@cython.wraparound(False)
def matrixnorm_OE(np.ndarray[DTYPE_t, ndim=2] cmap, int a):
        cdef int i,j
        cdef int N = cmap.shape[0]
        # a is normalization factor
        cdef int b = N/a

        cdef np.ndarray[DTYPE_t, ndim=2] output = np.zeros((b,b), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] expected = np.zeros(b-1, dtype=DTYPE)
        cdef DTYPE_t observed
        for i in xrange(b):
                for j in xrange(i):
                        expected[i-j-1] += np.sum(cmap[i*a:(i+1)*a,j*a:(j+1)*a])/float(b-(i-j))
        for i in xrange(b):
                for j in xrange(i):
                        observed = np.sum(cmap[i*a:(i+1)*a,j*a:(j+1)*a])
                        output[i, j] = observed/expected[i-j-1]
        return output


@cython.boundscheck(False)
@cython.wraparound(False)
def matrixnorm_zscore(np.ndarray[DTYPE_t, ndim=2] cmap, int a):
        cdef int i, j
        cdef int N = cmap.shape[0]
        cdef int b = N/a

        cdef np.ndarray[DTYPE_t, ndim=2] output = np.zeros((b,b), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] expected = np.zeros(b-1, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] square = np.zeros(b-1, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] sigma = np.zeros(b-1, dtype=DTYPE)
        cdef DTYPE_t observed
        for i in xrange(b):
                for j in xrange(i):
                        expected[i-j-1] += np.sum(cmap[i*a:(i+1)*a,j*a:(j+1)*a])/float(b-(i-j))
                        square[i-j-1] += pow(np.sum(cmap[i*a:(i+1)*a,j*a:(j+1)*a]),2.0)/float(b-(i-j))
        for i in xrange(b-1):
                sigma[i] = sqrt(square[i] - pow(expected[i],2.0))
        for i in xrange(b):
                for j in xrange(i):
                        observed = np.sum(cmap[i*a:(i+1)*a,j*a:(j+1)*a])
                        output[i, j] = (observed-expected[i-j-1])/sigma[i-j-1]
        return output

@cython.boundscheck(False)
@cython.wraparound(False)
def matrixnorm_correlation(np.ndarray[DTYPE_t, ndim=2] cmap_OE):
        cdef int i, j
        cdef int N = cmap_OE.shape[0]
        cdef np.ndarray[DTYPE_t, ndim=2] coeff_matrix = np.zeros((N,N), dtype=DTYPE)

        cdef DTYPE_t i_mean, j_mean, ij_mean, i_std, j_std
        for i in xrange(N):
                for j in xrange(i):
                        i_mean = np.mean(cmap_OE[i,:])
                        j_mean = np.mean(cmap_OE[j,:])
                        ij_mean = np.mean(cmap_OE[i,:]*cmap_OE[j,:])
                        i_std = np.std(cmap_OE[i,:])
                        j_std = np.std(cmap_OE[j,:])
                        pearson_coeff = (ij_mean - i_mean * j_mean)/(i_std * j_std)
                        coeff_matrix[i,j] = pearson_coeff
        return coeff_matrix
