import numpy as np
import sklearn
import scipy
import scipy.linalg
import scipy.spatial
import itertools

def xyz2dmap(xyz, a):
    """
    Return distance map provided the xyz coordinates
    """
    return np.power(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(xyz)), a)

def a2xyz(A, force_positive_definite = False):
    """
    Function to generate particle coordinates given the connectivity matrix (rouse matrix)
    """
    TOL = 10**8.0
    eigvalue, eigvector = scipy.linalg.eigh(A)
    temp = 1.0/eigvalue[:, np.newaxis]

    # replace close zero eigenvalue with zero
    temp[temp == -np.inf] = 0.0
    temp[temp == np.inf] = 0.0
    temp[temp >= TOL] = 0.0
    temp[temp <= -TOL] = 0.0

    #temp[np.abs(temp) <= 10**-7] = 0.0

    # replace all positive element to be zero
    if force_positive_definite:
        temp[temp > 0.0] = 0.0

    # get positions
    positions = eigvector @ (np.sqrt(-temp) * np.random.randn(len(eigvalue), 3))
    return positions

def generate_structure_raw(eigvalue_inv_neg, eigvector, force_positive_definite = False):
    """
    Function to generate particle coordinates given the negative inverse eigenvalue and eigenvector
    """
    TOL = 10**8
    temp = eigvalue_inv_neg[:, np.newaxis]

    # replace close zero eigenvalue with zero
    temp[temp == -np.inf] = 0.0
    temp[temp == np.inf] = 0.0
    temp[temp >= TOL] = 0.0
    temp[temp <= -TOL] = 0.0

    #temp[np.abs(temp) <= 10**-7] = 0.0
    #print(temp)

    # replace all positive element to be zero
    if force_positive_definite:
        temp[temp < 0.0] = 0.0

    positions = eigvector @ (np.sqrt(temp) * np.random.randn(len(eigvalue_inv_neg), 3))
    return positions

def sigma2omega(sigma_mtx):
    """
    Return Omega matrix given the sigma matrix
    """
    n = sigma_mtx.shape[0]
    sigma_mtx_square = np.power(sigma_mtx, 2.0)
    sigma_row_sum = np.sum(sigma_mtx_square, axis=1)
    sigma_sum = np.sum(sigma_mtx_square)
    return (sigma_row_sum[:, np.newaxis] + sigma_row_sum - sigma_sum / n) / (2 * n) - sigma_mtx_square / 2.0

def dmap2xyz(dmap):
    """
    Return a realization of xyz coordinates given the mean distance map
    """
    sigma_mtx = 0.5 * np.sqrt(np.pi / 2.0) * dmap
    Omega = sigma2omega(sigma_mtx)
    eigvalue, eigvector = scipy.linalg.eigh(Omega)
    #positions = generate_structure_raw(np.abs(eigvalue), eigvector)
    positions = generate_structure_raw(eigvalue, eigvector, force_positive_definite = True)
    return positions

def dmap2xyz_ensemble(dmap, ensemble):
    """
    Return an ensemble of xyz coordinates given the mean distance map
    """
    sigma_mtx = 0.5 * np.sqrt(np.pi / 2.0) * dmap
    Omega = sigma2omega(sigma_mtx)
    eigvalue, eigvector = scipy.linalg.eigh(Omega)
    xyz = []
    for _ in range(ensemble):
        #positions = generate_structure_raw(np.abs(eigvalue), eigvector)
        positions = generate_structure_raw(eigvalue, eigvector, force_positive_definite = True)
        xyz.append(positions)
    xyz = np.array(xyz)
    return xyz

def dmap2cmap(dmap, rc):
    """
    Return contact map given the mean distance map and the contact threshold
    """
    sigma_mtx = 0.5 * np.sqrt(np.pi / 2.0) * dmap
    cmap = scipy.special.erf(rc/(np.sqrt(2) * sigma_mtx)) - \
            np.sqrt(2.0/np.pi) * np.exp(-0.5 * rc**2.0/np.power(sigma_mtx, 2.0)) * rc / sigma_mtx
    np.fill_diagonal(cmap, 1.0)
    return cmap

def rebalance_connectivity_mtx(A):
    """
    Rebalance the connectivity matrix
    """
    A_copy = np.copy(A)
    for i in range(A.shape[0]-1):
        for j in range(i+1, A.shape[0]):
            if A[i,j] < 0.0:
                A_copy[i,j] = 0.0
                A_copy[j,i] = 0.0

    for i in range(A.shape[0]):
        A_copy[i,i] = -np.sum(np.delete(A_copy[:,i], i))

    A_copy += A_copy.T
    A_copy /= 2.0
    return A_copy

def dmap2a(dmap):
    """
    Return connectivity matrix A given the mean distance map
    """
    sigma_mtx = 0.5 * np.sqrt(np.pi / 2.0) * dmap
    Omega = sigma2omega(sigma_mtx)
    eigvalue, eigvector = scipy.linalg.eigh(Omega)
    eigvalue_inv = -1.0/eigvalue
    eigvalue_inv[0] = 0 # replace the first eigvalue with zero (corresponding pure connectivity matrix A)
    #eigvalue_inv[np.abs(eigvalue_inv) >= 10**7] = 0.0
    A_recover = eigvector @ np.diag(eigvalue_inv) @ eigvector.T

    return A_recover

def triangle_inequality_check(dmap):
    """
    Return the list of triplets which violate the triangle inequality
    """
    n = dmap.shape[0]
    triplets = itertools.combinations(np.arange(n), 3)
    violation = []
    for triplet in triplets:
        i,j,k = triplet
        if dmap[i,j] + dmap[j,k] <= dmap[i,k] or \
           dmap[i,j] + dmap[i,k] <= dmap[j,k] or \
           dmap[j,k] + dmap[i,k] <= dmap[i,j]:
            violation.append(triplet)
    return violation

def a2dmap(A, ensemble = 1000):
    """
    Generate mean distance map given the connectivity matrix
    """
    dmap = np.zeros(A.shape)
    for i in range(ensemble):
        xyz = a2xyz(A)
        dmap += xyz2dmap(xyz, 1.0)
    dmap /= ensemble
    return dmap

def a2dmap_theory(A):
    """
    Return mean distance map given the connectivity matrix A theoretically
    """
    TOL = 10**8
    eigvalue, eigvector = scipy.linalg.eigh(A)

    temp = -1.0 / eigvalue
    temp[temp == -np.inf] = 0.0
    temp[temp == np.inf] = 0.0
    temp[temp >= TOL] = 0.0
    temp[temp <= -TOL] = 0.0
    #temp[np.abs(temp) <= 10**-7] = 0.0

    Omega = eigvector @ np.diag(temp) @ eigvector.T
    Omega_diag = np.diag(Omega)
    sigma = np.sqrt(Omega_diag[:, np.newaxis] + Omega_diag - 2.0 * Omega)

    dmap = 2.0 * np.sqrt(2.0 / np.pi) * sigma
    return dmap

def a2cmap_theory(A, rc):
    """
    Return contact map given the connectivity matrix and contact threshold, theoretically
    """
    dmap = a2dmap_theory(A)
    cmap = dmap2cmap(dmap, rc)
    return cmap

def a_sld(A, idx):
    """
    Return a single loci deletion pertubated connectivity matrix given the index
    """
    A_mutate = np.delete(np.delete(A, idx, axis=0), idx, axis=1)
    if idx > 0 and idx < A.shape[0] - 1:
        A_mutate[idx-1, idx] = (A[idx-1, idx] + A[idx, idx+1]) / 2.0
        A_mutate[idx, idx-1] = A_mutate[idx-1, idx]

    for i in range(A_mutate.shape[0]):
        A_mutate[i, i] = - np.sum(np.delete(A_mutate[:, i], i))

    return A_mutate

def mtx_insert(mtx, idx):
    """
    Return a matrix after inserting a row/column at position idx
    """
    temp1 = np.insert(mtx, idx, np.zeros(mtx.shape[0]), axis=1)
    temp2 = np.insert(temp1, idx, np.zeros(mtx.shape[0]+1), axis=0)
    return temp2

def kth_diag_indices(a, k):
    """
    Return the indices of kth offset diagonal
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def a_si(A, idx_start, idx_end, offsets):
    """
    Return segment reverted pertubated connectivity matrix
    """
    A_mutate = np.copy(A)

    A_mutate[idx_start:idx_end, idx_start:idx_end] = np.flip(A[idx_start:idx_end, idx_start:idx_end])
    for k in range(1, offsets+1):
        indices = kth_diag_indices(A_mutate[idx_start:idx_end, idx_start:idx_end], k)
        A_mutate[idx_start:idx_end, idx_start:idx_end][indices] = A[idx_start:idx_end, idx_start:idx_end].diagonal(k)

    A_mutate[idx_start+offsets:idx_end-offsets, idx_end:] = np.flip(A[idx_start+offsets:idx_end-offsets, idx_end:], axis=0)
    A_mutate[:idx_start, idx_start+offsets:idx_end-offsets] = np.flip(A[:idx_start, idx_start+offsets:idx_end-offsets], axis=1)
    A_mutate = np.triu(A_mutate)
    A_mutate += A_mutate.T

    for i in range(A_mutate.shape[0]):
        A_mutate[i,i] = -np.sum(np.delete(A_mutate[:,i], i))
    return A_mutate

def optimal_rotate(P, Q, return_rotation = False):
    """
    Return aligned matrix referred to Q
    Can return rotation matrix if return_rotation is set True
    """
    # P and Q are two sets of vectors
    P = np.matrix(P)
    Q = np.matrix(Q)

    assert P.shape == Q.shape

    Qc = np.mean(Q,axis=0)

    P = P - np.mean(P,axis=0)
    Q = Q - np.mean(Q,axis=0)

    # calculate covariance matrix A = (P^T)Q
    A = P.T * Q

    # SVD for matrix A
    V, S, Wt = np.linalg.svd(A)

    # correct rotation matrix to ensure a right-handed system if necessary
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]

    # calculate the final rotation matrix U
    #U = V * Wt
    U = np.dot(V, Wt)

    if not return_rotation:
        return np.array(P * U + Qc)
    else:
        return np.array(P * U + Qc), U

def construct_connectivity_matrix_rouse(n,k):
    """
    Function to construct a ideal chain connectivity matrix given the number of monomers and the spring constant
    """
    A = np.diag(np.full(n-1,k),1)
    A += A.T
    A[np.diag_indices(n)] = -2*k
    A[0,0]=-k
    A[n-1,n-1]=-k
    return A

def construct_connectivity_matrix_random(n,m,k):
    """
    Generate random connected rouse chain
    n: the length of chain
    m: number of non-consecutive bonds
    k: the spring constant
    """
    A = construct_connectivity_matrix_rouse(n,k)
    pairs = list(itertools.combinations(np.arange(n),2))

    for pair in pairs:
        if pair[1] - pair[0] == 1:
            pairs.remove(pair)
    #print(len(pairs))
    pairs_indices = np.random.choice(len(pairs), m, replace=False)

    for idx in pairs_indices:
        pair = pairs[idx]
        A[pair[0], pair[1]] = k
        A[pair[1], pair[0]] = k

    for i in range(A.shape[0]):
        A[i,i] = -np.sum(np.delete(A[:,i], i))

    return A

def ree2xyz(n, k, xyz_start, xyz_end):
    """
    Generate random rouse chain conformation given the start and end monomer positions
    n: number of monomers
    k: string constant for chain connectivity
    """
    A = construct_connectivity_matrix_rouse(n, k)
    # set w to be large number
    w = -100000.0
    A[0,0] += w
    A[-1,-1] += w
    L = np.sqrt(np.sum(np.power(np.array(xyz_end) - np.array(xyz_start), 2.0)))

    # b array is the external constraints array on monomers
    b = np.zeros((n, 3))
    b[-1,2] = -w * L

    eigvalue, eigvector = scipy.linalg.eigh(A)

    xyz = eigvector @ ((np.sqrt(-1.0/eigvalue))[:, np.newaxis] *
                       np.random.randn(len(eigvalue), 3) + (eigvector.T @ b) *
                       (-1.0/eigvalue)[:, np.newaxis])
    xyz[0] = np.array([0.0,0.0,0.0])
    xyz[-1] = np.array([0.0,0.0,L])

    end_end_vector1 = np.array([[0,0,0],[0,0,L]])
    end_end_vector2 = np.array([np.array(xyz_start), np.array(xyz_end)])

    _, rotation_mtx = optimal_rotate(end_end_vector1,
                                           end_end_vector2,
                                           return_rotation=True)

    xyz = xyz - np.mean(end_end_vector1, axis=0)

    xyz = np.array(xyz * rotation_mtx + np.mean(end_end_vector2, axis=0))

    return xyz
