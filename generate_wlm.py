import numpy as np
import pandas as pd
import h5py
import sys
import _matrixnorm
import scipy
import scipy.interpolate
import scipy.cluster
from scipy.spatial.distance import pdist, squareform

import matplotlib
import matplotlib.pyplot as plt

# convert contact map to distance map
def cmap2dmap(cmap,exponent):
    cmap[np.where(cmap == 0.0)]=np.unique(np.sort(cmap.flatten()))[1]
    dmap = np.power(cmap, exponent)
    np.fill_diagonal(dmap, 0)
    return dmap

# Frobenius norm
def Frobenius_norm(m1, m2):
    temp = m1 - m2
    return scipy.linalg.norm(temp, 'fro')

# minimize WLM (Ward Linkage Matrix) comparison
def minimize_comparison_wlm(lam, m1, m2, compare_func, *kwargs):
    dmap1 = lam * m1.copy()
    dmap2 = cmap2dmap(m2, -1.0/4.1)

    dmap2 = np.delete(dmap2,[385,386],0)
    dmap2 = np.delete(dmap2,[385,386],1)

    linkage1 = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dmap1), method='centroid')
    linkage2 = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dmap2), method='centroid')

    clm1 = dendrogram2WLM(linkage1)
    clm2 = dendrogram2WLM(linkage2)
    clm2 = resize_matrix(clm2, 500)

    return compare_func(clm1, clm2, *kwargs)

def resize_matrix(matrix, resize):
    size = matrix.shape[0]
    f = scipy.interpolate.interp2d(np.arange(1,size+1),\
                                   np.arange(1,size+1),\
                                   matrix, kind='linear',bounds_error=True)
    xnew = np.linspace(1, size, resize)
    ynew = np.linspace(1, size, resize)
    resize_matrix = f(xnew, ynew)

    temp = np.zeros((resize,resize))
    temp += np.triu(resize_matrix)
    temp += temp.T

    return temp

def dendrogram2WLM(linkage_matrix):
    return squareform(scipy.cluster.hierarchy.cophenet(linkage_matrix))

# READ EXPERIMENT CONTACT MAP 25KB AND 5KB RESOLUTION
cmap_exp_25kb = np.loadtxt('/Users/gs27722/Dropbox/Documents/Academic_Document/Chromosome_packaging/results/exp_data/ps/GM12878_combines/chr5/chr5_25kb_combined_145870001_157870001_MAPQGE0_KR.normcmap', skiprows=2)
cmap_exp_25kb += cmap_exp_25kb.transpose()
cmap_exp_25kb_resize = resize_matrix(cmap_exp_25kb, 500)

fp = sys.argv[1]

dmap = np.load(fp)
dmap = dmap + dmap.transpose()

dmap_normalized = _matrixnorm.matrixnorm_mean(np.float32(dmap), 20)
dmap_normalized += dmap_normalized.T - - np.diag(dmap_normalized.diagonal())

diff = scipy.optimize.minimize(minimize_comparison_wlm, 1.0, args=(dmap_normalized, cmap_exp_25kb_resize, Frobenius_norm), \
                                method='L-BFGS-B',bounds=((10**-8,None),))
ward_linkage_dmap = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(diff['x'][0]*dmap_normalized), method='ward')
cophenetic_mtx = dendrogram2WLM(ward_linkage_dmap)

fig, ax = plt.subplots()
ax.imshow(dendrogram2WLM(ward_linkage_dmap), cmap=plt.cm.jet, vmax=10)
plt.savefig(fp+'_wlm.png', dpi=300)

np.save(fp+'_wlm.npy', dendrogram2WLM(ward_linkage_dmap))
