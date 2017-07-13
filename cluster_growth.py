import numpy as np
import h5py
import sys

import scipy
import scipy.spatial

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth

def cluster_analysis(traj, snapshot_index = None):
    nsnapshot = traj['particles/all/position/value'].shape[0]
    if snapshot_index is None:
        snapshot_index = np.unique(np.int_(np.exp(np.linspace(np.log(1), np.log(nsnapshot-1), 2000))))

    n_cluster_array = []
    cluster_size_array = []
    for i in snapshot_index:
        pos = traj['particles/all/position/value'][i]

        db = DBSCAN(eps=1.9,min_samples=10).fit(pos)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_cluster_array.append(n_clusters_)

        unique_labels = set(labels)
        cluster_size = []
        for k in unique_labels:
            class_member_mask = (labels == k)
            xyz = pos[class_member_mask & core_samples_mask]
            cluster_size.append(xyz.shape[0])

        cluster_size_array.append(cluster_size)

    return n_cluster_array, cluster_size_array

fp = sys.argv[1]
traj = h5py.File(fp, 'r')

n_cluster_array, cluster_size_array = cluster_analysis(traj)
np.save(fp+'_ncluster.npy', n_cluster_array)
np.save(fp+'_cluster_size.npy', cluster_size_array)
