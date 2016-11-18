import numpy as np
import scipy.io
import scipy.sparse
import sys

beta = sys.argv[1]

sys.stdout = open('log_beta{}.txt'.format(beta), 'w')

if beta == '1.0' or '2.0':
	t_lst = np.arange(0,13500,1000)
elif beta == '2.4' or '2.7':
	t_lst = np.arange(0,10500,1000)

for i in range(1,90):
	for j in range(i+1,91):
		for t in t_lst:
			ci = scipy.io.mmread('../beta{}/cmap_Chr5_SC_FENE_BETA{}_{}_traj_s{}.mtx'.format(beta, beta, i, t))
			cj = scipy.io.mmread('../beta{}/cmap_Chr5_SC_FENE_BETA{}_{}_traj_s{}.mtx'.format(beta, beta, j, t))

			ci.tocsr()
			cj.tocsr()

			ci = ci.toarray()
			cj = cj.toarray()
			hij = ci - cj
			hij_sparse = scipy.sparse.csr_matrix(hij)

			scipy.io.mmwrite('hmap_beta{}_{}_{}_t{}.mtx'.format(beta, i, j, t), hij_sparse)
                        sys.stdout.write('write file {} {} t{}\n'.format(i,j,t))
                        sys.stdout.flush()
