import numpy as np
import scipy.sparse
import scipy.io
import h5py
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from H5MD_Analysis import contactmap

def compute_dynamic_cmap(traj_file, cutoff, stride, logfile):
	traj_file_basename_without_extension = os.path.basename(os.path.splitext(traj_file)[0])
	try:
		traj = h5py.File(traj_file, 'r')
	except:
		raise

	if logfile is not None:
		sys.stdout = open(logfile, 'w')
	else:
		sys.stdout = open(os.devnull, 'w')

	nsnapshots = traj['particles/all/position/time'].shape[0]
	snapshot_lst = np.arange(0, nsnapshots, stride)

	sys.stdout.write('Starting to computing...\n')
	for s in snapshot_lst:
		sys.stdout.write('Computing snapshot #{}\n'.format(s))
		position = traj['particles/all/position/value'][s]
		cmap = contactmap.contactmap0(position, cutoff)
		cmap_sparse = scipy.sparse.csr_matrix(cmap)

		# save to file
		scipy.io.mmwrite('cmap_' + traj_file_basename_without_extension + '_s{}.mtx'.format(s), cmap_sparse)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='compute contact map from a trajectory file at different time steps.')
	parser.add_argument('-in', '--input', help='LAMMPS dump files path. Need to be in H5MD format.', dest='input')
	parser.add_argument('-c', '--cutoff', help='Specify the value of cutoff for determing contacts.', dest='cutoff', type=float)
	parser.add_argument('-s', '--stride', help='Output contact map every this many snapshots.', dest='stride', type=int)
	parser.add_argument('-l', '--log', help='Output information to log file.', dest='logfile')
	args = parser.parse_args()

	compute_dynamic_cmap(args.input, args.cutoff, args.stride, args.logfile)