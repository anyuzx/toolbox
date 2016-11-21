import numpy as np
import h5py
import argparse
import sys
import os

# First declare several flags
parser = argparse.ArgumentParser(description='Convert Lammps H5MD format dump file to custom format file.\
                                 IMPORTANT NOTES:Only used for simulation where number of particles does not change.')
parser.add_argument('lammps_hdf5_dump', help='H5MD file.')
parser.add_argument('lammps_custom_dump', help='Lammps custom dump file.')
args = parser.parse_args()

h5md_traj = h5py.File(args.lammps_hdf5_dump, 'r')
nsnapshots = h5md_traj['particles/all/position/value'].shape[0]
natoms = h5md_traj['particles/all/position/value'].shape[1]

with open(args.lammps_custom_dump, 'w') as f:
	for s in len(nsnapshots):
		position = h5md_traj['particles/all/position/value'][s]

		f.write('ITEM: TIMESTEP\n')
		f.write('{}\n'.format(h5md_traj['particles/all/position/step'][s]))
		f.write('ITEM: NUMBER OF ATOMS\n')
		f.write('{}\n'.format(natoms))
		f.write('ITEM: BOX BOUNDS ss ss ss\n')

		# get the box dimension
		xlo = position[:,0].min()
		xhi = position[:,0].max()
		ylo = position[:,1].min()
		yhi = position[:,1].max()
		zlo = position[:,2].min()
		zhi = position[:,2].max()

		f.write('{} {}\n'.format(xlo, xhi))
		f.write('{} {}\n'.format(ylo, yhi))
		f.write('{} {}\n'.format(zlo, zhi))

		type_lst = position['particles/all/species/value'][s]
		snapshot = np.hstack((np.arange(1,natoms+1).reshape(natoms, 1), type_lst.reshape(natoms, 1), position))

		f.write('ITEM: ATOMS id type x y z\n')
		for item in snapshot:
			f.write('{} {} {} {} {}\n'.format(int(item[0]), int(item[1]), item[2], item[3], item[4]))
