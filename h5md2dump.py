import numpy as np
import h5py
import argparse
import sys
import os
import time

# First declare several flags
parser = argparse.ArgumentParser(description='Convert Lammps H5MD format dump file to custom format file.\
                                 IMPORTANT NOTES:Only used for simulation where number of particles does not change.')
parser.add_argument('lammps_hdf5_dump', help='H5MD file.')
parser.add_argument('lammps_custom_dump', help='Lammps custom dump file.')
parser.add_argument('-s', '--stride', help='write H5MD file every this many snapshots.', \
                    dest='stride', type=int)
parser.add_argument('-l', '--log',  help='output to log files.',dest='logfile')
args = parser.parse_args()

if args.stride is None:
    stride = 1
else:
    stride = args.stride

start_time = time.time()
# redirect stdout to log file if specified
if args.logfile:
    sys.stdout = open(args.logfile, 'w')

sys.stdout.write('Start converting H5MD file to custom dump file.\n')
sys.stdout.flush()

h5md_traj = h5py.File(args.lammps_hdf5_dump, 'r')
nsnapshots = h5md_traj['particles/all/position/value'].shape[0]
natoms = h5md_traj['particles/all/position/value'].shape[1]

with open(args.lammps_custom_dump, 'w') as f:
	for s in range(0, nsnapshots, stride):
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

		type_lst = h5md_traj['particles/all/species/value'][s]
		snapshot = np.hstack((np.arange(1,natoms+1).reshape(natoms, 1), type_lst.reshape(natoms, 1), position))

		f.write('ITEM: ATOMS id type x y z\n')
		for item in snapshot:
			f.write('{} {} {} {} {}\n'.format(int(item[0]), int(item[1]), item[2], item[3], item[4]))

		sys.stdout.write("Writing snapshot #{}\n".format(s+1))
        sys.stdout.flush()

end_time = time.time()
sys.stdout.write('\nTotal {} snapshots been written to custom dump file. Time used:{} mins\n'.format(nsnapshots, (end_time-start_time)/60))
sys.stdout.flush()
h5md_traj.close()
