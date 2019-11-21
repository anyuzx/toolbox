import numpy as np
import h5py
import argparse
import sys
import os
import time

from io import BytesIO
from itertools import islice

def str2array(str):
    strIO = BytesIO(str.encode())
    return np.loadtxt(strIO)

# First declare several flags
parser = argparse.ArgumentParser(description='Convert xyz format file to Lammps H5MD format dump file.\
                                 IMPORTANT NOTES:Only used for simulation where number of particles does not change.')
parser.add_argument('xyz_dump', help="xyz format file")
parser.add_argument('lammps_hdf5_dump', help='H5MD file.')
parser.add_argument('-s', '--stride', help='write every this many snapshots.', \
                    dest='stride', type=int)
parser.add_argument('-i', '--index', help='write for atoms with index between two numbers. index starts from one not zero.',\
					dest='index', type=int, nargs = 2)
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

# first read number of atoms in the xyz file
# read atom attribute information
check = False
with open(args.xyz_dump) as f:
    natoms = int(str2array(f.readline()))

sys.stdout.write('Start converting H5MD file to xyz format file.\n')
sys.stdout.flush()

number_lines_one_frame = 9 + natoms # 9 = number of head lines for each frame

hdf_file = h5py.File(args.lammps_hdf5_dump, 'w')

h5md_traj = h5py.File(args.lammps_hdf5_dump, 'r')
nsnapshots = h5md_traj['particles/all/position/value'].shape[0]

natoms = h5md_traj['particles/all/position/value'].shape[1]

if args.index is not None:
	assert args.index[1] >= args.index[0], "first index must not be larger than the second index."
	assert args.index[1] <= natoms, "index larger than the total number of atoms."
	natoms = args.index[1] - args.index[0] + 1

with open(args.xyz_dump, 'w') as f:
	for s in range(0, nsnapshots, stride):
		position = h5md_traj['particles/all/position/value'][s]

		f.write('{}\n'.format(natoms))
		f.write('Atoms. Timestep: {}\n'.format(h5md_traj['particles/all/position/step'][s]))

		try:
			type_lst = h5md_traj['particles/all/species/value'][s]
		except KeyError:
			type_lst = np.ones(natoms, dtype=np.int)
		if args.index is None:
			snapshot = np.hstack((np.arange(1,natoms+1).reshape(natoms, 1), position))
		else:
			snapshot = np.hstack((np.arange(1,natoms+1).reshape(natoms, 1), position[args.index[0]-1:args.index[1]]))

		for item in snapshot:
			f.write('{} {} {} {}\n'.format(int(item[0]), item[1], item[2], item[3]))

		sys.stdout.write("Writing snapshot #{}\n".format(s+1))
        sys.stdout.flush()

end_time = time.time()
sys.stdout.write('\nTotal {} snapshots been written to xyz format file. Time used:{} mins\n'.format(nsnapshots, (end_time-start_time)/60))
sys.stdout.flush()
h5md_traj.close()
