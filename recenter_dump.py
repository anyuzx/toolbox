import numpy as np
import h5py
import sys
from io import StringIO
import time
from itertools import islice
import argparse
import os

def str2array(str):
    str_temp = str.decode('unicode-escape')
    strIO = StringIO(str_temp)
    return np.loadtxt(strIO)

def compute_com_pbc(coords, box_size):
    theta = 2.0 * np.pi * coords / np.array(box_size)
    xi = np.cos(theta) * box_size / (2.0 * np.pi)
    zeta = np.sin(theta) * box_size / (2.0 * np.pi)
    xi_mean = np.mean(xi, axis = 0)
    #print xi_mean
    zeta_mean = np.mean(zeta, axis = 0)
    com = box_size * (np.arctan2(-zeta_mean, -xi_mean) + np.pi) / (2.0 * np.pi)
    return com

def reposition(coords, box_size):
    com = compute_com_pbc(coords, box_size)
    cob = box_size / 2.0
    coords_recenter = coords - com + cob
    coords_recenter_x = coords_recenter[:,0]
    coords_recenter_y = coords_recenter[:,1]
    coords_recenter_z = coords_recenter[:,2]
    #print coords_recenter
    coords_recenter_x = np.piecewise(coords_recenter_x, [coords_recenter_x < 0.0, (coords_recenter_x >= 0.0) * (coords_recenter_x <= box_size[0]), coords_recenter_x > box_size[0]], \
        [lambda coords_recenter_x: coords_recenter_x + box_size[0], lambda coords_recenter_x: coords_recenter_x, lambda coords_recenter_x: coords_recenter_x - box_size[0]])
    coords_recenter_y = np.piecewise(coords_recenter_y, [coords_recenter_y < 0.0, (coords_recenter_y >= 0.0) * (coords_recenter_y <= box_size[1]), coords_recenter_y > box_size[1]], \
        [lambda coords_recenter_y: coords_recenter_y + box_size[1], lambda coords_recenter_y: coords_recenter_y, lambda coords_recenter_y: coords_recenter_y - box_size[1]])
    coords_recenter_z = np.piecewise(coords_recenter_z, [coords_recenter_z < 0.0, (coords_recenter_z >= 0.0) * (coords_recenter_z <= box_size[2]), coords_recenter_z > box_size[2]], \
        [lambda coords_recenter_z: coords_recenter_z + box_size[2], lambda coords_recenter_z: coords_recenter_z, lambda coords_recenter_z: coords_recenter_z - box_size[2]])
    return np.array(zip(coords_recenter_x,coords_recenter_y,coords_recenter_z))


def mystr(s):
	if s.is_integer():
		return str(int(s))
	else:
		return str(s)

######################################
velocity_flag = False
others_flag = True
unwrap_flag = False
image_flag = False

# First declare several flags
parser = argparse.ArgumentParser(description='Convert Lammps custom dump file to output format file.\
                                 IMPORTANT NOTES:Only used for simulation where number of particles does not change.')
parser.add_argument('lammps_custom_dump', help='Lammps custom dump file.')
parser.add_argument('lammps_new_dump', help='new dump file.')
parser.add_argument('-nv', '--no-velocity', help='disable writing velocity to output file.', \
                    action='store_true', dest='no_velocity')
parser.add_argument('-no', '--no-others', help='disbale writing other informations to output file.', \
                    action='store_true', dest='no_others')
parser.add_argument('-s', '--stride', help='write output file every this many snapshots.', \
                    dest='stride', type=int)
parser.add_argument('-b', '--begin', help='write output file starting from this index.', \
                    dest='begin', type=int)
parser.add_argument('-t', '--terminate', help='stop write output file starting from this index.', \
                    dest='terminate', type=int)
parser.add_argument('-uw', '--unwrap', help='write unwrapped coordinates of particles in output file.', \
                    action='store_true', dest='unwrap')
parser.add_argument('-i', '--image', help='store image flags of particles in output file.', \
                    action='store_true', dest='image')
parser.add_argument('-q', '--quite', help='turn of printing information on screen.',\
                    action='store_true', dest='quite')
parser.add_argument('-l', '--log',  help='output to log files.',\
                    dest='logfile')
args = parser.parse_args()

# report error if both args.quite and args.logfile are required
if args.quite and args.logfile:
    sys.stdout.write('ERROR: Both quite and log argument are specified. Program terminated.\n')
    sys.stdout.flush()
    sys.exit(0)

# redirect stdout to log file if specified
if args.logfile:
    sys.stdout = open(args.logfile, 'w')
elif args.quite:
    sys.stdout = open(os.devnull, 'w')

if args.stride is None:
    stride = 1
else:
    stride = args.stride

# first read number of atoms in the dump file
# read atom attribute information
check = False
with open(args.lammps_custom_dump) as f:
    for i, line in enumerate(f):
        if line == 'ITEM: NUMBER OF ATOMS\n':
            check = True
            continue
        if check:
            natoms = np.int_(line.split()[0])
            check = False
        if 'ITEM: BOX BOUNDS' in line:
            if not np.any(['p' in line.split()[i] for i in range(3,len(line.split()))]):
                if args.unwrap:
                    sys.stdout.write("No periodic boundary found. Ignore argument '--unwrap'.\n")
                if args.image:
                    sys.stdtout.write("No periodic boundary found. Ignore argument '--image'.\n")
            else:
                if not args.unwrap and not args.image:
                    sys.stdout.write("\033[93mWARNING: Periodic boundary found. Neither argument '--unwrap' nor '--image' are provided.\033[0m\n")
        if 'ITEM: ATOMS' in line:
            attribute_info = line.split()[2:]
            break

# check attribute information
if 'id' in attribute_info:
    id_index = attribute_info.index('id')
else:
    sys.stdout.write('\033[93mERROR: No particle ID is found in dump file. \
Make sure that the order of particles does not change in dump file.\033[0m\n')
    sys.stdout.flush()

if 'x' in attribute_info and 'y' in attribute_info and 'z' in attribute_info:
    x_index = attribute_info.index('x')
    y_index = attribute_info.index('y')
    z_index = attribute_info.index('z')
else:
    sys.stdout.write('*** No position information found in dump file. Terminated. ***\n')
    sys.stdout.flush()
    sys.exit(0)

if 'vx' in attribute_info and 'vy' in attribute_info and 'vz' in attribute_info:
    vx_index = attribute_info.index('vx')
    vy_index = attribute_info.index('vy')
    vz_index = attribute_info.index('vz')
    if not args.no_velocity:
        velocity_flag = True
else:
    if not args.no_velocity:
        sys.stdout.write('*** No velocity information found in dump file. Skip it. ***\n')
        sys.stdout.flush()

if 'ix' in attribute_info and 'iy' in attribute_info and 'iz' in attribute_info:
    ix_index = attribute_info.index('ix')
    iy_index = attribute_info.index('iy')
    iz_index = attribute_info.index('iz')
    if not args.unwrap and not args.image:
        sys.stdout.write('\033[93mWARNING: Image flags found in dump file.\033[0m\n')
        sys.stdout.flush()
    unwrap_flag = args.unwrap
    image_flag = args.image
else:
    if args.unwrap or args.image:
        sys.stdout.write('*** No image information found in dump file. Skip it. ***\n')
        sys.stdout.flush()

if 'xu' in attribute_info and 'yu' in attribute_info and 'zu' in attribute_info:
    sys.stdout.write('\033[93mWARNING: Unwrapped position found in dump file.\033[0m\n')
    sys.stdout.flush()

number_lines_one_frame = 9 + natoms # 9 = number of head lines for each frame

attribute_info_new = attribute_info[:]
for key in ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ix', 'iy', 'iz', 'xu', 'yu', 'zu']:
    try:
        attribute_info_new.remove(key)
    except ValueError:
        pass

snap_index = 0 # keep track of the index of frames when reading
snap_index_write = 0 # keep track the actual number of snapshots written to output file

start_time = time.time() # get the start time
if args.logfile:
    sys.stdout.write('Start to convert file {}...\n'.format(args.lammps_custom_dump))
    sys.stdout.flush()
else:
    sys.stdout.write('\033[1mStart to convert file {}...\033[0m\n'.format(args.lammps_custom_dump))
    sys.stdout.flush()

with open(args.lammps_new_dump, 'w') as fout:
	with open(args.lammps_custom_dump, 'r') as f:
	    while True:
	        next_n_lines = list(islice(f, number_lines_one_frame))

	        # enumerate all the posiibilities
	        if args.begin is None and args.terminate is None:
	            if snap_index % stride == 0:
	                pass
	            else:
	                snap_index += 1
	                continue
	        elif args.begin is not None and args.terminate is None:
	            if snap_index >= args.begin:
	                if snap_index % stride == 0:
	                    pass
	                else:
	                    snap_index += 1
	                    continue
	            else:
	                snap_index += 1
	                continue
	        elif args.begin is None and args.terminate is not None:
	            if snap_index <= args.terminate:
	                if snap_index % stride == 0:
	                    pass
	                else:
	                    snap_index += 1
	                    continue
	            else:
	                break
	        elif args.begin is not None and args.terminate is not None:
	            if snap_index >= args.begin and snap_index <= args.terminate:
	                if snap_index % stride == 0:
	                    pass
	                else:
	                    snap_index += 1
	                    continue
	            elif snap_index > args.terminate:
	                break
	            else:
	                snap_index += 1
	                continue

	        if not next_n_lines:
	            break

	        # process next_n_lines
	        # get timestep
	        timestep = int(next_n_lines[1])
	        # get box
	        box = ''.join(next_n_lines[5:8])
	        box = str2array(box)
	        box_shape = box.shape
	        if unwrap_flag:
	            box_edge_size = box[:,1] - box[:,0]

	        # get per atom information: id, position, velocity, energy ...
	        atom_info = ''.join(next_n_lines[9:])
	        atom_info = str2array(atom_info)
	        assert len(atom_info) == natoms

	        # sort the atom information based on atom id
	        try:
	            atom_info = atom_info[atom_info[:,id_index].argsort()]
	        except:
	            pass

	        coords = np.float64(atom_info[:,x_index:z_index+1])
	        # compute center of mass (see wiki page for periodic boudary condition)
	        coords_reposition = reposition(coords, box[:,1] - box[:,0])

	        atom_info[:,x_index:z_index+1] = coords_reposition

	        fout.write("".join(next_n_lines[:9]))
	        for row in atom_info:
	        	fout.write(" ".join([mystr(elem) for elem in row])+"\n")
	        #fout.write(atom_info)

	        snap_index += 1
	        snap_index_write += 1
	        if args.logfile:
	            sys.stdout.write("Writing snapshot #{}\n".format(snap_index))
	            sys.stdout.flush()
	        else:
	            sys.stdout.write("\rWriting snapshot #{}...".format(snap_index))
	            sys.stdout.flush()

end_time = time.time()
sys.stdout.write('\nTotal {} snapshots been written to dump file. Time used:{} mins\n'.format(snap_index_write, (end_time-start_time)/60))
sys.stdout.flush()
