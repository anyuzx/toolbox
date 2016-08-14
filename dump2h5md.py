import numpy as np
import h5py
import sys
from io import StringIO
import time
from itertools import islice
import argparse

def str2array(str):
    str_temp = str.decode('unicode-escape')
    strIO = StringIO(str_temp)
    return np.loadtxt(strIO)


######################################
position_flag = False
velocity_flag = False
others_flag = True
unwrap_flag = False
image_flag = False

# First declare several flags
parser = argparse.ArgumentParser(description='Convert Lammps custom dump file to H5MD format file.\
                                 IMPORTANT NOTES:Only used for simulation where number of particles does not change.')
parser.add_argument('lammps_custom_dump', help='Lammps custom dump file.')
parser.add_argument('lammps_hdf5_dump', help='H5MD file.')
parser.add_argument('-np', '--no-position', help='disable writing position to H5MD file.', \
                    action='store_true', dest='no_position')
parser.add_argument('-nv', '--no-velocity', help='disable writing velocity to H5MD file.', \
                    action='store_true', dest='no_velocity')
parser.add_argument('-no', '--no-others', help='disbale writing other informations to H5MD file.', \
                    action='store_true', dest='no_others')
parser.add_argument('-s', '--stride', help='write H5MD file every this many snapshots.', \
                    dest='stride', type=int)
parser.add_argument('-uw', '--unwrap', help='write unwrapped coordinates of particles in H5MD file.', \
                    action='store_true', dest='unwrap')
parser.add_argument('-i', '--image', help='store image flags of particles in H5MD file.', \
                    action='store_true', dest='image')
parser.add_argument('-q', '--quite', help='turn of printing information on screen.',\
                    action='store_true', dest='quite')
args = parser.parse_args()


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
    if not args.no_position:
        position_flag = True
else:
    if not args.no_position:
        sys.stdout.write('*** No position information found in dump file. Skip it. ***\n')
        sys.stdout.flush()

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

hdf_file = h5py.File(args.lammps_hdf5_dump, 'w')

# create `h5md` group
hdf_file.create_group('h5md')
hdf_file.create_group('h5md/author')
hdf_file.create_group('h5md/creator')

# create `observables` group
hdf_file.create_group('observables')

# create `parameters` group
hdf_file.create_group('parameters')

# create particles group
hdf_file.create_group('particles')
hdf_file.create_group('particles/all')
# create box group
hdf_file.create_group('particles/all/box')
hdf_file.create_group('particles/all/box/edges')

attribute_info_new = attribute_info[:]
for key in ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ix', 'iy', 'iz', 'xu', 'yu', 'zu']:
    try:
        attribute_info_new.remove(key)
    except ValueError:
        pass


if others_flag:
    for attribute in attribute_info_new:
        if attribute == 'type':
            hdf_file['particles/all/'].create_dataset('species', (natoms,), dtype='i4')
            continue
        elif attribute == 'q':
            attribute = 'charge'
        hdf_file.create_group('particles/all/'+attribute)

if position_flag:
    hdf_file.create_group('particles/all/position')
if velocity_flag:
    hdf_file.create_group('particles/all/velocity')
if image_flag:
    hdf_file.create_group('particles/all/image')

snap_index = 0 # keep track of the index of frames when reading
snap_index_write = 0 # keep track the actual number of snapshots written to H5MD file

start_time = time.time() # get the start time
sys.stdout.write('\033[1mStart to convert data...\033[0m\n')
sys.stdout.flush()
with open(args.lammps_custom_dump, 'r') as f:
    while True:
        next_n_lines = list(islice(f, number_lines_one_frame))

        if snap_index % stride == 0:
            pass
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

        # initiliaze the datasets otherwise resize the dataset
        if snap_index == 0:
            if others_flag:
                for attribute in attribute_info_new:
                    if attribute == 'type':
                        hdf_file['particles/all/species'][:] = np.int32(atom_info[:, attribute_info.index(attribute)])
                        continue
                    hdf_file['particles/all/'+attribute].create_dataset('value',\
                                (1, natoms), maxshape=(None, natoms), dtype='f8')
                    hdf_file['particles/all/'+attribute].create_dataset('step', \
                                (1,), maxshape=(None,), dtype='i4')
                    hdf_file['particles/all/'+attribute].create_dataset('time', \
                                (1,), maxshape=(None,), dtype='f8')
            if position_flag:
                hdf_file['particles/all/position'].create_dataset('value', \
                                                                  (1, natoms, 3),\
                                                                  maxshape=(None, natoms,3), \
                                                                  dtype='f8')
                hdf_file['particles/all/position'].create_dataset('step', (1,), \
                                                                  maxshape=(None,),\
                                                                  dtype='i4')
                hdf_file['particles/all/position'].create_dataset('time', (1,), \
                                                                  maxshape=(None,),\
                                                                  dtype='f8')
            if velocity_flag:
                hdf_file['particles/all/velocity'].create_dataset('value', \
                                                                  (1, natoms, 3), \
                                                                  maxshape=(None, natoms,3), \
                                                                  dtype='f8')
                hdf_file['particles/all/velocity'].create_dataset('step', (1,), \
                                                                  maxshape=(None,),\
                                                                  dtype='i4')
                hdf_file['particles/all/velocity'].create_dataset('time', (1,), \
                                                                  maxshape=(None,),\
                                                                  dtype='f8')
            if image_flag:
                hdf_file['particles/all/image'].create_dataset('value', \
                                                               (1, natoms, 3),\
                                                               maxshape=(None, natoms, 3),\
                                                               dtype='i4')
                hdf_file['particles/all/image'].create_dataset('step', (1,), \
                                                               maxshape=(None,),\
                                                               dtype='i4')
                hdf_file['particles/all/image'].create_dataset('time', (1,), \
                                                               maxshape=(None,),\
                                                               dtype='f8')
            hdf_file['particles/all/box/edges'].create_dataset('value', \
                                                               (1, box_shape[0], \
                                                                box_shape[1]), \
                                                               maxshape=(None, box_shape[0], box_shape[1]), \
                                                               dtype='f8')
            hdf_file['particles/all/box/edges'].create_dataset('step', (1,), \
                                                               maxshape=(None,), \
                                                               dtype='i4')
            hdf_file['particles/all/box/edges'].create_dataset('time', (1,),\
                                                               maxshape=(None,), \
                                                               dtype='f8')
        else:
            if others_flag:
                for attribute in attribute_info_new:
                    if attribute == 'type':
                        continue
                    hdf_file['particles/all/'+attribute]['value'].resize((snap_index_write+1, natoms))
                    hdf_file['particles/all/'+attribute]['step'].resize((snap_index_write+1,))
                    hdf_file['particles/all/'+attribute]['time'].resize((snap_index_write+1,))
            if position_flag:
                hdf_file['particles/all/position']['value'].resize((snap_index_write+1, natoms, 3))
                hdf_file['particles/all/position']['step'].resize((snap_index_write+1,))
                hdf_file['particles/all/position']['time'].resize((snap_index_write+1,))
            if velocity_flag:
                hdf_file['particles/all/velocity']['value'].resize((snap_index_write+1, natoms, 3))
                hdf_file['particles/all/velocity']['step'].resize((snap_index_write+1,))
                hdf_file['particles/all/velocity']['time'].resize((snap_index_write+1,))
            if image_flag:
                hdf_file['particles/all/image']['value'].resize((snap_index_write+1, natoms, 3))
                hdf_file['particles/all/image']['step'].resize((snap_index_write+1,))
                hdf_file['particles/all/image']['time'].resize((snap_index_write+1,))
            hdf_file['particles/all/box/edges']['value'].resize((snap_index_write+1,box_shape[0],box_shape[1]))
            hdf_file['particles/all/box/edges']['step'].resize((snap_index_write+1,))
            hdf_file['particles/all/box/edges']['time'].resize((snap_index_write+1,))


        # initial dataset
        if others_flag:
            for attribute in attribute_info_new:
                if attribute != 'type'
                    hdf_file['particles/all/'+attribute]['value'][snap_index_write] = np.float64(atom_info[:, attribute_info.index(attribute)])
                    hdf_file['particles/all/'+attribute]['step'][snap_index_write] = timestep
                    hdf_file['particles/all/'+attribute]['time'][snap_index_write] = timestep
        if position_flag:
            if unwrap_flag:
                hdf_file['particles/all/position']['value'][snap_index_write] = np.float64(atom_info[:,x_index:z_index+1]) + \
                np.float64(atom_info[:,ix_index:iz_index+1])*box_edge_size
            else:
                hdf_file['particles/all/position']['value'][snap_index_write] = np.float64(atom_info[:,x_index:z_index+1])
            hdf_file['particles/all/position']['step'][snap_index_write] = timestep
            hdf_file['particles/all/position']['time'][snap_index_write] = timestep
        if velocity_flag:
            hdf_file['particles/all/velocity']['value'][snap_index_write] = np.float64(atom_info[:,vx_index:vz_index+1])
            hdf_file['particles/all/velocity']['step'][snap_index_write] = timestep
            hdf_file['particles/all/velocity']['time'][snap_index_write] = timestep
        if image_flag:
            hdf_file['particles/all/image']['value'][snap_index_write] = np.int_(atom_info[:,ix_index:iz_index+1])
            hdf_file['particles/all/image']['step'][snap_index_write] = timestep
            hdf_file['particles/all/image']['time'][snap_index_write] = timestep
        hdf_file['particles/all/box/edges']['value'][snap_index_write] = np.float64(box)
        hdf_file['particles/all/box/edges']['step'][snap_index_write] = timestep
        hdf_file['particles/all/box/edges']['time'][snap_index_write] = timestep

        snap_index += 1
        snap_index_write += 1
        if not args.quite:
            sys.stdout.write("\rWriting snapshot #{}...".format(snap_index))
            sys.stdout.flush()
        hdf_file.flush()

end_time = time.time()
sys.stdout.write('\nTotal {} snapshots been written to H5MD file. Time used:{} mins\n'.format(snap_index_write, (end_time-start_time)/60))
sys.stdout.flush()
hdf_file.close()
