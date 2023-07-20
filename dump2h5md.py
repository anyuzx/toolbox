import numpy as np
import h5py
from rich import print
from rich.progress import Progress
from io import BytesIO
import time
from itertools import islice
import click
import os

def str2array(str):
    #str_temp = str.decode('unicode-escape')
    strIO = BytesIO(str.encode())
    return np.loadtxt(strIO)


######################################
@click.command()
@click.argument("lammps_custom_dump")
@click.argument("lammps_hdf5_dump")
@click.option("--no-position", is_flag=True, default=False, help="disable writing position to H5MD file.")
@click.option("--no-velocity", is_flag=True, default=False, help="disable writing velocity to H5MD file.")
@click.option("--no-others", is_flag=True, default=False, help="disable writing other informations to H5MD file.")
@click.option("--stride", default=1, type=int, help="write H5MD file every this many snapshots.")
@click.option("--begin", default=None, type=int, help="write H5MD file starting from this index.")
@click.option("--terminate", default=None, type=int, help="stop write H5MD file starting from this index.")
@click.option("--unwrap", is_flag=True, default=False, help="write unwrapped coordinates of particles in H5MD file.")
@click.option("--image", is_flag=True, default=False, help="store image flags of particles in H5MD file.")
@click.option("--quiet", is_flag=True, default=False, help="turn off printing information on screen.")
@click.option("--log", "logfile", default=None, help="output to log files.")
def main(lammps_custom_dump, lammps_hdf5_dump, no_position, no_velocity, no_others, stride, begin, terminate, unwrap, image, quiet, logfile):
    position_flag = False
    velocity_flag = False
    others_flag = True
    unwrap_flag = False
    image_flag = False

    # report error if both quiet and logfile are required
    if quiet and logfile:
        print('[red]ERROR: Both quiet and log argument are specified. Program terminated.')
        sys.exit(0)

    # redirect stdout to log file if specified
    if logfile:
        sys.stdout = open(logfile, 'w')
    elif quiet:
        sys.stdout = open(os.devnull, 'w')

    if stride is None:
        stride = 1
    else:
        stride = stride

    # first read number of atoms in the dump file
    # read atom attribute information
    check = False
    with open(lammps_custom_dump) as f:
        for i, line in enumerate(f):
            if line == 'ITEM: NUMBER OF ATOMS\n':
                check = True
                continue
            if check:
                natoms = np.int_(line.split()[0])
                check = False
            if 'ITEM: BOX BOUNDS' in line:
                if not np.any(['p' in line.split()[i] for i in range(3,len(line.split()))]):
                    if unwrap:
                        print("No periodic boundary found. Ignore argument '--unwrap'.")
                    if image:
                        print("No periodic boundary found. Ignore argument '--image'.")
                else:
                    if not unwrap and not image:
                        print("[yellow]WARNING: Periodic boundary found. Neither argument '--unwrap' nor '--image' are provided.")
            if 'ITEM: ATOMS' in line:
                attribute_info = line.split()[2:]
                break

    # check attribute information
    if 'id' in attribute_info:
        id_index = attribute_info.index('id')
    else:
        print('[red]ERROR: No particle ID is found in dump file. \
    Make sure that the order of particles does not change in dump file.')

    if 'x' in attribute_info and 'y' in attribute_info and 'z' in attribute_info:
        x_index = attribute_info.index('x')
        y_index = attribute_info.index('y')
        z_index = attribute_info.index('z')
        if not no_position:
            position_flag = True
    else:
        if not no_position:
            print('[yellow]No position information found in dump file. Skip it.')

    if 'vx' in attribute_info and 'vy' in attribute_info and 'vz' in attribute_info:
        vx_index = attribute_info.index('vx')
        vy_index = attribute_info.index('vy')
        vz_index = attribute_info.index('vz')
        if not no_velocity:
            velocity_flag = True
    else:
        if not no_velocity:
            print('[yellow]No velocity information found in dump file. Skip it.')

    if 'ix' in attribute_info and 'iy' in attribute_info and 'iz' in attribute_info:
        ix_index = attribute_info.index('ix')
        iy_index = attribute_info.index('iy')
        iz_index = attribute_info.index('iz')
        if not unwrap and not image:
            print('[red]WARNING: Image flags found in dump file.')
        unwrap_flag = unwrap
        image_flag = image
    else:
        if unwrap or image:
            print('[yellow]No image information found in dump file. Skip it.')

    if 'xu' in attribute_info and 'yu' in attribute_info and 'zu' in attribute_info:
        print('[red]WARNING: Unwrapped position found in dump file.')

    number_lines_one_frame = 9 + natoms # 9 = number of head lines for each frame

    hdf_file = h5py.File(lammps_hdf5_dump, 'w')

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
    print(f'Start to convert file [blue]{lammps_custom_dump}[/blue]...')

    with open(lammps_custom_dump, 'r') as f:
        with Progress() as progress:
            task = progress.add_task("[cyan]Writing snapshots 0...", total=None)
            while True:
                next_n_lines = list(islice(f, number_lines_one_frame))

                # enumerate all the posiibilities
                if begin is None and terminate is None:
                    if snap_index % stride == 0:
                        pass
                    else:
                        snap_index += 1
                        continue
                elif begin is not None and terminate is None:
                    if snap_index >= begin:
                        if snap_index % stride == 0:
                            pass
                        else:
                            snap_index += 1
                            continue
                    else:
                        snap_index += 1
                        continue
                elif begin is None and terminate is not None:
                    if snap_index <= terminate:
                        if snap_index % stride == 0:
                            pass
                        else:
                            snap_index += 1
                            continue
                    else:
                        break
                elif begin is not None and terminate is not None:
                    if snap_index >= begin and snap_index <= terminate:
                        if snap_index % stride == 0:
                            pass
                        else:
                            snap_index += 1
                            continue
                    elif snap_index > terminate:
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
                    atom_info = atom_info[atom_info[:,id_index].rt()]
                except:
                    pass

                # initiliaze the datasets otherwise resize the dataset
                if snap_index_write == 0:
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
                        if attribute != 'type':
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
                """
                if logfile:
                    print("Writing snapshot #{}\n".format(snap_index))
                else:
                    print("\rWriting snapshot #{}...".format(snap_index))
                """
                progress.update(task, advance=1, description=f"[cyan]Writing snapshot {snap_index}")
                hdf_file.flush()

        end_time = time.time()
        print(f'Total {snap_index_write} snapshots been written to H5MD file. Time used:{(end_time - start_time)/60.0:.2f} mins')
        hdf_file.close()

if __name__ == "__main__":
    main()
