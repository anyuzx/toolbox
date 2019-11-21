import numpy as np
import scipy
import scipy.spatial
import argparse
import h5py
import pandas as pd
from tqdm import tqdm
import MDAnalysis

# parse arguments
parser = argparse.ArgumentParser(description='Compute contact life time from xyz format trajectory file.')
parser.add_argument('xyz_traj', help='trajectory file in xyz format.')
parser.add_argument('pdb_file', help='PDB file associated with xyz trajectory.')
parser.add_argument('-c', '--cutoff', help='specify the threshold for determining contact.', \
                    dest='cutoff', type=float)
parser.add_argument('-o', '--output', help="specify the path to store the result as a Pandas dataframe.",\
                    dest="output")
parser.add_argument('-e', '--exclude', help="specify exclusion of nearest-x-number-neighbors", \
                    dest="exclude", type=int, default=2)
parser.add_argument('--with-start-time', help="enable outputing the ", \
                    dest="withStartTime", default=False, action='store_true')
args = parser.parse_args()


# define functions
def get_distance_array(coordinates):
    natoms = len(coordinates[0])
    nsteps = len(coordinates)
    distance_array = np.zeros((nsteps, natoms, natoms))

    for t in range(nsteps):
        distance_array[t] = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coordinates[t]))

    return distance_array

def get_contact_life_time(single_distance_array, cutoff):
    temp1 = np.where(single_distance_array <= cutoff)[0]
    temp2 = np.where(np.diff(temp1) != 1)[0] + 1
    if len(temp2) == 0:
        return np.array([len(temp1)])
    else:
        if temp1[0] == 0:
            return np.insert(np.diff(temp2), 0, temp2[0])[1:]
        else:
            return np.insert(np.diff(temp2), 0, temp2[0])

def get_contact_life_time_full(single_distance_array, cutoff):
    temp1 = np.where(single_distance_array <= cutoff)[0]
    temp2 = np.where(np.diff(temp1) != 1)[0] + 1
    if len(temp2) == 0:
        return np.array([len(temp1)]), np.array([len(temp1)])
    else:
        if temp1[0] == 0 and temp1[-1] == len(single_distance_array) - 1:
            return np.diff(temp2), temp1[temp2][:-1]
        elif temp1[0] != 0 and temp1[-1] == len(single_distance_array) - 1:
            return np.insert(np.diff(temp2), 0, temp2[0]), np.insert(temp1[temp2],0,temp1[0])[:-1]
        elif temp1[0] !=0 and temp1[-1] != len(single_distance_array) - 1:
            temp3 = np.insert(np.diff(temp2), 0, temp2[0])
            temp3 = np.append(temp3, len(temp1[temp2[-1]:]))
            return temp3, np.insert(temp1[temp2],0,temp1[0])
        elif temp1[0] == 0 and temp1[-1] != len(single_distance_array) - 1:
            temp3 = np.diff(temp2)
            temp3 = np.append(temp3, len(temp1[temp2[-1]:]))
            return temp3, temp1[temp2]


def get_contact_life_time_wrapper(coordinates, cutoff):
    distance_array = get_distance_array(coordinates)

    assert distance_array.shape[1] == distance_array.shape[2]
    nsteps = distance_array.shape[0]
    natoms = distance_array.shape[1]

    for i in tqdm(range(natoms-args.exclude)):
        for j in range(i+args.exclude, natoms):
            temp_result1 = get_contact_life_time(distance_array[:,i,j], cutoff)
            temp_array = np.repeat(np.array([[i,j]]), len(temp_result1), axis=0)
            temp_result2 = np.hstack((temp_array, temp_result1.reshape(len(temp_result1), 1)))
            try:
                contact_life_data = np.vstack((contact_life_data, temp_result2))
            except NameError:
                contact_life_data = temp_result2
    return contact_life_data


def get_contact_life_time_wrapper_full(coordinates, cutoff):
    distance_array = get_distance_array(coordinates)

    assert distance_array.shape[1] == distance_array.shape[2]
    nsteps = distance_array.shape[0]
    natoms = distance_array.shape[1]

    for i in tqdm(range(natoms-args.exclude)):
        for j in range(i+args.exclude, natoms):
            temp_result1, temp_result2 = get_contact_life_time_full(distance_array[:,i,j], cutoff)
            temp_array = np.repeat(np.array([[i,j]]), len(temp_result1), axis=0)
            temp_result2 = np.hstack((temp_array, \
                                      temp_result1.reshape(len(temp_result1), 1), \
                                      temp_result2.reshape(len(temp_result2), 1)))
            try:
                contact_life_data = np.vstack((contact_life_data, temp_result2))
            except NameError:
                contact_life_data = temp_result2
    return contact_life_data
# ------------------------------------

traj = MDAnalysis.Universe(args.pdb_file, args.xyz_traj)
coordinates = []
for frame in traj.trajectory[:]:
    coordinates.append(np.copy(frame.positions))
coordinates = np.array(coordinates)

contact_life_data = get_contact_life_time_wrapper_full(coordinates, args.cutoff)

if args.withStartTime:
    contact_life_data_df = pd.DataFrame(contact_life_data, columns=['i','j','t','t-start'])
else:
    contact_life_data_df = pd.DataFrame(contact_life_data[:,:3], columns=['i','j','t'])

if args.output:
    contact_life_data_df.to_csv(args.output, index=False)
