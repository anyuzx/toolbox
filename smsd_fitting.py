import numpy as np
import tables
import h5py
import pandas as pd
import argparse
import sys
import os
import time

import scipy
from scipy.optimize import curve_fit

# DEFINE TIME UNIT CONVERSION AND LENGTH UNIT CONVERSION
tau = 4.3097*10**-8
sigma = 0.07 # micro meter


# First declare several flags
parser = argparse.ArgumentParser(description='Compute the diffusion parameters from single MSD data.')
parser.add_argument('sMSD_file', help='single particle MSD data file.')
parser.add_argument('output_file', help='output file.')
parser.add_argument('-l', '--log',  help='output to log files.',dest='logfile')
args = parser.parse_args()

# redirect stdout to log file if specified
if args.logfile:
    sys.stdout = open(args.logfile, 'w')


def power_law(t, D, alpha):
    return D*np.power(t, alpha)


def fitting(data):
    index = np.where((data[:,0] >= 0.1/tau) & (data[:,0] <= 10.0/tau))
    effective_fitting_data = data[index]
    popt, pcov = curve_fit(power_law, effective_fitting_data[:,0], effective_fitting_data[:,1])
    return popt, pcov


def create_dataframe(data, particle_index):
    array = data['data'][:, np.array([0, 1, particle_index+1])]
    t0 = array[:, 0]
    t1 = array[:, 1]
    quantity = array[:, 2]
    dt = compute_dt(t0, t1)
    traj_column = np.full(len(dt), 1)
    df_array = np.dstack((traj_column, t0, t1, dt, quantity))[0]
    df = pd.DataFrame(df_array, columns = ['traj', 't0', 't1', 'dt', 'quantity'])
    return df


def ComputeAverage2(dataframe, t0_max, count):
    df_temp = dataframe[dataframe['t0'] <= t0_max]

    stats = df_temp[['traj','t0','dt','quantity']].groupby(['dt']).agg(['mean', 'var', 'count'])
    stats = stats[stats['quantity']['count'] >= count]

    mean = np.column_stack((stats.index.values, stats['quantity']['mean'].values))
    var = np.column_stack((stats.index.values, stats['quantity']['var'].values))

    return mean, var


def compute_dt(t1_lst, t2_lst):
    assert len(t1_lst) == len(t2_lst)

    dt = np.zeros(len(t1_lst))

    for i in range(len(t1_lst)):
        t1 = t1_lst[i]
        t2 = t2_lst[i]

        if t1 < 20000 and t2 > 20000:
            dt_temp = (20000-t1)*3.0 + (t2 - 20000)*10000.0*3.0
        elif t2 <= 20000 and t1 <= 20000:
            dt_temp = (t2-t1)*3.0
        elif t1 >= 20000 and t2 > 20000:
            dt_temp = (t2-t1)*10000.0*3.0

        dt[i] = dt_temp
    return dt


def fitting_from_data(file_name, stream = None):
    if stream is None:
        stream = sys.stdout

    start_time = time.time()

    diffusion_parameter = []
    data = []
    fp = h5py.File(file_name, 'r')
    temp_df = []
    for particle_index in np.arange(1,1001):
        stream.write('particle {} being analyzed\n'.format(particle_index))
        stream.flush()
        tmp = create_dataframe(fp, particle_index)
        temp_df.append(tmp)


    data_temp = []
    for df in temp_df:
        mean_tmp, var_tmp = ComputeAverage2(df, 45000, 1990)
        data_temp.append(mean_tmp)

    data.append(data_temp)

    for index, p in enumerate(data_temp):
        pcov, popt = fitting(p)
        diffusion_parameter.append(pcov)
        stream.write('Computing the particle {}.\n'.format(index+1))
        stream.flush()

    end_time = time.time()
    stream.write('\nTotal {} particles have been analyzed. Time used:{} mins\n'.format(index+1, (end_time-start_time)/60))
    stream.flush()

    fp.close()
    del fp

    diffusion_parameter = np.array(diffusion_parameter)
    return diffusion_parameter, data

sys.stdout.write(args.sMSD_file+'\n')
sys.stdout.flush()

diffusion_parameter, data = fitting_from_data(args.sMSD_file)
np.save(args.output_file, diffusion_parameter)
np.save('data_'+args.output_file, data)
