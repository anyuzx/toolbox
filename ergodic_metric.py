import numpy as np
import sys
import h5py
import time

class OnlineMean:
    """
    compute the sample mean in a one-pass fashion
    """
    def __init__(self, dim):
        if dim == 1:
            self.mean = 0.0
        else:
            self.mean = np.zeros(dim)
        self.n = 0.0

    def stream(self, data_point):
        self.n += 1.0
        self.delta = data_point - self.mean
        self.mean += self.delta/self.n

    def stat(self):
        return self.mean

energy_file_1 = sys.argv[1]
energy_file_2 = sys.argv[2]
foutname = sys.argv[3]
startsnap = int(sys.argv[4])


class LammpsH5MD:
    def __init__(self):
        self.file = self.filename = None
        self.timesteps = None

    def load(self, finname):
        try:
            self.filename = finname
            self.file = h5py.File(finname, 'r')
        except:
            print 'Load error\n'
            exit(0)

    def get_info(self):
        try:
            self.snapshot_step = self.file['particles/c_pe/step'][:]
            self.snapshot_number = self.snapshot_step.shape[0]
            self.n_atoms = len(self.file['particles/c_pe/value'][0])
        except:
            print 'ERROR\n'
            exit(0)

    def get_energy(self, timestep):
        pe_t = self.file['particles/c_pe/value'][timestep]
        ke_t = self.file['particles/c_pe/value'][timestep]
        return pe_t, ke_t, pe_t + ke_t

data1 = LammpsH5MD()
data2 = LammpsH5MD()

sys.stdout.write('Reading file 1...\n')
sys.stdout.flush()
data1.load(energy_file_1)
sys.stdout.write('Reading file 2...\n')
sys.stdout.flush()
data2.load(energy_file_2)

sys.stdout.write('Extracting necessary information...\n')
sys.stdout.flush()
data1.get_info()
sys.stdout.write('Extracting necessary information...\n')
sys.stdout.flush()
data2.get_info()

assert data1.snapshot_number == data2.snapshot_number
assert data1.n_atoms == data2.n_atoms

start = time.time()

# get energy average
snapshot_number = data1.snapshot_number
n_atoms = data1.n_atoms

ergodic_metric = np.zeros(snapshot_number - startsnap)
energy1_mean = OnlineMean(data1.n_atoms)
energy2_mean = OnlineMean(data2.n_atoms)
for t in np.arange(startsnap, snapshot_number):
    sys.stdout.write('Analyzing timestep {}...\n'.format(t))
    sys.stdout.flush()

    pe1, ke1, energy1 = data1.get_energy(t)
    pe2, ke2, energy2 = data2.get_energy(t)
    energy1_mean.stream(energy1)
    energy2_mean.stream(energy2)
    ergodic_metric[t - startsnap] = np.mean(np.power(energy1_mean.stat() - energy2_mean.stat(), 2))

ergodic_metric = np.dstack((np.arange(startsnap, snapshot_number), ergodic_metric))[0]
end = time.time()

sys.stdout.write('Run time: {:.2f} mins\n'.format((end-start)/60.0))
sys.stdout.flush()

with open(foutname, 'w') as fout:
    for item in ergodic_metric:
        fout.write('{:<15.2f}{:<15.5f}\n'.format(item[0], item[1]))
sys.stdout.write('Read to file {}\n'.format(foutname))
sys.stdout.flush()
