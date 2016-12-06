import sys
import numpy as np
import h5py
import glob
import argparse

def memory_usage_resource():
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

class traj:
    def __init__(self):
        self.file = None
        self.starttime = None
        self.starttimestep = None
        self.endtime = None
        self.endtimestep = None

    def load(self, file):
        try:
            self.file = h5py.File(file, 'r')
        except:
            raise

    def get_firsttime(self):
        #starttime = self.file['particles/all/position/time'][0]
        endtime = self.file['particles/all/position/time'][-1]
        #starttimestep = self.file['particles/all/position/step'][0]
        endtimestep = self.file['particles/all/position/step'][-1]

        return endtime, endtimestep

    def get_framenumber(self):
        try:
            return self.file['particles/all/position/value'].shape[0]
        except:
            raise

    def get_atomnumber(self):
        # get the total number of atoms/particles stored in file
        try:
            return self.file['particles/all/position/value'].shape[1]
        except:
            raise

    def get_frame(self,t):
        return self.file['particles/all/position/value'][t]

    def delete(self):
        self.file.close()

def mergetraj(filelst, foutname, stride):
    new_file = h5py.File(foutname, 'w')
    # create `h5md` group
    new_file.create_group('h5md')
    new_file.create_group('h5md/author')
    new_file.create_group('h5md/creator')

    # create `observables` group
    new_file.create_group('observables')

    # create `parameters` group
    new_file.create_group('parameters')

    # create particles group
    new_file.create_group('particles')
    new_file.create_group('particles/all')
    # create box group
    new_file.create_group('particles/all/box')
    new_file.create_group('particles/all/box/edges')

    new_file.create_group('particles/all/position')

    for index, fp in enumerate(filelst):
        try:
            traj0
        except NameError:
            traj0 = traj()
            traj0.load(fp)
            endtime0, endtimestep0 = traj0.get_firsttime()
            natoms0 = traj0.get_atomnumber()
            framenum0 = traj0.get_framenumber()
            framenum0_stride = len(np.arange(framenum0)[::stride[index]])
            lastframe = traj0.get_frame(-1)

            box_shape = traj0.file['particles/all/box/edges/value'].shape

            new_file['particles/all/position'].create_dataset('value', \
                                                             (framenum0_stride, natoms0, 3),\
                                                              maxshape=(None, natoms0,3), \
                                                              dtype='f8')
            new_file['particles/all/position'].create_dataset('step', (framenum0_stride,), \
                                                               maxshape=(None,),\
                                                               dtype='i4')
            new_file['particles/all/position'].create_dataset('time', (framenum0_stride,), \
                                                               maxshape=(None,),\
                                                               dtype='f8')

            '''
            if len(box_shape) == 3:
                new_file['particles/all/box/edges'].create_dataset('value', \
                                                               (framenum0, box_shape[1], \
                                                                box_shape[2]), \
                                                               maxshape=(None, box_shape[1], box_shape[2]), \
                                                               dtype='f8')
            elif len(box_shape) == 2:
                new_file['particles/all/box/edges'].create_dataset('value', \
                                                                  (framenum0, box_shape[1]), \
                                                                  maxshape=(None, box_shape[1]),\
                                                                  dtype='f8')
            new_file['particles/all/box/edges'].create_dataset('step', (framenum0,), \
                                                               maxshape=(None,), \
                                                               dtype='i4')
            new_file['particles/all/box/edges'].create_dataset('time', (framenum0,),\
                                                               maxshape=(None,), \
                                                               dtype='f8')
            '''

            new_file['particles/all/position/value'][:] = traj0.file['particles/all/position/value'][::stride[index]]
            new_file['particles/all/position/step'][:] = traj0.file['particles/all/position/step'][::stride[index]]
            new_file['particles/all/position/time'][:] = traj0.file['particles/all/position/time'][::stride[index]]

            '''
            new_file['particles/all/box/edges/value'][:] = traj0.file['particles/all/box/edges/value'][:]
            new_file['particles/all/box/edges/step'][:] = traj0.file['particles/all/box/edges/step'][:]
            new_file['particles/all/box/edges/time'][:] = traj0.file['particles/all/box/edges/time'][:]
            '''

            continue

        traj1 = traj()
        traj1.load(fp)
        endtime1, endtimestep1 = traj1.get_firsttime()
        natoms1 = traj1.get_atomnumber()
        framenum1 = traj1.get_framenumber()
        framenum1_stride = len(np.arange(framenum1)[::stride[index]])
        firstframe = traj1.get_frame(0)

        assert natoms1 == natoms0
        assert np.sum(np.power(firstframe - lastframe, 2.0)) <= 2.0

        # resize dataset
        framenum_temp = new_file['particles/all/position/value'].shape[0]
        new_file['particles/all/position/value'].resize((framenum_temp+framenum1_stride-1, natoms1, 3))
        new_file['particles/all/position/step'].resize((framenum_temp+framenum1_stride-1,))
        new_file['particles/all/position/time'].resize((framenum_temp+framenum1_stride-1,))

        '''
        if len(box_shape) == 3:
            new_file['particles/all/box/edges/value'].resize((framenum_temp+framenum1-1,box_shape[1],box_shape[2]))
        elif len(box_shape) == 2:
            new_file['particles/all/box/edges/value'].resize((framenum_temp+framenum1-1,box_shape[1]))

        new_file['particles/all/box/edges/step'].resize((framenum_temp+framenum1-1,))
        new_file['particles/all/box/edges/time'].resize((framenum_temp+framenum1-1,))
        '''

        # append new data
        new_file['particles/all/position/value'][framenum_temp:] = traj1.file['particles/all/position/value'][::stride[index]][1::]
        new_file['particles/all/position/step'][framenum_temp:] = traj1.file['particles/all/position/step'][::stride[index]][1::] + endtimestep0
        new_file['particles/all/position/time'][framenum_temp:] = traj1.file['particles/all/position/time'][::stride[index]][1::] + endtime0

        '''
        new_file['particles/all/box/edges/value'][framenum_temp:] = traj1.file['particles/all/box/edges/value'][1:]
        new_file['particles/all/box/edges/step'][framenum_temp:] = traj1.file['particles/all/box/edges/step'][1:] + endtimestep0
        new_file['particles/all/box/edges/time'][framenum_temp:] = traj1.file['particles/all/box/edges/time'][1:] + endtime0
        '''

        lastframe = traj1.get_frame(-1)

        endtimestep0 = new_file['particles/all/position/step'][-1]
        endtime0 = new_file['particles/all/position/time'][-1]

        #starttime0, endtime0, starttimestep0, endtimestep0 = starttime1, endtime1, starttimestep1, endtimestep1
        framenum0 = framenum1
        natoms0 = natoms1

        traj1.delete()
        del traj1

    new_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge multiple trajectories file together.')
    parser.add_argument('-in', '--input', help='list of trajectory files.', dest='input', nargs='*')
    parser.add_argument('-out', '--output', help='path of output trajectory file.', dest='output')
    parser.add_argument('-s', '--stride', help='stride option.provided as list', dest='stride', nargs='*')
    args = parser.parse_args()

    if args.stride is None:
    	stride = np.int_(np.ones(len(args.input)))
    else:
    	stride = np.int_(args.stride)

    mergetraj(args.input, args.output, stride)