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

    def load(self, file):
        try:
            self.file = h5py.File(file, 'r')
        except:
            raise

    def get_endtime(self, keyword):
        endtime = self.file['particles/all/{}/time'.format(keyword)][-1]
        endtimestep = self.file['particles/all/{}/step'.format(keyword)][-1]

        return endtime, endtimestep

    def get_framenumber(self, keyword):
        try:
            return self.file['particles/all/{}/value'.format(keyword)].shape[0]
        except:
            raise

    def get_atomnumber(self, keyword):
        # get the total number of atoms/particles stored in file
        try:
            return self.file['particles/all/{}/value'.format(keyword)].shape[1]
        except:
            raise

    def get_frame(self, t, keyword):
        return self.file['particles/all/{}/value'.format(keyword)][t]

    def delete(self):
        self.file.close()

def mergetraj(filelst, foutname, stride, keyword_lst, check):
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

    for keyword in keyword_lst:
        new_file.create_group('particles/all/{}'.format(keyword))

    for index, fp in enumerate(filelst):
        try:
            traj0
        except NameError:
            traj0 = traj()
            traj0.load(fp)
            endtime0, endtimestep0 = traj0.get_endtime(keyword_lst[0])
            natoms0 = traj0.get_atomnumber(keyword_lst[0])
            framenum0 = traj0.get_framenumber(keyword_lst[0])
            framenum0_stride = len(np.arange(framenum0)[::stride[index]])

            lastframe = {}
            for keyword in keyword_lst:
                lastframe[keyword] = traj0.get_frame(-1, keyword)

            box_shape = traj0.file['particles/all/box/edges/value'].shape

            keyword_shape = {}
            for keyword in keyword_lst:
                keyword_shape['{}'.format(keyword)] = traj0.file['particles/all/{}/value'.format(keyword)].shape

            for keyword in keyword_lst:
                shape_temp = keyword_shape[keyword]

                if len(shape_temp) == 3:
                    new_file['particles/all/{}'.format(keyword)].create_dataset('value', \
                                                                     (framenum0_stride, shape_temp[1], shape_temp[2]),\
                                                                      maxshape=(None, shape_temp[1], shape_temp[2]), \
                                                                      dtype='f8')
                elif len(shape_temp) == 2:
                    new_file['particles/all/{}'.format(keyword)].create_dataset('value', \
                                                                     (framenum0_stride, shape_temp[1]),\
                                                                     maxshape=(None, shape_temp[1]),\
                                                                     dtype='f8')
                new_file['particles/all/{}'.format(keyword)].create_dataset('step', (framenum0_stride,), \
                                                                   maxshape=(None,),\
                                                                   dtype='i4')
                new_file['particles/all/{}'.format(keyword)].create_dataset('time', (framenum0_stride,), \
                                                                   maxshape=(None,),\
                                                                   dtype='f8')

            
            if len(box_shape) == 3:
                new_file['particles/all/box/edges'].create_dataset('value', \
                                                               (framenum0_stride, box_shape[1], \
                                                                box_shape[2]), \
                                                               maxshape=(None, box_shape[1], box_shape[2]), \
                                                               dtype='f8')
            elif len(box_shape) == 2:
                new_file['particles/all/box/edges'].create_dataset('value', \
                                                                  (framenum0_stride, box_shape[1]), \
                                                                  maxshape=(None, box_shape[1]),\
                                                                  dtype='f8')
            new_file['particles/all/box/edges'].create_dataset('step', (framenum0_stride,), \
                                                               maxshape=(None,), \
                                                               dtype='i4')
            new_file['particles/all/box/edges'].create_dataset('time', (framenum0_stride,),\
                                                               maxshape=(None,), \
                                                               dtype='f8')
            
            for keyword in keyword_lst:
                new_file['particles/all/{}/value'.format(keyword)][:] = traj0.file['particles/all/{}/value'.format(keyword)][::stride[index]]
                new_file['particles/all/{}/step'.format(keyword)][:] = traj0.file['particles/all/{}/step'.format(keyword)][::stride[index]]
                new_file['particles/all/{}/time'.format(keyword)][:] = traj0.file['particles/all/{}/time'.format(keyword)][::stride[index]]

            
            new_file['particles/all/box/edges/value'][:] = traj0.file['particles/all/box/edges/value'][::stride[index]]
            new_file['particles/all/box/edges/step'][:] = traj0.file['particles/all/box/edges/step'][::stride[index]]
            new_file['particles/all/box/edges/time'][:] = traj0.file['particles/all/box/edges/time'][::stride[index]]
            

            continue

        traj1 = traj()
        traj1.load(fp)
        endtime1, endtimestep1 = traj1.get_endtime(keyword_lst[0])
        natoms1 = traj1.get_atomnumber(keyword_lst[0])
        framenum1 = traj1.get_framenumber(keyword_lst[0])
        framenum1_stride = len(np.arange(framenum1)[::stride[index]])

        firstframe = {}
        for keyword in keyword_lst:
            firstframe[keyword] = traj1.get_frame(0, keyword)

        assert natoms1 == natoms0, "Number of atoms in two trajectories files are not the same."

        if check:
            for keyword in keyword_lst:
                assert np.sum(np.power(firstframe[keyword] - lastframe[keyword], 2.0)) <= 2.0, "The last snapshot of first trajectory is as the same as the first snapshot of the second trajectory."

        # resize dataset
        for keyword in keyword_lst:
            framenum_temp = new_file['particles/all/{}/value'.format(keyword)].shape[0]
            shape_temp = keyword_shape[keyword]
            if len(shape_temp) == 3:
                new_file['particles/all/{}/value'.format(keyword)].resize((framenum_temp+framenum1_stride-1, shape_temp[1], shape_temp[2]))
            elif len(shape_temp) == 2:
                new_file['particles/all/{}/value'.format(keyword)].resize((framenum_temp+framenum1_stride-1, shape_temp[1]))
            new_file['particles/all/{}/step'.format(keyword)].resize((framenum_temp+framenum1_stride-1,))
            new_file['particles/all/{}/time'.format(keyword)].resize((framenum_temp+framenum1_stride-1,))

        
        if len(box_shape) == 3:
            new_file['particles/all/box/edges/value'].resize((framenum_temp+framenum1-1,box_shape[1],box_shape[2]))
        elif len(box_shape) == 2:
            new_file['particles/all/box/edges/value'].resize((framenum_temp+framenum1-1,box_shape[1]))

        new_file['particles/all/box/edges/step'].resize((framenum_temp+framenum1-1,))
        new_file['particles/all/box/edges/time'].resize((framenum_temp+framenum1-1,))
        

        # append new data
        for keyword in keyword_lst:
            new_file['particles/all/{}/value'.format(keyword)][framenum_temp:] = traj1.file['particles/all/{}/value'.format(keyword)][::stride[index]][1::]
            new_file['particles/all/{}/step'.format(keyword)][framenum_temp:] = traj1.file['particles/all/{}/step'.format(keyword)][::stride[index]][1::] + endtimestep0
            new_file['particles/all/{}/time'.format(keyword)][framenum_temp:] = traj1.file['particles/all/{}/time'.format(keyword)][::stride[index]][1::] + endtime0

        
        new_file['particles/all/box/edges/value'][framenum_temp:] = traj1.file['particles/all/box/edges/value'][::stride[index]][1::]
        new_file['particles/all/box/edges/step'][framenum_temp:] = traj1.file['particles/all/box/edges/step'][::stride[index]][1::] + endtimestep0
        new_file['particles/all/box/edges/time'][framenum_temp:] = traj1.file['particles/all/box/edges/time'][::stride[index]][1::] + endtime0
        
        for keyword in keyword_lst:
            lastframe[keyword] = traj1.get_frame(-1, keyword)

        endtimestep0 = new_file['particles/all/{}/step'.format(keyword_lst[0])][-1]
        endtime0 = new_file['particles/all/{}/time'.format(keyword_lst[0])][-1]

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
    parser.add_argument('-k', '--keyword', help='provide keyword list.', dest='keyword_lst', nargs='*')
    parser.add_argument('-c', '--check', help='enable to check the continuity of files.', dest='check', action='store_true')
    args = parser.parse_args()

    if args.stride is None:
    	stride = np.int_(np.ones(len(args.input)))
    else:
    	stride = np.int_(args.stride)

    if args.keyword_lst is None:
        sys.stdout.write('ERROR: Please provide keyword\n')
        sys.stdout.flush()
        exit(0)

    mergetraj(args.input, args.output, stride, args.keyword_lst, args.check)
