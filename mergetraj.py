import sys
import numpy as np
import h5py
import glob
import argparse

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
		starttime = self.file['particles/all/position/time'][0]
		endtime = self.file['particles/all/position/time'][-1]
		starttimestep = self.file['partciles/all/position/step'][0]
		endtimestep = self.file['particles/all/position/step'][-1]

		return starttime, endtime, starttimestep, endtimestep

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
		return self.file['particles/all/positon/value'][t]

	def delete(self):
		self.file.close()

def mergetraj(filelst, foutname):
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

	for fp in filelst:
		try:
			traj0
		except NameError:
			traj0 = traj()
			traj0.load(fp)
			starttime0, endtime0, starttimestep0, endtimestep0 = traj0.get_firsttime()
			natoms0 = traj0.get_atomnumber()
			framenum0 = traj0.get_framenumber()
			lastframe = traj0.get_frame(-1)

			box_shape = traj0['particles/all/box/edges/value'].shape

			new_file['particles/all/position'].create_dataset('value', \
                                                             (framenum0, natoms, 3),\
                                                              maxshape=(None, natoms,3), \
                                                              dtype='f8')
            new_file['particles/all/position'].create_dataset('step', (framenum0,), \
                                                               maxshape=(None,),\
                                                               dtype='i4')
            new_file['particles/all/position'].create_dataset('time', (framenum0,), \
                                                               maxshape=(None,),\
                                                               dtype='f8')

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

            new_file['particles/all/position/value'][:] = traj0['particles/all/position/value'][:]
            new_file['particles/all/position/step'][:] = traj0['particles/all/position/step'][:]
            new_file['particles/all/position/time'][:] = traj0['particles/all/position/time'][:]

            new_file['particles/all/box/edges/value'][:] = traj0['particles/all/box/edges/value'][:]
            new_file['particles/all/box/edges/step'][:] = traj0['particles/all/box/edges/step'][:]
            new_file['particles/all/box/edges/time'][:] = traj0['particles/all/box/edges/time'][:]

            continue

        traj1 = traj()
        traj1.load(fp)
        starttime1, endtime1, starttimestep1, endtimestep1 = traj1.get_firsttime()
        natoms1 = traj1.get_atomnumber()
        framenum1 = traj1.get_framenumber()
        firstframe = traj1.get_frame(-1)

        assert natoms1 == natoms0
        assert np.all(firstframe == lastframe)

        # resize dataset
        framenum_temp = new_file['particles/all/position/value'].shape[0]
        new_file['particles/all/position/value'].resize((framenum_temp+framenum1-1, natoms1))
        new_file['particles/all/position/step'].resize((framenum_temp+framenum1-1,))
        new_file['particles/all/position/time'].resize((framenum_temp+framenum1-1,))

        if len(box_shape) == 3:
        	new_file['particles/all/box/edges/value'].resize((framenum_temp+framenum1-1,box_shape[1],box_shape[2]))
        elif len(box_shape) == 2:
        	new_file['particles/all/box/edges/value'].resize((framenum_temp+framenum1-1,box_shape[1]))

        new_file['particles/all/box/edges/step'].resize((framenum_temp+framenum1-1,))
        new_file['particles/all/box/edges/time'].resize((framenum_temp+framenum1-1,))

        # append new data
        new_file['particles/all/position/value'][framenum_temp:] = traj1['particles/all/position/value'][1:]
        new_file['particles/all/position/step'][framenum_temp:] = traj1['particles/all/position/step'][1:] + endtimestep0
        new_file['particles/all/position/time'][framenum_temp:] = traj1['particles/all/position/time'][1:] + endtime0

        new_file['particles/all/box/edges/value'][framenum_temp:] = traj1['particles/all/box/edges/value'][1:]
        new_file['particles/all/box/edges/step'][framenum_temp:] = traj1['particles/all/box/edges/step'][1:] + endtimestep0
        new_file['particles/all/box/edges/time'][framenum_temp:] = traj1['particles/all/box/edges/time'][1:] + endtime0

        lastframe = traj1.get_frame(-1)

        starttime0, endtime0, starttimestep0, endtimestep0 = starttime1, endtime1, starttimestep1, endtimestep1
        framenum0 = framenum1
        natoms0 = natoms1

        traj1.delete()
        del traj1

        new_file.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Merge multiple trajectories file together.')
	parser.add_argument('-in', '--input', help='list of trajectory files.', dest='input', nargs='*')
	parser.add_argument('-out', '--output', help='path of output trajectory file.', dest='output')
	args = parser.parse_args()

	mergetraj(args.input, args.output)