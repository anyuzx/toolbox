import sys
import numpy as np
import h5py
import glob
import argparse


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

def reordertraj(fin, fout, order):
    new_file = h5py.File(fout, 'w')

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


