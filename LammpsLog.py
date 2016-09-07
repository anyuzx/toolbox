import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os

class LammpsLog:
    def __init__(self, *file_lst):
        self.data = {}
        if len(file_lst) == 0:
            raise ValueError('No files provided\n')

        for fp in file_lst:
            file_name = os.path.basename(fp)
            self.data[file_name] = read_log_file(fp)

    def plot(self, file_name, section_index=0, foutname=None):
        df = self.data[file_name][section_index]
        number_attribute = len(df.columns)
        if 'Step' not in df.columns:
            raise ValueError('Step is not in attribute. Cannot plot.\n')
        fig, ax = plt.subplots(number_attribute - 1, sharex=True, sharey=False,figsize=(10,40))
        for index, ylabel in enumerate(df.columns[1:]):
            ax[index].plot(df['Step'].values, df[ylabel].values)
            ax[index].set_ylabel(ylabel)
            ax[index].set_ylim(np.min(df[ylabel].values), np.max(df[ylabel].values))
        ax[-1].set_xlabel('Step')
        ax[-1].set_xlim(df['Step'].values[1], df['Step'].values[-1])
        ax[-1].set_xscale('log')
        if foutname is None:
            plt.show()
        else:
            if '.png' in foutname:
                plt.savefig(foutname, dpi=300)
            else:
                plt.savefig(foutname+'.png', dpi=300)


###############################################################################
"""
Function read_log_file read Log file
"""
def read_log_file(fp):
    count = 0
    record = False
    data = []
    headers = []
    with open(fp, 'r') as f:
        sys.stdout.write('Reading file {}\n'.format(fp))
        sys.stdout.flush()
        for i, line in enumerate(f):
            try:
                if 'Step' in line.split():
                    count += 1
                    data.append([])
                    header = line.split()
                    headers.append(header)
                    record = True
                    continue
                if 'Loop time of' in line and record:
                    record = False
                    continue
                if record:
                    try:
                        data[-1].append(np.float32(line.split()))
                    except ValueError:
                        continue
            except IndexError:
                continue

    for index, section in enumerate(data):
        data[index] = pd.DataFrame(np.array(data[index]), columns=headers[index])

    return data
