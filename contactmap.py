import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import _matrixnorm
import pandas as pd
import datetime
import sys
import os

class contactmap:
    def __init__(self, chrom, start, end, bin_size):
        self.map = None
        self.normalized_map = None
        self.OE_map = None
        self.PearsonCoeff_map = None
        self.zscore_map = None
        self.contact_probability = None
        self.subchain_contact = None
        self.chrom = chrom
        self.start = start
        self.end = end
        self.bin_size = bin_size
        self.norm_factor = None
        self.norm_factor_OE = None
        self.norm_factor_coeff = None
        self.norm_factor_zscore = None

    def read(self, fp):
        if '.npy' in fp:
            self.map = np.load(fp)
            self.map = np.float32(self.map)
        else:
            self.map = np.loadtxt(fp, dtype=np.float32)

        self.map = self.map + self.map.T - np.diag(self.map.diagonal()) # make it symmetry if not

    def get_contact_prob(self, bin_method='linear', number_bins=None):
        if self.map is None:
            raise ValueError('No contact map present\n')

        n = self.map.shape[0]
        if bin_method == 'linear':
            if number_bins is None:
                number_bins = n - 1
            bin_lst = np.unique(np.linspace(1, n-1, number_bins, dtype = np.int))
        elif bin_method == 'log' or bin_method == 'logarithm':
            bin_lst = np.unique(np.int_(np.floor(np.exp(np.linspace(np.log(1),np.log(n-1),number_bins)))))

        contact_count = np.zeros(n-1)
        for i in range(n-1):
            for j in range(i+1, n):
                contact_count[j-i-1] += self.map[i, j]

        norm_factor = np.arange(1,n)[::-1]
        contact_prob_temp = []
        for i in range(len(bin_lst)):
                edge1 = bin_lst[i] - 1
                if i == len(bin_lst) - 1:
                    edge2 = edge1 + 1
                else:
                    edge2 = bin_lst[i+1] - 1
                norm = np.sum(norm_factor[edge1:edge2])
                contacts = np.sum(contact_count[edge1:edge2])
                contact_prob_temp.append(float(contacts)/float(norm))

        contact_prob_temp = np.array(contact_prob_temp)
        contact_prob_temp = contact_prob_temp/np.sum(contact_prob_temp)
        self.contact_probability = np.float32(np.column_stack((bin_lst, contact_prob_temp)))
        return self.contact_probability

    def get_subchain_contact(self, mode='normalize'):
        if mode == 'normalize':
            if self.normalized_map is None:
                raise ValueError('No normalized contact map present\n')
        else:
            if self.map is None:
                raise ValueError('No contact map present\n')

        if mode == 'normalize':
            n = self.normalized_map.shape[0]
        else:
            n = self.map.shape[0]
        s_lst = np.arange(1, n+1)

        if mode == 'normalize':
            self.subchain_contact = _matrixnorm.matrixnorm_subchain_contact(self.normalized_map)
        else:
            self.subchain_contact = _matrixnorm.matrixnorm_subchain_contact(self.map)

        #self.subchain_contact = np.zeros(n, dtype=np.float32)
        #for i in range(n):
        #    for j in range(i, n):
        #        sys.stdout.write('\rindex ({},{})'.format(i,j))
        #        contact_part1 = np.sum(self.map[i:j+1, :i+1])
        #        contact_part2 = np.sum(self.map[j:, i:j+1])
        #        self.subchain_contact[j-i] += (contact_part1 + contact_part2)/(n-(j-i))

        self.subchain_contact = np.float32(np.column_stack((s_lst, self.subchain_contact)))
        return self.subchain_contact

    def normalize(self, norm_factor):
        self.norm_factor = norm_factor
        assert type(norm_factor) == int
        self.normalized_map = _matrixnorm.matrixnorm(self.map, norm_factor)
        self.normalized_map += self.normalized_map.T - - np.diag(self.normalized_map.diagonal())
        return self.normalized_map

    def get_OE(self, norm_factor):
        self.norm_factor_OE = norm_factor
        assert type(norm_factor) == int
        self.OE_map = _matrixnorm.matrixnorm_OE(self.map, norm_factor)
        self.OE_map += self.OE_map.T - np.diag(self.OE_map.diagonal())
        return self.OE_map

    def get_zscore(self, norm_factor):
        self.norm_factor_zscore = norm_factor
        assert type(norm_factor) == int
        self.zscore_map = _matrixnorm.matrixnorm_zscore(self.map, norm_factor)
        self.zscore_map += self.zscore_map.T - np.diag(self.zscore_map.diagonal())
        return self.zscore_map

    def get_PearsonCoeff(self, norm_factor):
        assert type(norm_factor) == int
        self.norm_factor_coeff = norm_factor
        if self.zscore_map is None:
            self.get_zscore(norm_factor)

        self.PearsonCoeff_map = _matrixnorm.matrixnorm_correlation(self.zscore_map)
        self.PearsonCoeff_map += self.PearsonCoeff_map.T - np.diag(self.PearsonCoeff_map.diagonal())
        return self.PearsonCoeff_map

    def plot_map(self, fp, color_range=[0.0,1.0], mode=None, state_mode='two state'):
        try:
            path = os.path.dirname(__file__)
            encode_state = get_state(os.path.join(path, 'wgEncodeBroadHmmGm12878HMM.bed'), chrom=self.chrom,\
                                     start=self.start, end=self.end, bin_size=self.bin_size,\
                                     data_source='Encode_Chrom_State', \
                                     mode=state_mode)
            subcompartment_state = get_state(os.path.join(path, 'GSE63525_GM12878_subcompartments.bed'),\
                                             chrom=self.chrom, start=self.start, \
                                             end=self.end, bin_size=self.bin_size,\
                                             data_source='Cell_Subcompartment')
        except IOError:
            sys.stdout.write('Could not find EncodeHMM bed file and Subcompartment bed file\n')
            sys.exit(0)

        if mode is None:
            mode_flag = 1
            norm = 1
            map_temp = self.map
        elif mode == 'normalize':
            mode_flag = 2
            norm = self.norm_factor
            map_temp = self.normalized_map
        elif mode == 'observe/expected':
            mode_flag = 3
            norm = self.norm_factor_OE
            map_temp = self.OE_map
        elif mode == 'pearson coeff':
            mode_flag = 4
            norm = self.norm_factor_coeff
            map_temp = self.PearsonCoeff_map

        encode_state_lst = []
        subcompartment_state_lst = []


        if state_mode == 'two state':
            encode_state_lst.append(np.array(encode_state['state 1'])/norm)
            encode_state_lst.append(np.array(encode_state['state 2'])/norm)
        elif state_mode == 'encode original':
            for state_index in range(1,16):
                encode_state_lst.append(np.array(encode_state['state {}'.format(state_index)])/norm)
        subcompartment_state_lst.append(np.array(subcompartment_state['state 1'])/norm)
        subcompartment_state_lst.append(np.array(subcompartment_state['state 2'])/norm)

        cdict = {'red': [(0.0, 1.0, 1.0),
                        (1.0, 1.0, 1.0)],
                 'green': [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)],
                 'blue': [(0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)],
                 'alpha': [(0.0, 0.0, 0.0),
                         (1.0, 1.0, 1.0)]}

        rvb = mcolors.LinearSegmentedColormap('CustomMap', cdict)

        fig, ax = plt.subplots()
        n = self.map.shape[0] # used to make plot aspect appropriate
        baroffset = (1.0/20.0) * (n/norm)
        barlength = 0.88 * baroffset
        barwidth = (1.0/norm)
        # plot encode state bar
        ax.eventplot(encode_state_lst, colors=['lime', 'red'],\
                     lineoffsets=[-baroffset*1.5 for i in range(len(encode_state_lst))], \
                     linewidth=barwidth, linelengths=barlength)
        # plot subcompartment bar
        ax.eventplot(subcompartment_state_lst, colors=['magenta', 'cyan'],\
                     lineoffsets=[-baroffset*0.5 for i in range(len(subcompartment_state_lst))], \
                     linelengths=barlength)
        # plot contactmap or observe/expected map
        if mode_flag == 1 or mode_flag == 2:
            img = ax.imshow(map_temp, interpolation='nearest', cmap = plt.cm.Reds,\
                            vmin=color_range[0]*map_temp.min(), vmax=color_range[1]*map_temp.max())
        elif mode_flag == 3:
            color_norm = MidPointNorm(midpoint=1.0)
            img = ax.imshow(map_temp, interpolation='nearest', cmap = plt.cm.bwr,vmax=3.4)
            cbar = fig.colorbar(img)
        elif mode_flag == 4:
            img = ax.imshow(map_temp, interpolation='nearest', cmap = plt.cm.bwr, vmin=-0.05, vmax=0.05)
            #cbar = fig.colorbar(img)
            #cbar.ax.set_yticklabels(['','',''])
        ax.text(1,1,self.chrom, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
        ax.text(0,1,'25kb resolution', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        plt.xticks([0, n/norm], ['{:.2f}Mb'.format(self.start/1000000.0), '{:.2f}Mb'.format(self.end/1000000.0)])
        plt.yticks([-2.0*baroffset, n/norm], ['', ''])
        plt.text(-0.04,0.97,r'$\mathrm{I}$', transform=ax.transAxes)
        plt.text(-0.04,0.92,r'$\mathrm{II}$', transform=ax.transAxes)
        if '.png' in fp:
            plt.savefig(fp, dpi=300)
        else:
            plt.savefig(fp+'.png', dpi=300)

    def plot_ps(self, foutname=None, guide=False):
        if self.contact_probability is None:
            raise ValueError('No contact probability profile available. Please \
                             first run method get_contact_prob\n')

        x_fit = self.contact_probability[:, 0]*self.bin_size/2.0
        ymax = self.contact_probability[:, 1].max()
        y_fit1 = np.power(x_fit, -1.0)
        y_fit2 = np.power(x_fit, -0.75)
        y_fit3 = np.power(x_fit, -1.2)

        y_fit1 = (ymax/y_fit1.max())*y_fit1
        y_fit2 = (ymax/y_fit2.max())*y_fit2
        y_fit3 = (ymax/y_fit3.max())*y_fit3

        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(self.contact_probability[:, 0]*self.bin_size/2.0, \
                self.contact_probability[:, 1])
        if guide:
            ax.plot(x_fit, y_fit1, label=r'$\sim s^{-1}$')
            ax.plot(x_fit, y_fit2, label=r'$\sim s^{-0.75}$')
            ax.plot(x_fit, y_fit3, label=r'$\sim s^{-1.2}$')
            plt.legend(loc='upper right')
        ax.set_xlabel('genome distance(bps)')
        ax.set_ylabel('relative contact frequency')
        if foutname is None:
            plt.show()
        else:
            if '.png' in foutname:
                plt.savefig(foutname)
            else:
                plt.savefig(foutname+'.png')

    def write_ps(self, fp):
        if self.contact_probability is None:
            raise ValueError('No contact probability profile available. Please \
                             first run method get_contact_prob\n')

        with open(fp, 'w') as f:
            f.write('File created at {}. Author: Guang Shi\n'.format(datetime.date.today()))
            f.write('separation contact_probability\n')
            if '.npy' in fp:
                np.save(f, self.contact_probability)
            else:
                np.savetxt(f,  self.contact_probability)


# -----------------------------------------------------------------------------
# define several functions used for plotting contact map

def state2number(state, data_source, mode = 'two state'):
    if data_source == 'Encode_Chrom_State':
        if mode == 'two state':
            # state number 1..15. For details, see this site:
            # http://ucscbrowser.genap.ca/cgi-bin/hgTrackUi?db=hg19&g=wgEncodeBroadHmm
            for i in range(1,16):
                if str(i) == state.split('_')[0]:
                    if i <= 11:
                        return 1 # 1 represents state 1. Open chromatin
                    else:
                        return 2 # 2 represents state 2. Close chromatin
        elif mode == 'encode original':
            return int(state.split('_')[0])
    # one can also feed the subcompartment data in 2014 Cell paper
    elif data_source == 'Cell_Subcompartment':
        if state == 'NA':
            return np.random.choice([1,2],1)[0]
        elif 'A' in state:
            return 1
        elif 'B' in state:
            return 2

def get_state(fp, chrom, start, end, bin_size, data_source, mode='two state'):
    # pandas dataframe
    data = pd.read_csv(fp, names = ['chrom', 'chromStart', 'chromEnd', 'name', 'score',\
                                    'strand', 'thickStart','thickEnd','itemRgb'], header=None,\
                       delim_whitespace=True)

    if data_source != 'Encode_Chrom_State' and data_source != 'Cell_Subcompartment':
        raise ValueError('Please specify correct data type:\n Options: Encode_Chrom_State or Cell_Subcompartment\n')

    chrom = chrom.lower()

    # convert the experimental subcompartment or chromatin state to simulation
    # monomer state (monomer atom type)
    data['name'] = data['name'].fillna('NA')
    data['name'] = data['name'].apply(state2number, args=(data_source, mode, ))

    simulation_data = np.linspace(start, end, int((end-start)/bin_size)+1, dtype=np.int)
    # 10000000 is just a number to make sure that selected experiment data starts from
    # loci which is before the actual start point of sequence specified in the function
    experiment_data = data[(data.chrom == chrom) & (data.chromStart >= start - 10000000) & \
                           (data.chromEnd <= end + 10000000)]
    experiment_start_end_state_array = experiment_data[['chromStart', 'chromEnd', 'name']].values

    # initialize the state dictionary
    if mode == 'two state':
        state_of_monomer_dic = {'state 1':[], 'state 2':[]} # two state
    elif mode == 'encode original':
        state_of_monomer_dic = {'state {}'.format(i):[] for i in range(1,16)}

    # loop the monomer to check the state of the monomer
    for i in range(len(simulation_data)-1):
        monomer_start = simulation_data[i]
        monomer_end = simulation_data[i+1]

        if mode == 'two state':
            state_number_array = np.zeros(2)
        elif mode == 'encode original':
            state_number_array = np.zeros(15)
        for state in experiment_start_end_state_array:
            # case where the subcompartment is inside a monomer range
            if monomer_start < state[0] and monomer_end > state[1]:
                state_number_array[state[2]-1] += state[1] - state[0]
            # case where the monomer is inside the subcompartment
            elif monomer_start >= state[0] and monomer_end <= state[1]:
                state_number_array[state[2]-1] += monomer_end - monomer_start
            # case where the left part of monomer is in the subcompartment
            elif monomer_start >= state[0] and monomer_start <= state[1]:
                state_number_array[state[2]-1] += state[1] - monomer_start
            # case where the right part of monomer is in the subcompartment
            elif monomer_end >= state[0] and monomer_end <= state[1]:
                state_number_array[state[2]-1] += monomer_end - state[0]

        # determine the final state of one monomer
        # each monomer can only be assigned to one state
        # criterion: whichever is the state has more base paris in one monomer
        state_of_monomer = np.argmax(state_number_array) + 1

        # create a dictionary where
        # key are state name
        # value are index of monomer which has that state

        state_of_monomer_dic['state {}'.format(state_of_monomer)].append(i+1)

    return state_of_monomer_dic

class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = np.getmask(result)
                result = np.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = np.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint
