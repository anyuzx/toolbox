import numpy as np
import pandas as pd
import sys
import argparse
import glob
import LammpsData as ld

def main():
    parser = argparse.ArgumentParser(description='Convert homopolymer configurations \
                                     to subcompartment model.')
    parser.add_argument('-in', '--input', help='data files need to be converted.', \
                        dest='input')
    parser.add_argument('-chr', '--chromosome', help='choose a chromosome', \
                        dest='chr', type=int)
    parser.add_argument('-s', '--select', help='select a range',\
                        dest='select', nargs=2)
    parser.add_argument('-bin', '--binsize', help='specify bin size',\
                        dest='bin_size', type=int)
    args = parser.parse_args()

    files = glob.glob(args.input)
    chromosome = 'chr'+str(args.chr)
    start = int(args.select[0])
    end = int(args.select[1])
    bin_size = args.bin_size

    trash, monomer_state = get_state('wgEncodeBroadHmmGm12878HMM.bed', \
    	                                       chrom=chromosome, start=start,end=\
    	                                       end, bin_size=bin_size, \
    	                                       data_source='Encode_Chrom_State')


    # convert data

    for fp in files:
        lammps_data = ld.LammpsData()
        lammps_data.read(fp)
        for atom in lammps_data.sections['Atoms']:
            atom[2] = str(monomer_state[int(atom[0])-1])
        lammps_data.headers['atom types'] = 2
        lammps_data.sections['Masses'].append(['2', '1'])
        lammps_data.write('new_'+fp)


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# define function to get state from expereiment data
# define function to convert the subcompartment or chromatin state to number
# if providing the Encode Chromatin state data, state number <=1 --> open chromatin
#                                                                    state 1
# if providing the subcompartment state data, A subcompartment --> open chromatin
#                                                                  state 1
def convert_state_2_number(state, data_source):
    if data_source == 'Encode_Chrom_State':
        for i in range(1,16):
            if str(i) == state.split('_')[0]:
                if i <= 11:
                    return 1  # 1 represents state 1. open chromatin
                else:
                    return 2  # 2 represents state 2. close chromatin
    elif data_source == 'Cell_Subcompartment':
        if state == 'NA':
            return np.random.choice([1,2],1)[0]
        elif 'A' in state:
            return 1
        elif 'B' in state:
            return 2


# This function gives epigenetic state from the experimental data
# Argument:
#           filename: directory for experimental data file.
#           chrom: name of chromosome
#           start: starting point of the sequence
#           end: ending point of the sequence
#           bin_size: the number of base pairs each bead represent
#           data_source: Now two kinds of files are accepted:
#                        1. Cell Rao, etc paper
#                        2. Encode Chromatin state
def get_state(filename, chrom, start, end, bin_size, data_source):
    # pandas dataframe for data
    data = pd.read_csv(filename, names = ['chrom','chromStart','chromEnd','name','score',\
                       'strand','thickStart','thickEnd','itemRgb'], header = None, \
                        delim_whitespace=True)

    if data_source != 'Encode_Chrom_State' and data_source != 'Cell_Subcompartment':
        sys.stdout.write("Please specify correct data source type:\n 1. 'Encode_Chrom_State'\n 2. 'Cell_Subcompartment'\n")
        raise ValueError

    # convert the experimental subcompartment or chromatin state to simulation \
    # monomer state (monomer atom type)
    data['name'] = data['name'].fillna('NA')
    data['name'] = data['name'].apply(convert_state_2_number, args=(data_source,))

    simulation_data = np.linspace(start, end, int((end-start)/bin_size)+1, dtype=np.int)
    # 10000000 is just a number to make sure that selected experiment data starts from
    # before the actual start point of sequence specified in the function
    experiment_data = data[(data.chrom == chrom) & (data.chromStart >= start - 10000000) & \
                           (data.chromEnd <= end + 10000000)]
    experiment_start_end_state_array = experiment_data[['chromStart', 'chromEnd', \
                                                        'name']].values

    state_of_monomer_dic = {'state 1':[], 'state 2':[]} # initialize the dictionary
    monomer_state = []
    # loop the monomer to check the state of the monomer
    for i in range(len(simulation_data)-1):
        monomer_start = simulation_data[i]
        monomer_end = simulation_data[i+1]
        #np.searchsorted(data[data.chrom == chrom][[]].values, monomer_start)
        state_number_array = np.zeros(2)  # 2 means two distinct epeigenetic states.\
                                          # more states may be implemented in the future.
        for state in experiment_start_end_state_array:
            # case where the subcompartment is inside a monomer
            if monomer_start < state[0] and monomer_end > state[1]:
                state_number_array[state[2]-1] += state[1] - state[0]
                break
            # case where the monomer is inside the subcompartment
            elif monomer_start >= state[0] and monomer_end <= state[1]:
                state_number_array[state[2]-1] += monomer_end - monomer_start
                break
            # case where the left part of monomer is in the subcompartment
            elif monomer_start >= state[0] and monomer_start <= state[1]:
                state_number_array[state[2]-1] += state[1] - monomer_start
                break
            # case where the right part of monomer is in the subcompartment
            elif monomer_end >= state[0] and monomer_end <= state[1]:
                state_number_array[state[2]-1] += monomer_end - state[0]
                break

        # get the state of the monomer
        # criterion: whichever the state has more base pairs in one monomer
        state_of_monomer = np.argmax(state_number_array) + 1
        monomer_state.append(state_of_monomer)

        # create the dictionary of state of monomers
        if state_of_monomer == 1:
            state_of_monomer_dic['state 1'].append(i+1)
        elif state_of_monomer == 2:
            state_of_monomer_dic['state 2'].append(i+1)

    monomer_state = np.asarray(monomer_state)
    return state_of_monomer_dic, monomer_state

if __name__ == '__main__':
    main()
