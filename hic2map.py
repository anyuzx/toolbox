# ------------------------------------#
# This script is to read Hi-C data to #
# generate contact map. Rao 2014 Cell #
# paper                               #
# ------------------------------------#

import numpy as np
import argparse
import sys

# define function to normalize the raw contact matrix
# if the norm factor is nan or 0.0, the normalization
# is not performed
def norm_contact(loci_i, loci_j, raw_contact, resolution, norm_vector):
	index_i = int(loci_i/resolution)
	index_j = int(loci_j/resolution)
	if np.isnan(norm_vector[index_i]) or np.isnan(norm_vector[index_j])\
		or norm_vector[index_i] == 0.0 or norm_vector[index_j] == 0.0:
		norm_factor = 1.0
	else:
		norm_factor = norm_vector[index_i] * norm_vector[index_j]
	return raw_contact/norm_factor

# DEFINE ARGUMENTS
# USAGE:
#	python hic2map.py input-file output-file -s 10000 300000 -r 25000 -n norm-file
#	--select(-s): the range of the selected sequence
#	--resolution(-r): resolution of the data
#	--normalize(-n): specify the normalization vector file
parser = argparse.ArgumentParser(description='Generate contact matrices from Hi-C data (Rao 2014 Cell Paper)')
parser.add_argument('input_file', help='experimental Hi-C data')
parser.add_argument('output_file', help='output file name')
parser.add_argument('-s', '--select', help='select the range of chromosome', nargs=2,\
					dest='select_range', type = float)
parser.add_argument('-r', '--resolution', help='assign the resoultion of the data',\
					dest='resoultion', type=float)
parser.add_argument('-n',' --normalize', help='specify the normalization vector file',\
					dest='norm_vector_file', type=str)
args = parser.parse_args()

# read the experimental data (Rao 2014 Cell)
# format of the data:
#	loci_1 loci_2 contacts
hic = np.loadtxt(args.input_file, dtype = np.float32)

# infer the resolution from the data or user can manually assign the resolution
if args.resolution is None:
	resolution = float(np.unique(np.sort(hic[:,1] - hic[:,0]))[1])
else:
	resolution = args.resolution

# get the method of normalization
if args.norm_vector_file is not None:
	norm_vector = np.loadtxt(args.norm_vector_file, dtype=np.float32)

# 
if args.select_range is None:
	start = np.min(hic[:,:-1])
	end = np.max(hic[:,:-1])
	n = int((end - start)/resolution) + 1
else:
	start = args.select_range[0]
	end = args.select_range[1]
	n = int((end - start)/resolution) + 1

# initial contact map and normalize the contact map
contact_map = np.zeros((n,n))
if args.norm_vector_file is not None:
	contact_map_norm = np.zeros((n,n))

# get the count of each contact into the contact map
for row in hic:
	if row[1] >= start and row[1] <= end and row[0] >= start and row[0] <= end:
		index_i = int((row[0] - start)/resolution)  # location of first loci
		index_j = int((row[1] - start)/resolution)  # location of second loci
		contact_map[index_i, index_j] += row[2] # number of contacts
		if args.norm_vector_file is not None:
			contact_map_norm[index_i, index_j] += norm_contact(row[0], row[1], row[2], resolution, norm_vector)

# write out the contact map
np.savetxt(args.output_file + '.cmap', contact_map, header='file:{}.start:{}.end:{}'.format(args.input_file, start, end))

# write out the normalized contact map
if args.norm_vector_file is not None:
	np.savetxt(args.output_file + '.normcmap', contact_map_norm, header='file:{}.start:{}.end:{}.norm:{}'.format(args.input_file, start, end, args.norm_vector_file))
