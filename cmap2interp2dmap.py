import sys
import numpy as np
import scipy
import scipy.interpolate
import argparse

# -------------------------------------------------------
# INTERPOLATION
def interpolate_miss(dmap):
    dmap_copy = np.copy(dmap)
    x = np.arange(0, dmap_copy.shape[1])
    y = np.arange(0, dmap_copy.shape[0])
    #mask invalid values
    dmap_copy = np.ma.masked_invalid(dmap_copy)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~dmap_copy.mask]
    y1 = yy[~dmap_copy.mask]
    newarr = dmap_copy[~dmap_copy.mask]

    GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='nearest')
    return GD1

# FUNCTION TO READ CMAP FILE
def read_cmap(fp):
    cmap = np.loadtxt(fp, skiprows=2)
    cmap = cmap + cmap.T - np.diag(np.diag(cmap))
    return cmap

# FUNCTION TO CONVERT CMAP TO DMAP
def cmap2dmap_exp(cmap_exp, rc, alpha, norm_max=1.0, mode='log'):
    # rc is the prefactor
    # norm_max is the maximum contact probability
    if mode == 'raw':
        log10_pmap = np.log10(cmap_exp) + np.log10(norm_max) - np.log10(np.max(cmap_exp))
    elif mode == 'log':
        log10_pmap = cmap_exp + np.log10(norm_max) - np.max(cmap_exp)

    return rc * 10 ** (-1.0/alpha * log10_pmap)
# -------------------------------------------------------

parser = argparse.ArgumentParser(description='convert raw contact map to distance map with given alpha value')
parser.add_argument('-c', type=str, help='specify the chromosome number')
parser.add_argument('-i', type=str, help='specify the contact map file')
parser.add_argument('-o', type=str, help='prefix for the output files')
parser.add_argument('--alpha', type=float, default=4.0, help='specify the value of alpha values')
parser.add_argument('--scale', type=float, default=0.065*1.8, help='specify value of prefactor scaling factor')
parser.add_argument('--ignore', type=float, default=0, help='specify the missing data threshold (fraction)')
args = parser.parse_args()

chr_cmap_raw = read_cmap(args.i)
natoms = chr_cmap_raw.shape[0]

pos = []
for i in range(len(chr_cmap_raw)):
    row = chr_cmap_raw[i, :]
    if args.c in ['chr1','chr2','chr3','chr4','chr5','chr6','chr7',\
    'chr8','chr9','chr10','chr11','chr12','chrX']:
        if np.sum(row != 0.0) > args.ignore * natoms:
            pos.append(i)
    else:
        if np.sum(row !=0.0) > args.ignore * natoms:
            pos.append(i)
pos = np.array(pos)

chr_cmap = chr_cmap_raw[pos[:, None], pos]
chr_cmap_interp = interpolate_miss(np.log10(chr_cmap))
chr_cmap_interp = np.array((chr_cmap_interp + chr_cmap_interp.T) / 2.)

dmap_chr = cmap2dmap_exp(chr_cmap_interp, args.scale, args.alpha)

np.savetxt('{}_interp.cmap'.format(args.o), chr_cmap_interp)
np.savetxt('{}.dmap'.format(args.o), dmap_chr)
