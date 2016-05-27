import numpy as np
import contactmap.contactmap as contactmap
import sys

temp = sys.argv[1]

cmap = contactmap(chrom='Chr5',start=145870001,end=157870001,bin_size=1200)
for i in range(1,31):
    cmap.read('Chr5_T{}_SC_{}_traj_cmap.npy'.format(temp, i))
    cmap.normalize(20)
    cmap_temp = cmap.normalized_map
    try:
        square_cmap += np.power(cmap_temp, 2.0)/30.0
        mean_cmap += cmap_temp/30.0
    except NameError:
        square_map = np.power(cmap_temp, 2.0)/30.0
        mean_cmap = cmap_temp/30.0

std = np.sqrt(square_cmap - np.power(nosquare_cmap, 2.0))
coeff_var = std/mean_cmap

np.save('coeff_var.npy', coeff_var)
