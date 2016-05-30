## README

This is a collection of several scripts and modules I use in my daily research.

## List

### `contactmap`

To start to use this module

```
import contactmap
cmap = contactmap.contactmap(chrom='Chr5',start=145870001,end=157870001,bin_size=1200)
```

In the above code, specify the chromosome using argument `chrom`. The range and the bin resolution is specified by `start`, `end` and `bin_size`.

To read contact map, do the following

```
cmap.read('/Users/gs27722/Desktop/Chr5_145870001_157870001_SC/model_1/temp_dependence/analysis/CONTACTMAP/Chr5_T0.60_SC_traj_cmap_avg.npy')
```

`read` method can read contactmap file in `.npy` formart or normal ASCII format. After successfully load the contactmap, the contact map matrix can be retrieved by class attribute `cmap.map`. To normalize the contact map, use the method `self.normalize()`

```
cmap.normalize(10)
```

`10` is used for normalize the contact map by a factor of 10. The way of normalization is to use block of loci of size 10. The number of contacts between normalized blocks is calculated by summing all the contacts of all loci between blocks.

To calculate the contact probability profile, use the method

```
self.get_contact_prob(bin_method='linear')
```

the calculated probability profile is retrieved by `cmap.contact_probability`. 
