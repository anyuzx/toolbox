## README

This is a collection of several scripts and modules I use in my daily research.

## List

### `contactmap`

To start to use this module

```python
import contactmap
cmap = contactmap.contactmap(chrom='Chr5',start=145870001,end=157870001,bin_size=1200)
```

In the above code, specify the chromosome using argument `chrom`. The range and the bin resolution is specified by `start`, `end` and `bin_size`.

To read contact map, do the following

```python
cmap.read('/Users/gs27722/Desktop/Chr5_145870001_157870001_SC/model_1/temp_dependence/analysis/CONTACTMAP/Chr5_T0.60_SC_traj_cmap_avg.npy')
```

`read` method can read contactmap file in `.npy` formart or normal ASCII format. After successfully load the contactmap, the contact map matrix can be retrieved by class attribute `cmap.map`. To normalize the contact map, use the method `self.normalize()`

```python
cmap.normalize(10)
```

`10` is used for normalize the contact map by a factor of 10. The way of normalization is to use block of loci of size 10. The number of contacts between normalized blocks is calculated by summing all the contacts of all loci between blocks.

To calculate the contact probability profile, use the method

```python
self.get_contact_prob(bin_method='linear')
```

the calculated probability profile is retrieved by `cmap.contact_probability`.

### `ergodic_metric.py`

This script calculate the energy metric described in this [paper](http://journals.aps.org/pra/abstract/10.1103/PhysRevA.39.3563). The module takes four arguments: first data file, second data file, output file, start snapshot index.

```bash
python ergodic_metric.py data1.h5 data2.h5 output.txt 1000
```

data file need to be `hdf5` format. `1000` means that calculation starts from snapshot #1000. 

### `dump2hdf5.py`

This script convert normal Lammps custom file to [H5MD](http://nongnu.org/h5md/index.html) formatted file. It takes two arguments and several optional arguments

```bash
python dump2h5md.py lammps_custom_dump_file H5MD_formatted_file.h5 -s 100
```

Use argument `--help` for additional information.
