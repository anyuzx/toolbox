## DESCRIPTION

This repository is a collection of some scripts and modules I frequently use in my daily research.

-----

### MODULES

#### `contactmap`

To start to use this module

```python
import contactmap
cmap = contactmap.contactmap(chrom='Chr5',start=145870001,end=157870001,bin_size=1200)
```

In the above code, specify the chromosome using argument `chrom`. The range and the bin resolution is specified by `start`, `end` and `bin_size`.

To read contact map, do the following

```python
cmap.read('my_cmap.npy')
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


#### `LammpsLog`

This module can be used to read Lammps Log file and plot the attributes respect to timestep. 

```python
import LammpsLog

logfile = LammpsLog.LammpsLog('test1.log', 'test2.log', 'test3.log')
logfile.plot('test1.log', section_index=0, foutname='test1_log.png')
logfile.plot('test2.log', section_index=0, foutname='test2_log.png')
```

In the above code, given a series of Lammps log files. To plot the result, use method `plot` and specify which file you want to plot and output `.png` format image file. Sometimes log file may contains several parts of consecutive records. You can use `section_index` to specify which part you want to plot. Default value is 0.


#### `LammpsData`

This module can be used to read, write, manipulate Lammps Data file. 

### Example

```python
import H5MD_Analysis as HA

ld = HA.LammpsData()
ld.read('lammps_data_file.dat')
ld.AddAngle()
ld.write('new_lammps_data_file.dat')
```

method `read` can be used to read Lammps Data file. The information in one data file is decomposed to two dictionary `LammpsData.headers` and `LammpsData.sections`. `headers` dictionary contains information like number of atoms, bonds, angles, box dimension, et al. `sections` dictionary contains information like `Masses`, `Atoms`, `Bonds`, et al. The description of data file at the first line of file can be retrieved by keyword `description` in `LammpsData.headers`.

method `AddAngle` can be used to add angles between atoms in a atom index order. Like atom i, atom i+1, and atom i+2 form a angle. This method is used to change data file generated from simulation without angle potential to a data file with angles information, and then used to feed into simulation with angle potential.

method `SetDescription` can be used to overwrite the description line.

method `GetDataFrame` can be used to create a Pandas DataFrame object for every keyword in `LammpsData.sections`.


------

### SCRIPTS

#### `ergodic_metric.py`

This script calculate the energy metric described in this [paper](http://journals.aps.org/pra/abstract/10.1103/PhysRevA.39.3563). The module takes four arguments: first data file, second data file, output file, start snapshot index.

```bash
python ergodic_metric.py data1.h5 data2.h5 output.txt 1000
```

data file need to be `hdf5` format. `1000` means that calculation starts from snapshot #1000. 

#### `dump2hdf5.py`

This script convert normal Lammps custom file to [H5MD](http://nongnu.org/h5md/index.html) formatted file. It takes two arguments and several optional arguments

```bash
python dump2h5md.py lammps_custom_dump_file H5MD_formatted_file.h5 -s 100
```

Use argument `--help` for additional information.

#### `mergetraj.py`

This script merge multiple H5MD trajectory or other H5MD formatted files into one file. It takes 3 arguments and 2 optional arguments.

```bash
python mergetraj.py -in traj1.h5 traj2.h5 traj3.h5 -out traj.h5 -c -k position species -s 10
```

The argument `-in` specify the H5MD files used for merging. `-out` provides the name of output file. `-c` enable the continuity check of files, i.e. whether the last frame of the previous file is as the same as the first frame of the following file. `-k` provides the keyword you want to store in the output file. In the above example, we provide `position` and `species` to the argument `-k`, thus the merged file will contains these information. `-s` is the stride argument. You can use `--help` to see the information.
