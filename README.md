# Toolbox

A collection of Python scripts and modules for computational biology and molecular dynamics simulation analysis used in my own research.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Modules](#modules)
  - [contactmap](#contactmap)
  - [LammpsLog](#lammpslog)
  - [LammpsData](#lammpsdata)
  - [ChipseqAnalysisTool](#chipseqanalysistool)
  - [CustomPlot](#customplot)
  - [misc](#misc)
- [Scripts](#scripts)
  - [Molecular Dynamics Tools](#molecular-dynamics-tools)
  - [Hi-C and Contact Map Tools](#hi-c-and-contact-map-tools)
  - [ChIP-seq and Genomics Tools](#chip-seq-and-genomics-tools)
- [Examples](#examples)
- [License](#license)

---

## Overview

This repository contains a collection of scripts and modules for:
- **Molecular Dynamics (MD) simulation analysis** using LAMMPS
- **Hi-C contact map analysis** and visualization
- **ChIP-seq data processing** and genomic analysis
- **Data format conversion** (LAMMPS dump, H5MD, XYZ formats)
- **Statistical analysis** and plotting utilities

---

## Installation

### Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Cython (for compilation)
- pyBigWig (for bigWig file support)
- h5py (for H5MD format support)

### Build from source

```bash
pip install -e .
```

This will build the Cython extensions and install the package in development mode.

---

## Modules

### `contactmap`

A module for analyzing chromosome contact maps from Hi-C data.

#### Features
- Read contact maps in `.npy` or ASCII format
- Normalize contact maps using block normalization
- Calculate contact probability profiles
- Support for genomic region specification

#### Example Usage

```python
import contactmap

# Initialize contact map for a specific genomic region
cmap = contactmap.contactmap(
    chrom='Chr5',
    start=145870001,
    end=157870001,
    bin_size=1200
)

# Read contact map data
cmap.read('my_cmap.npy')

# Access the contact map matrix
contact_matrix = cmap.map

# Normalize the contact map (block size = 10)
cmap.normalize(10)

# Calculate contact probability profile
cmap.get_contact_prob(bin_method='linear')

# Access the calculated probability profile
prob_profile = cmap.contact_probability
```

---

### `LammpsLog`

A module for reading and visualizing LAMMPS log files.

#### Features
- Parse LAMMPS log files
- Handle multiple log files simultaneously
- Plot thermodynamic data vs timestep
- Support for multi-section log files

#### Example Usage

```python
import LammpsLog

# Load multiple log files
logfile = LammpsLog.LammpsLog('test1.log', 'test2.log', 'test3.log')

# Plot data from specific log file and section
logfile.plot('test1.log', section_index=0, foutname='test1_log.png')
logfile.plot('test2.log', section_index=0, foutname='test2_log.png')
```

**Note:** Use `section_index` to specify which section to plot if the log file contains multiple consecutive records. Default is `0`.

---

### `LammpsData`

A module for reading, writing, and manipulating LAMMPS data files.

#### Features
- Read/write LAMMPS data files
- Access headers (atom count, bond count, box dimensions, etc.)
- Manipulate sections (Masses, Atoms, Bonds, Angles, etc.)
- Add angles between consecutive atoms
- Convert to Pandas DataFrames for analysis

#### Data Structure

- `LammpsData.headers`: Dictionary containing file metadata
- `LammpsData.sections`: Dictionary containing data sections

#### Example Usage

```python
import LammpsData

# Create LammpsData object
ld = LammpsData.LammpsData()

# Read LAMMPS data file
ld.read('lammps_data_file.dat')

# Add angles between consecutive atoms (i, i+1, i+2)
ld.AddAngle()

# Set custom description
ld.SetDescription('Modified data file with angles')

# Convert section to Pandas DataFrame for analysis
atoms_df = ld.GetDataFrame('Atoms')

# Write modified data file
ld.write('new_lammps_data_file.dat')
```

---

### `ChipseqAnalysisTool`

A module for processing ChIP-seq data including peak files and bigWig signal files.

#### Functions

##### `read_peak_bed_data(file, chrom, start, end, resolution)`

Read and bin ChIP-seq peak data from narrowPeak/BED files.

**Parameters:**
- `file` (str): Path to the narrowPeak file
- `chrom` (str): Chromosome name
- `start` (int): Start position of the region
- `end` (int): End position of the region
- `resolution` (int): Bin size for binning the data

**Returns:**
- `bin_midpoints`: Array of bin center positions
- `binned_values`: Array of summed signal values per bin

##### `read_bigwig_data(file, chrom, start, end, resolution)`

Read and bin signal data from bigWig files.

**Parameters:** Same as `read_peak_bed_data`

**Returns:** Same as `read_peak_bed_data`

#### Example Usage

```python
from ChipseqAnalysisTool import read_peak_bed_data, read_bigwig_data

# Read peak data
bin_midpoints, binned_values = read_peak_bed_data(
    file='peaks.narrowPeak',
    chrom='chr1',
    start=1000000,
    end=7000000,
    resolution=40000
)

# Read bigWig data
bin_midpoints, signal_values = read_bigwig_data(
    file='signal.bw',
    chrom='chr1',
    start=1000000,
    end=7000000,
    resolution=40000
)
```

---

### `CustomPlot`

Custom plotting utilities (exposed as importable module).

---

### `misc`

Miscellaneous utility functions (exposed as importable module).

---

## Scripts

### Molecular Dynamics Tools

#### `ergodic_metric.py`

Calculate the ergodic metric described in [Thirumalai et al., Phys. Rev. A 39, 3563 (1989)](http://journals.aps.org/pra/abstract/10.1103/PhysRevA.39.3563).

**Usage:**
```bash
python ergodic_metric.py data1.h5 data2.h5 output.txt 1000
```

**Arguments:**
- `data1.h5`: First H5MD trajectory file
- `data2.h5`: Second H5MD trajectory file
- `output.txt`: Output file for results
- `1000`: Start snapshot index (calculation begins from snapshot #1000)

---

#### `dump2h5md.py`

Convert LAMMPS custom dump files to [H5MD](http://nongnu.org/h5md/index.html) formatted HDF5 files.

**Usage:**
```bash
python dump2h5md.py lammps_dump_file output.h5 -s 100
```

**Arguments:**
- `-s, --stride`: Read every Nth frame (default: 1)

Use `--help` for additional options.

---

#### `h5md2dump.py`

Convert H5MD formatted files back to LAMMPS dump format.

---

#### `h5md2xyz.py`

Convert H5MD formatted files to XYZ format for visualization.

---

#### `xyz2h5md.py`

Convert XYZ trajectory files to H5MD format.

---

#### `mergetraj.py`

Merge multiple H5MD trajectory files into a single file.

**Usage:**
```bash
python mergetraj.py -in traj1.h5 traj2.h5 traj3.h5 -out merged.h5 -c -k position species -s 10
```

**Arguments:**
- `-in`: Input H5MD files to merge
- `-out`: Output filename
- `-c`: Enable continuity check (verify last frame of file N matches first frame of file N+1)
- `-k`: Keywords/datasets to store in output (e.g., `position`, `species`)
- `-s`: Stride (save every Nth frame)

Use `--help` for more information.

---

#### `recenter_dump.py`

Recenter molecular trajectories in LAMMPS dump files.

---

#### `TrajReorder.py`

Reorder atoms in trajectory files.

---

#### `smsd_fitting.py`

Perform spatial MSD (mean square displacement) fitting.

---

### Hi-C and Contact Map Tools

#### `hic2map.py`

Convert Hi-C data to contact map format.

---

#### `cmap2interp2dmap.py`

Interpolate 2D contact maps for missing data.

---

#### `matrix_interpolate_missing.py`

Interpolate missing values in contact matrices.

---

#### `dynamic_cmap.py`

Analyze dynamic contact maps over time.

---

#### `cmap_dispersion.py`

Calculate dispersion metrics for contact maps.

---

#### `CompContactLife.py` / `CompContactLife-xyz.py`

Compute contact lifetimes from simulation trajectories.

---

#### `cluster_growth.py`

Analyze cluster formation and growth dynamics.

---

### ChIP-seq and Genomics Tools

#### `grm_tools.py`

Tools for genomic regulatory map analysis.

---

#### `calculate_variance.py`

Calculate variance metrics for genomic data.

---

#### `get_state.py`

Extract chromatin state information.

---

#### `homo2sc.py`

Convert between homolog and single chromosome representations.

---

#### `generate_wlm.py`

Generate weighted locality maps.

---

## Examples

Example notebooks and scripts are available in the `examples/` directory:

- `test_contactmap.ipynb`: Jupyter notebook demonstrating contact map analysis
- `test_contactmap.py`: Python script version of contact map example
- Sample output images: `test_observed.png`, `test_pearson_coeff.png`

---

## Project Structure

```
toolbox/
├── README.md
├── setup.py
├── pyproject.toml
├── __init__.py
├── examples/
│   ├── test_contactmap.ipynb
│   └── test_contactmap.py
├── Modules:
│   ├── contactmap.py
│   ├── LammpsData.py
│   ├── LammpsLog.py
│   ├── ChipseqAnalysisTool.py
│   ├── CustomPlot.py
│   └── misc.py
├── Scripts:
│   ├── ergodic_metric.py
│   ├── dump2h5md.py
│   ├── mergetraj.py
│   └── ... (see Scripts section above)
└── Cython Extensions:
    ├── _matrixnorm.pyx
    └── _matrixnorm.c
```

---

## License

This is a personal research toolbox. Please contact the author for usage permissions.

---

## Contributing

This is a personal research repository. For questions or collaboration inquiries, please open an issue.

---

## Changelog

### Version 0.1
- Initial collection of research tools
- Support for LAMMPS data analysis
- Hi-C contact map analysis
- ChIP-seq data processing
- H5MD format conversion tools
