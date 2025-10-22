# Installation Guide

## Quick Install

### Standard Installation (with Cython extensions)

If you want full functionality including contact map normalization features:

```bash
pip install -e .
```

This will build the Cython extensions and install the package in development mode.

### Minimal Installation (without Cython extensions)

If you only need basic functionality and don't want to build Cython extensions:

```bash
# Just add the directory to your PYTHONPATH
export PYTHONPATH=/path/to/toolbox:$PYTHONPATH
```

Or in Python:
```python
import sys
sys.path.insert(0, '/path/to/toolbox')
import toolbox
```

**Note:** Without building the Cython extensions, the following `contactmap` methods will not be available:
- `normalize()`
- `get_OE()`
- `get_zscore()`
- `get_PearsonCoeff()`
- `get_subchain_contact()`

The script `generate_wlm.py` also requires the Cython extension to be built.

## Requirements

### Basic Requirements
- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- h5py (for H5MD format support)

### Optional Requirements
- **Cython** (for building `_matrixnorm` extension - enables contact map normalization)
- **pyBigWig** (for ChIP-seq bigWig file support)
- **scipy** (for some analysis scripts)

## Building Cython Extensions

### Method 1: Using pip (Recommended)

```bash
pip install -e .
```

### Method 2: Using setup.py directly

```bash
python setup.py build_ext --inplace
```

This will compile the `_matrixnorm.pyx` file and create a `.so` (Linux/Mac) or `.pyd` (Windows) file.

## Troubleshooting

### ImportError: cannot import name '_matrixnorm'

This means the Cython extension hasn't been built. You have two options:

1. **Build the extension** (recommended if you need normalization features):
   ```bash
   pip install -e .
   ```

2. **Use without the extension** (if you only need basic features):
   The package will work but skip methods that require `_matrixnorm`. You'll see a warning on import.

### Building on Different Systems

#### Linux
```bash
# Install build dependencies
sudo apt-get install python3-dev
pip install -e .
```

#### macOS
```bash
# Install Xcode Command Line Tools if not already installed
xcode-select --install
pip install -e .
```

#### Windows
You'll need Visual C++ Build Tools:
1. Download from: https://visualstudio.microsoft.com/downloads/
2. Install "Build Tools for Visual Studio"
3. Run: `pip install -e .`

## Verifying Installation

```python
import toolbox

# Check if Cython extension is available
from toolbox import contactmap
try:
    # This will work if extension is built
    cmap = contactmap.contactmap(chrom='chr1', start=1000000, end=2000000, bin_size=1000)
    print("✓ Cython extension available - full functionality")
except ImportError as e:
    print(f"✗ Cython extension not available: {e}")
    print("  Basic functionality only")
```

## Platform-Specific Compiled Extensions

This repository includes pre-compiled extensions for various platforms:
- `_matrixnorm.cpython-312-x86_64-linux-gnu.so` (Linux, Python 3.12)
- `_matrixnorm.cpython-312-darwin.so` (macOS, Python 3.12)
- `_matrixnorm.cpython-38-x86_64-linux-gnu.so` (Linux, Python 3.8)
- `_matrixnorm.cpython-38-darwin.so` (macOS, Python 3.8)

Python will automatically use the correct one for your system if available. If your system/Python version doesn't match, you'll need to build from source.

## Development Installation

For development with automatic reloading:

```bash
pip install -e . --no-build-isolation
```

After modifying `.pyx` files, rebuild:

```bash
python setup.py build_ext --inplace
```

