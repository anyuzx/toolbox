import pandas as pd
import numpy as np
import pyBigWig

def read_peak_bed_data(file, chrom, start, end, resolution):
    # Define column names for a narrowPeak file.
    columns = ['chrom', 'start', 'end', 'name', 'score', 'strand',
               'signalValue', 'pValue', 'qValue', 'peak']

    df = pd.read_csv(file, sep='\t', header=None, names=columns)

    region_start = np.int_(np.ceil(start / resolution) * resolution)
    region_end   = np.int_(np.floor(end / resolution) * resolution)
    bin_size = resolution

    # filter peaks overlapping region
    df_region = df[
        (df['chrom'] == chrom) &
        (df['start'] <=  region_end) &
        (df['end']   >=  region_start)
    ].copy()
    if df_region.empty:
        print("No peaks found in the specified region.")
        return None, None

    # summit position
    df_region['summit'] = df_region['start'] + df_region['peak']
    df_region = df_region[
        (df_region['summit'] >= region_start) &
        (df_region['summit'] <=  region_end)
    ]

    # bin index
    df_region['bin_index'] = (
        (df_region['summit'] - region_start) // bin_size
    ).astype(int)

    # total bins (149 for 6 Mb/40 kb)
    n_bins = int(np.ceil((region_end - region_start) / bin_size))
    
    # sum signalValue per bin
    bin_sums = df_region.groupby('bin_index')['signalValue'].sum()

    # fill array
    binned_values = np.zeros(n_bins)
    for b, v in bin_sums.items():
        if 0 <= b < n_bins:
            binned_values[b] = v

    # midpoints
    bin_midpoints = region_start + (np.arange(n_bins) + 0.5) * bin_size

    return bin_midpoints, binned_values

def read_bigwig_data(file, chrom, start, end, resolution):
    """
    Read bigwig file data for a specific genomic region and bin it at the specified resolution.
    
    Parameters:
    -----------
    file : str
        Path to the bigwig file
    chrom : str
        Chromosome name
    start : int
        Start position of the region
    end : int
        End position of the region
    resolution : int
        Bin size for binning the data
        
    Returns:
    --------
    tuple : (bin_midpoints, binned_values)
        bin_midpoints: array of bin center positions
        binned_values: array of binned signal values
    """
    try:
        # Open bigwig file
        bw = pyBigWig.open(file)
        
        # Define region boundaries
        region_start = np.int_(np.ceil(start / resolution) * resolution)
        region_end = np.int_(np.floor(end / resolution) * resolution)
        bin_size = resolution
        
        # Get values from bigwig for the specified region
        values = bw.values(chrom, region_start, region_end)
        
        if values is None or len(values) == 0:
            print("No data found in the specified region.")
            return None, None
        
        # Convert to numpy array and handle None values
        values = np.array([v if v is not None else 0.0 for v in values])
        
        # Create position array
        positions = np.arange(region_start, region_end)
        
        # Bin the data
        n_bins = int(np.ceil((region_end - region_start) / bin_size))
        binned_values = np.zeros(n_bins)
        
        for i, pos in enumerate(positions):
            if i < len(values):
                bin_index = (pos - region_start) // bin_size
                if 0 <= bin_index < n_bins:
                    binned_values[bin_index] += values[i]
        
        # Calculate bin midpoints
        bin_midpoints = region_start + (np.arange(n_bins) + 0.5) * bin_size
        
        # Close bigwig file
        bw.close()
        
        return bin_midpoints, binned_values
        
    except Exception as e:
        print(f"Error reading bigwig file: {e}")
        return None, None
