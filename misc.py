import numpy as np
import scipy.optimize
from tqdm import tqdm
import pandas as pd

def linear_fit(x, a, b):
    """
    Simple linear model for curve fitting in log–log space.

    Parameters
    ----------
    x : array‐like or float
        Independent variable (e.g., log(time)).
    a : float
        Slope of the line.
    b : float
        Intercept of the line.

    Returns
    -------
    array‐like or float
        The value of a * x + b evaluated at each element of x.
    """
    return a * x + b

def msd2params(
    msds: list[np.ndarray],
    t_range: tuple[float, float]
) -> np.ndarray:
    """
    Fit a power‐law MSD(t) = D * t^alpha over a specified time window,
    and return the diffusion coefficient D and exponent alpha for each curve.

    Parameters
    ----------
    msds : list of np.ndarray, each of shape (M, 2)
        A list of MSD curves. Each array should have two columns:
            - column 0: time values (t_i)
            - column 1: MSD values at those times (msd_i)

    t_range : tuple (t_min, t_max)
        The inclusive time interval over which to perform the fit.
        Only points with t_min <= t <= t_max will be used.

    Returns
    -------
    np.ndarray of shape (len(msds), 2)
        For each input MSD curve, returns [D, alpha], where MSD ≈ D * t^alpha
        on the interval [t_min, t_max]. If a curve has fewer than two points
        in that interval, its row will be [np.nan, np.nan].
    """
    t_min, t_max = t_range
    if t_min > t_max:
        raise ValueError(f"Invalid t_range: {t_min} > {t_max}")

    results = []  # to hold [D, alpha] for each curve

    for curve in msds:
        times = curve[:, 0]
        msd_vals = curve[:, 1]

        # Select points within [t_min, t_max]
        mask = (times >= t_min) & (times <= t_max)
        t_sel = times[mask]
        msd_sel = msd_vals[mask]

        # Need at least two points to fit log–log
        if t_sel.size < 2:
            results.append([np.nan, np.nan])
            continue

        # Perform linear fit on log–log data:
        #   log(msd) ≈ alpha * log(t) + log(D)
        log_t = np.log(t_sel)
        log_msd = np.log(msd_sel)

        # functions.linear_fit(x, a, b) should return a * x + b
        popt, _ = scipy.optimize.curve_fit(
            linear_fit,
            log_t,
            log_msd
        )
        alpha, logD = popt
        D = np.exp(logD)

        results.append([D, alpha])

    return np.array(results)

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
        
def exponent2dmap(n, nu):
    dmap = np.zeros((n, n))
    for s in range(n):
        dmap[kth_diag_indices(dmap, s)] = s**nu
    dmap = dmap + dmap.T
    return dmap

def zscore_matrix(matrix):
    """
    Given an N×N array of relaxation times, compute the z‐score for each element
    relative to the distribution of all elements.
    
    Parameters
    ----------
    matrix : array_like, shape (N, N)
        Matrix of relaxation times.
    
    Returns
    -------
    z_mat : ndarray, shape (N, N)
        Z‐score matrix, z_ij = (tau_ij – μ) / σ, where μ and σ are the
        mean and standard deviation of all entries in tau_matrix.
    """
    # Convert to array and flatten
    arr = np.asarray(matrix)
    n = len(arr)

    z_matrix = np.full_like(arr, 0.0)
    
    for s in np.arange(1, n):
        mu = np.nanmean(np.diag(arr, k=s))
        sigma = np.nanstd(np.diag(arr, k=s))
        if sigma == 0.0:
            z = 0.0
        else:
            z = (np.diag(arr, k=s) - mu) / sigma
        z_matrix[kth_diag_indices(arr, k=s)] = z
    z_matrix = z_matrix + z_matrix.T
    return z_matrix

def correlation_matrix(
    data: np.ndarray
) -> np.ndarray:
    """
    Compute the Pearson correlation matrix between rows of a 2D array.

    Parameters
    ----------
    data : np.ndarray, shape (n_rows, n_cols)
        A 2D NumPy array where each row is a 1D vector. Correlations are
        computed pairwise between rows.

    Returns
    -------
    corr_mat : np.ndarray, shape (n_rows, n_rows)
        A symmetric matrix where corr_mat[i, j] is the Pearson correlation
        coefficient between data[i, :] and data[j, :]. The diagonal entries
        are 1.0.

    Raises
    ------
    ValueError
        If `data` is not a 2D array or if any row has constant values (which
        would make the Pearson correlation undefined).

    Notes
    -----
    - This implementation uses NumPy’s `np.corrcoef`, which under the hood
      computes means and standard deviations for each row and then the
      covariance. It is both cleaner and generally faster than looping with
      scipy.stats.pearsonr.
    - If you need p-values or want to handle NaNs differently, you can revert
      to a manual loop using `scipy.stats.pearsonr`. See the alternative
      implementation below.
    """
    # 1) Input validation
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got an array with shape {data.shape}.")

    n_rows, n_cols = data.shape

    # 2) Check for any row that is constant (zero variance) → Pearson undefined
    #    We compute the standard deviation of each row.
    row_stds = np.std(data, axis=1)
    if np.any(row_stds == 0):
        idx = np.where(row_stds == 0)[0][0]
        raise ValueError(f"Row {idx} has zero variance; Pearson correlation is undefined.")

    # 3) Use NumPy’s corrcoef which returns the correlation matrix for rows when
    #    you pass `rowvar=True` (the default). By default, np.corrcoef interprets
    #    each row as a variable and each column as an observation.
    corr_mat = np.corrcoef(data)

    # 4) Force exact ones on the diagonal in case numerical precision made them slightly off.
    np.fill_diagonal(corr_mat, 1.0)

    return corr_mat


def compute_monomer_average_msd_log(traj, num_lags=50):
    """
    Compute the monomer-averaged MSD at log-spaced lag times from trajectory data.

    Parameters
    ----------
    traj : np.ndarray, shape (T, N, 3)
        Trajectory array: T time points, N monomers, 3 spatial coordinates.
    num_lags : int
        Number of log-spaced lag times to compute between 1 and T-1. Default is 50.
    
    Returns
    -------
    msd_log_array : np.ndarray, shape (M, 2)
        Column 0: selected lag times dt (log-spaced, integer, unique, sorted)
        Column 1: monomer-averaged MSD(dt) for those dt
    """
    T, N, _ = traj.shape
    max_lag = T - 1
    
    # Generate log-spaced lags between 1 and max_lag
    raw_lags = np.logspace(np.log10(1), np.log10(max_lag), num_lags)
    dt_candidates = np.unique(np.round(raw_lags).astype(int))
    dt_candidates = dt_candidates[dt_candidates >= 1]
    dt_candidates = dt_candidates[dt_candidates <= max_lag]
    
    M = len(dt_candidates)
    msd_log = np.zeros(M)
    
    # Compute MSD only at the selected log-spaced lags
    for idx, dt in enumerate(tqdm(dt_candidates)):
        disp = traj[dt:] - traj[:-dt]  # shape (T-dt, N, 3)
        sq_disp = np.sum(disp**2, axis=2)  # shape (T-dt, N)
        msd_log[idx] = np.mean(sq_disp)
    
    msd_log_array = np.column_stack((dt_candidates, msd_log))
    return msd_log_array

def coarse_grain_matrix(mat, block_size, method='mean'):
    """
    Coarse-grain a square matrix by reducing non-overlapping blocks to single values.

    Parameters
    ----------
    mat : np.ndarray
        Input square matrix (2D array).
    block_size : int
        Size of the square block for coarse-graining.
    method : str
        Reduction method: 'mean', 'sum', 'median', 'min', 'max'.

    Returns
    -------
    np.ndarray
        Coarse-grained matrix.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = mat.shape[0]
    new_n = n // block_size
    mat_cg = mat[:new_n*block_size, :new_n*block_size].reshape(new_n, block_size, new_n, block_size)
    mat_cg = mat_cg.transpose(0,2,1,3).reshape(new_n, new_n, block_size, block_size)

    if method == 'mean':
        return mat_cg.mean(axis=(2,3))
    elif method == 'sum':
        return mat_cg.sum(axis=(2,3))
    elif method == 'median':
        return np.median(mat_cg, axis=(2,3))
    elif method == 'min':
        return mat_cg.min(axis=(2,3))
    elif method == 'max':
        return mat_cg.max(axis=(2,3))
    else:
        raise ValueError(f"Unknown method: {method}")

def build_contact_map_from_straw(records, resolution, start_pos=None, end_pos=None):
    """
    Build contact map from hicstraw.straw() records.
    
    Parameters:
    -----------
    records : list
        List of records from hicstraw.straw(), each with binX, binY, counts
    resolution : int
        Resolution in bp (e.g., 25000 for 25kb)
    start_pos : int, optional
        Start position in bp. If None, will use min(binX, binY)
    end_pos : int, optional
        End position in bp. If None, will use max(binX, binY)
    
    Returns:
    --------
    contact_map : numpy.ndarray
        Contact matrix
    bin_positions : numpy.ndarray
        Array of bin positions in bp
    """
    if len(records) == 0:
        raise ValueError("No records found")
    
    # Find the range of bins if not provided
    if start_pos is None:
        start_pos = min(min(r.binX for r in records), min(r.binY for r in records))
    if end_pos is None:
        end_pos = max(max(r.binX for r in records), max(r.binY for r in records))
    
    # Calculate number of bins
    n_bins = (end_pos - start_pos) // resolution + 1
    
    print(f"Building {n_bins} x {n_bins} contact map...")
    print(f"Genomic range: {start_pos:,} - {end_pos:,} bp")
    print(f"Resolution: {resolution:,} bp")
    print(f"Number of records: {len(records):,}")
    
    # Initialize contact map
    contact_map = np.zeros((n_bins, n_bins))
    
    # Fill contact map from records
    for record in tqdm(records, desc="Processing contacts"):
        # Convert bp positions to bin indices
        i = (record.binX - start_pos) // resolution
        j = (record.binY - start_pos) // resolution
        
        # Check bounds
        if 0 <= i < n_bins and 0 <= j < n_bins:
            contact_map[i, j] = record.counts
            # Make symmetric (Hi-C matrices are symmetric)
            if i != j:
                contact_map[j, i] = record.counts
    
    # Create array of bin positions
    bin_positions = np.arange(start_pos, end_pos + resolution, resolution)[:n_bins]
    
    return contact_map, bin_positions

def generate_contact_matrix(bed_file, contact_file, chrom, region_start, region_end):
    """
    Generate a Hi-C contact matrix for a given chromosome and genomic region.

    Parameters:
        bed_file (str): Path to the BED file. Can have either:
                        - 3 columns: "chrom", "start", "end" (row number used as index)
                        - 4 columns: "chrom", "start", "end", "index" (index column used)
        contact_file (str): Path to the contact file with 3 columns: i, j, contact.
                            i and j correspond to bin indices from the BED file.
        chrom (str): Chromosome of interest (e.g., "chr1").
        region_start (int): Start coordinate of the region.
        region_end (int): End coordinate of the region.

    Returns:
        matrix (np.ndarray): 2D NumPy array representing the contact matrix for the specified region.
        region_bins (pd.DataFrame): DataFrame of the BED file entries (bins) that fall into the region.
    """
    # Read the BED file and detect number of columns
    # First read without column names to check structure
    bed_df = pd.read_csv(bed_file, sep='\t', header=None)
    n_cols = bed_df.shape[1]
    
    if n_cols == 3:
        # 3-column BED file: chrom, start, end
        bed_df.columns = ["chrom", "start", "end"]
        # Add row index column (0-indexed) - this will be used to map to contact file indices.
        # Each row in the BED file is unique, so row number serves as the bin index.
        bed_df["index"] = bed_df.index
    elif n_cols == 4:
        # 4-column BED file: chrom, start, end, index
        bed_df.columns = ["chrom", "start", "end", "index"]
    else:
        raise ValueError(f"BED file must have 3 or 4 columns, but found {n_cols} columns.")
    
    
    # Filter bins for the specified chromosome that overlap the region.
    # Here we consider bins whose 'end' is >= region_start and 'start' is <= region_end.
    region_bins = bed_df[
        (bed_df["chrom"] == chrom) &
        (bed_df["start"] >= region_start) &
        (bed_df["end"] <= region_end)
    ].copy()
    
    if len(region_bins) == 0:
        raise ValueError("No bins found in the specified region.")
    
    # Get the list of bin indices (row numbers) in the region and sort them.
    bin_indices = sorted(region_bins["index"].unique())
    
    # Create a mapping from the original bin index (row number) to the row/column index in the matrix.
    index_map = {bin_idx: i for i, bin_idx in enumerate(bin_indices)}
    
    # Initialize the matrix with zeros.
    n = len(bin_indices)
    matrix = np.zeros((n, n))
    
    # Read the contact file.
    contact_df = pd.read_csv(contact_file, sep='\t', header=None, names=["i", "j", "contact"])
    
    # Filter the contacts to only those where both i and j are in the selected bin indices.
    region_contacts = contact_df[
        contact_df["i"].isin(bin_indices) & contact_df["j"].isin(bin_indices)
    ]
    
    # Fill the matrix with contact values.
    for _, row in region_contacts.iterrows():
        i_bin = row["i"]
        j_bin = row["j"]
        contact_val = row["contact"]
        # Map the original bin indices (row numbers) to matrix indices.
        i_mat = index_map[i_bin]
        j_mat = index_map[j_bin]
        matrix[i_mat, j_mat] = contact_val
        # Since Hi-C matrices are symmetric, fill in the mirror element.
        matrix[j_mat, i_mat] = contact_val

    return matrix, region_bins