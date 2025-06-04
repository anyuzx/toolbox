import numpy as np
import scipy.optimize
from tqdm import tqdm

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