import numpy as np
import scipy.optimize

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

