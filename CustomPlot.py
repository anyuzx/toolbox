import numpy as np
import matplotlib.pyplot as plt

def plot_contact_triangle(
    contact_matrix: np.ndarray,
    genomic_start: float,
    genomic_end: float,
    outpath: str | None = None, 
    region: tuple[int, int] = None,
    cmap: str = 'Reds',
    show_colorbar: bool = True,
    pseudocount: float = 1e-6,
    figsize: float = 6,
    track_data: np.ndarray | None = None,
    track_height: float = 0.2
) -> None:
    """
    Plot a Hi-C/Micro-C contact map as a half-matrix "triangle" with:
      • the diagonal (i = j) rendered as a horizontal line at the BOTTOM,
      • all off-diagonals (j > i) forming a right triangle that points UPWARD,
      • the base of that triangle spanning from genomic_start to genomic_end,
      • values on a log10 scale,
      • no rectangular border, and
      • equal aspect ratio so the apex angle is 90°.
      • Optionally display a 1D track underneath the contact map

    Parameters
    ----------
    contact_matrix : np.ndarray, shape (N, N)
        The full N×N contact-matrix (must be square).

    genomic_start : float
        Genomic coordinate corresponding to bin index 0 (e.g., in bp or kb).

    genomic_end : float
        Genomic coordinate corresponding to the end of bin index N. Each bin thus spans
        (genomic_end – genomic_start)/N in genomic length.

    region : tuple[int, int], optional (default=None)
        If specified as (i_min, i_max), the function will first slice:
            submatrix = contact_matrix[i_min:i_max, i_min:i_max]
        and plot that submatrix. Indices are 0-based, and i_max is exclusive. If None, use entire matrix.

    cmap : str, optional (default='Reds')
        Any valid Matplotlib colormap name (e.g. 'Reds', 'viridis', 'magma').

    show_colorbar : bool, optional (default=True)
        Whether to draw a vertical colorbar to the right of the heatmap.

    pseudocount : float, optional (default=1e-6)
        A small constant added before taking log10 to avoid log(0). Adjust if your data require it.

    track_data : np.ndarray, optional (default=None)
        1D array containing the track data to be displayed underneath the contact map.
        Must have the same length as the number of bins in the contact matrix.

    track_height : float, optional (default=0.2)
        Height of the track plot relative to the contact map height.

    Returns
    -------
    None
        Displays the triangular heatmap in the current figure. Does not return any object.
    """

    # 1. Ensure the matrix is square
    if contact_matrix.ndim != 2 or contact_matrix.shape[0] != contact_matrix.shape[1]:
        raise ValueError("`contact_matrix` must be a square 2D numpy array.")

    # 2. If a sub-region is specified, crop to that bin range
    if region is not None:
        i_min, i_max = region
        N_full = contact_matrix.shape[0]
        if not (0 <= i_min < i_max <= N_full):
            raise ValueError(f"Invalid region {region} for matrix of size {N_full}.")
        cm = contact_matrix[i_min:i_max, i_min:i_max]
    else:
        cm = contact_matrix.copy()

    # 3. Compute log10(contact + pseudocount)
    log_cm = np.log10(cm + pseudocount)

    # 4. Mask out the strict lower-triangle (we only plot j >= i)
    mask = np.tril_indices_from(log_cm, k=-1)
    log_cm_masked = np.ma.array(log_cm, mask=False)
    log_cm_masked[mask] = np.ma.masked

    # 5. Build vertices so that (i, j) → (x_bin, y_bin) with:
    #       x_bin = (i + j)/2,    y_bin = (j - i)/2
    #    The diagonal (i=j) has y_bin=0; every j>i has y_bin>0.
    n = log_cm_masked.shape[0]
    I_vert, J_vert = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing='ij')

    X_vert_bin = (I_vert + J_vert) / 2.0
    Y_vert_bin = (J_vert - I_vert) / 2.0

    # 6. Convert "bin-units" → "genomic-coordinates" along X. Keep Y positive.
    if region is not None:
        N_full = contact_matrix.shape[0]
        bin_size = (genomic_end - genomic_start) / N_full
        base_start = genomic_start + i_min * bin_size
        X_genomic = base_start + X_vert_bin * bin_size
        x_left = base_start
        x_right = base_start + n * bin_size
    else:
        N_full = n
        bin_size = (genomic_end - genomic_start) / N_full
        X_genomic = genomic_start + X_vert_bin * bin_size
        x_left = genomic_start
        x_right = genomic_end

    # Because Y_vert_bin ≥ 0 for j ≥ i, we multiply by bin_size to get genomic-scale Y.
    Y_genomic = Y_vert_bin * bin_size
    y_bottom = 0.0
    y_top = (n / 2.0) * bin_size  # maximum Y when j-i = (n-1)

    # 7. Plot via pcolormesh. Now the diagonal (y=0) is at the BOTTOM, off-diagonals go up.
    if track_data is not None:
        # Create figure with two subplots
        fig = plt.figure(figsize=(figsize, figsize * (1 + track_height)))
        gs = plt.GridSpec(2, 1, height_ratios=[1, track_height], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_track = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    pcm = ax.pcolormesh(
        X_genomic,
        Y_genomic,
        log_cm_masked,
        cmap=cmap,
        shading='flat',
        edgecolors='none'
    )

    # Add borders for the triangle
    # Bottom edge (diagonal)
    ax.plot([x_left, x_right], [0, 0], color='black', linewidth=1.5, zorder=3)
    # Left edge (45° angle)
    ax.plot([x_left, x_left + (x_right - x_left)/2], [0, y_top], color='black', linewidth=1, zorder=3)
    # Right edge (135° angle)
    ax.plot([x_right, x_left + (x_right - x_left)/2], [0, y_top], color='black', linewidth=1, zorder=3)

    # 8. Set axis limits so that y=0 (diagonal) is at the bottom and y_top is at the top.
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    # 9. Force equal aspect: 1 unit in X = 1 unit in Y. The triangle's apex angle will be 90°.
    ax.set_aspect('equal')

    # 10. Label the x-axis; hide the y-axis completely
    ax.set_xlabel("Genomic coordinate")
    ax.set_ylabel("")    # we typically don't show y-label for a Hi-C triangle
    ax.set_yticks([])    # remove all y-tick marks

    # Show only the start and end on the x-axis
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels([f"{x_left:.2f}", f"{x_right:.2f}"])

    # 11. Remove all spines so there's no rectangular border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 12. Optionally add a colorbar on the right
    if show_colorbar:
        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label("log$_{10}$(contact + pseudocount)")

    # 13. Plot track data if provided
    if track_data is not None:
        if len(track_data) != n:
            raise ValueError(f"Track data length ({len(track_data)}) must match the number of bins ({n})")
        
        # Handle NaN values in track data
        track_data = np.nan_to_num(track_data, nan=0.0)
        
        # Create x coordinates for track data
        x_track = np.linspace(x_left, x_right, n)
        
        # Plot track data
        ax_track.fill_between(x_track, 0, track_data, alpha=0.7)
        
        # Set track plot properties
        ax_track.set_xlim(x_left, x_right)
        ax_track.set_ylim(0, np.max(track_data) * 1.1)  # Add 10% padding
        ax_track.set_xticks([x_left, x_right])
        ax_track.set_xticklabels([f"{x_left:.2f}", f"{x_right:.2f}"])
        ax_track.set_yticks([])  # Hide y-ticks
        ax_track.set_ylabel("Track")
        
        # Remove spines except bottom
        for spine in ax_track.spines.values():
            spine.set_visible(False)
        ax_track.spines['bottom'].set_visible(True)
        
        # Add bottom border line
        ax_track.plot([x_left, x_right], [0, 0], color='black', linewidth=1)

    plt.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        
    plt.show()
