import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch


_abs_s_colors = np.vstack(
    [
        np.array([[1.0, 1.0, 1.0, 1.0]]),
        np.array([
            [0.93333333, 0.96470588, 1.0, 1.0],
            [0.61960784, 0.77254902, 1.0, 1.0],
            [0.23921569, 0.49411765, 0.90980392, 1.0],
            [0.04313725, 0.23529412, 0.54901961, 1.0],
        ]),
    ]
)
ABS_S_CMAP = LinearSegmentedColormap.from_list("absS_white_blue", _abs_s_colors)


def _subsample(points, max_points=50000, seed=1234):
    points = np.asarray(points)
    if max_points is None:
        return points
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _point_bin_densities(points, bins=180, extent=(0.0, 1.0, 0.0, 1.0)):
    points = np.asarray(points)
    xmin, xmax, ymin, ymax = extent
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[xedges, yedges], density=True)

    ix = np.clip(np.searchsorted(xedges, points[:, 0], side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(yedges, points[:, 1], side="right") - 1, 0, bins - 1)
    dens = hist[ix, iy]
    return dens, hist


def _scatter_panel(ax, points, *, vmax, title, sample_label=None, label_color="white"):
    extent = (0.0, 1.0, 0.0, 1.0)
    dens, _ = _point_bin_densities(points, bins=180, extent=extent)
    order = np.argsort(dens)
    pts = points[order]
    dens = dens[order]

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=dens,
        s=2.0,
        alpha=0.55,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_facecolor(plt.get_cmap("turbo")(0.0))

    if sample_label is not None:
        ax.text(
            0.95,
            0.92,
            sample_label,
            transform=ax.transAxes,
            fontsize=14,
            color=label_color,
            ha="right",
            va="bottom",
            weight="bold",
        )


def _hist2d_panel(
    ax,
    points,
    *,
    vmax,
    title,
    sample_label=None,
    label_color="white",
    bins=320,
):
    extent = (0.0, 1.0, 0.0, 1.0)
    xedges = np.linspace(extent[0], extent[1], bins + 1)
    yedges = np.linspace(extent[2], extent[3], bins + 1)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[xedges, yedges], density=True)

    ax.imshow(
        hist.T,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$")
    ax.set_title(title)
    ax.set_facecolor(plt.get_cmap("turbo")(0.0))
    ax.grid(color="0.65", linestyle="--", linewidth=1.0, alpha=0.75)

    if sample_label is not None:
        ax.text(
            0.95,
            0.08,
            sample_label,
            transform=ax.transAxes,
            fontsize=14,
            color=label_color,
            ha="right",
            va="bottom",
            weight="bold",
        )
    return hist


def plot_joint_marginals_density(
    dens_exact_2d,
    dens_flow_2d,
    mprime_exact,
    theta_exact,
    dens_m_exact,
    dens_theta_exact,
    mprime_flow,
    theta_flow,
    dens_m_flow,
    dens_theta_flow,
    *,
    title_left="Isobar model",
    title_right="Flow model",
    sample_label=None,
    savepath=None,
):
    left_start = 0.08
    bottom = 0.12
    main_width = 0.32
    main_height = 0.70
    main_gap = 0.03
    marginal_gap = 0.01
    top_height = 0.25
    right_width = 0.10

    lm = [left_start, bottom, main_width, main_height]
    rm = [left_start + main_width + main_gap, bottom, main_width, main_height]
    rtop = [rm[0], rm[1] + rm[3] + marginal_gap, rm[2], top_height]
    rright = [rm[0] + rm[2] + marginal_gap - 0.02, rm[1], right_width, rm[3]]

    fig = plt.figure(figsize=(12, 5))

    ref = np.concatenate([dens_exact_2d.ravel(), dens_flow_2d.ravel()])
    vmax = np.nanpercentile(ref, 99)
    extent = [mprime_exact[0], mprime_exact[-1], theta_exact[0], theta_exact[-1]]

    ax_joint_l = fig.add_axes(lm)
    ax_joint_l.imshow(
        dens_exact_2d,
        origin="lower",
        extent=extent,
        rasterized=True,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax_joint_l.set_xlabel(r"$m'$")
    ax_joint_l.set_ylabel(r"$\theta'$")
    ax_joint_l.set_title(title_left)
    if sample_label is not None:
        ax_joint_l.text(
            0.95,
            0.92,
            sample_label,
            transform=ax_joint_l.transAxes,
            fontsize=14,
            color="white",
            ha="right",
            va="bottom",
            weight="bold",
        )

    ax_joint_r = fig.add_axes(rm)
    ax_joint_r.imshow(
        dens_flow_2d,
        origin="lower",
        extent=extent,
        rasterized=True,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax_joint_r.set_xlabel(r"$m'$")
    ax_joint_r.set_ylabel(r"$\theta'$")
    ax_joint_r.set_title(title_right)

    ax_top_r = fig.add_axes(rtop)
    ax_right_r = fig.add_axes(rright)

    left, _, width, _ = ax_joint_r.get_position().bounds
    ax_top_r.set_position([left, rtop[1], width, rtop[3]])

    xmin, xmax = extent[0], extent[1]
    ymin, ymax = extent[2], extent[3]
    ax_joint_r.set_xlim(xmin, xmax)
    ax_joint_r.set_ylim(ymin, ymax)
    ax_top_r.set_xlim(xmin, xmax)
    ax_right_r.set_ylim(ymin, ymax)

    ax_top_r.fill_between(mprime_exact, dens_m_exact, 0, color="black", alpha=0.15, zorder=1)
    ax_top_r.plot(mprime_exact, dens_m_exact, color="black", lw=1.5, zorder=2, label="Isobar model")
    ax_top_r.fill_between(mprime_flow, dens_m_flow, 0, color="red", alpha=0.15, zorder=1)
    ax_top_r.plot(mprime_flow, dens_m_flow, "--", color="red", lw=1.5, zorder=2, label="Flow model")
    ax_top_r.axis("off")
    ax_top_r.legend(loc="upper right", bbox_to_anchor=(1.04, 1.02), fontsize=12, frameon=False)

    ax_right_r.fill_betweenx(theta_exact, 0, dens_theta_exact, color="black", alpha=0.15, zorder=1)
    ax_right_r.plot(dens_theta_exact, theta_exact, color="black", lw=1.5, zorder=2)
    ax_right_r.fill_betweenx(theta_flow, 0, dens_theta_flow, color="red", alpha=0.15, zorder=1)
    ax_right_r.plot(dens_theta_flow, theta_flow, "--", color="red", lw=1.5, zorder=2)
    ax_right_r.axis("off")

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def plot_C_comparison_density(
    C_exact,
    C_flow,
    *,
    extent=(0.0, 1.0, 0.0, 1.0),
    percentile=98,
    cmap="RdBu_r",
    title_left=r"Isobar model: $\mathcal{C}(m',\theta')$",
    title_right=r"Flow model: $\mathcal{C}(m',\theta')$",
    xlabel=r"$m'$",
    ylabel=r"$\theta'$",
    figsize=(10.5, 5.0),
    dpi=300,
    savepath=None,
):
    ref = np.concatenate([np.ravel(C_exact), np.ravel(C_flow)])
    ref = ref[np.isfinite(ref)]
    vmax = np.nanpercentile(np.abs(ref), percentile) if ref.size else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, (ax_joint_l, ax_joint_r) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )

    im1 = ax_joint_l.imshow(
        C_exact,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax_joint_l.set_title(title_left)
    ax_joint_l.set_xlabel(xlabel)
    ax_joint_l.set_ylabel(ylabel)

    im2 = ax_joint_r.imshow(
        C_flow,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax_joint_r.set_title(title_right)
    ax_joint_r.set_xlabel(xlabel)
    ax_joint_r.set_ylabel(ylabel)

    for ax in (ax_joint_l, ax_joint_r):
        ax.tick_params(direction='out', length=3, width=0.8)

    cbar = fig.colorbar(im2, ax=[ax_joint_l, ax_joint_r], shrink=0.82)
    cbar.set_label(r"$\mathcal{C}(m',\theta')$")

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig


def plot_absS_comparison_density(
    S_exact,
    S_flow,
    *,
    extent=(0.0, 1.0, 0.0, 1.0),
    percentile=99,
    cmap=ABS_S_CMAP,
    title_left=r"Isobar model: $|S(m',\theta')|$",
    title_right=r"Flow model: $|S(m',\theta')|$",
    xlabel=r"$m'$",
    ylabel=r"$\theta'$",
    figsize=(10.5, 5.0),
    dpi=300,
    savepath=None,
):
    ref = np.concatenate([np.ravel(S_exact), np.ravel(S_flow)])
    ref = ref[np.isfinite(ref)]
    vmax = np.nanpercentile(ref, percentile) if ref.size else 1.0

    fig, (ax_joint_l, ax_joint_r) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )

    im1 = ax_joint_l.imshow(
        S_exact,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax_joint_l.set_title(title_left)
    ax_joint_l.set_xlabel(xlabel)
    ax_joint_l.set_ylabel(ylabel)

    im2 = ax_joint_r.imshow(
        S_flow,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax_joint_r.set_title(title_right)
    ax_joint_r.set_xlabel(xlabel)
    ax_joint_r.set_ylabel(ylabel)

    for ax in (ax_joint_l, ax_joint_r):
        ax.tick_params(direction='out', length=3, width=0.8)

    cbar = fig.colorbar(im2, ax=[ax_joint_l, ax_joint_r], shrink=0.82)
    cbar.set_label(r"$|S(m',\theta')|$")

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig


def plot_S2_comparison_density(
    S2_exact,
    S2_flow,
    *,
    extent=(0.0, 1.0, 0.0, 1.0),
    percentile=98,
    cmap="RdBu_r",
    title_left=r"Isobar model: $S^2(m',\theta')$",
    title_right=r"Flow model: $S^2(m',\theta')$",
    xlabel=r"$m'$",
    ylabel=r"$\theta'$",
    figsize=(10.5, 5.0),
    dpi=300,
    savepath=None,
):
    ref = np.concatenate([np.ravel(S2_exact), np.ravel(S2_flow)])
    ref = ref[np.isfinite(ref)]
    vmax = np.nanpercentile(np.abs(ref), percentile) if ref.size else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, (ax_joint_l, ax_joint_r) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )

    im1 = ax_joint_l.imshow(
        S2_exact,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax_joint_l.set_title(title_left)
    ax_joint_l.set_xlabel(xlabel)
    ax_joint_l.set_ylabel(ylabel)

    im2 = ax_joint_r.imshow(
        S2_flow,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax_joint_r.set_title(title_right)
    ax_joint_r.set_xlabel(xlabel)
    ax_joint_r.set_ylabel(ylabel)

    for ax in (ax_joint_l, ax_joint_r):
        ax.tick_params(direction='out', length=3, width=0.8)

    cbar = fig.colorbar(im2, ax=[ax_joint_l, ax_joint_r], shrink=0.82)
    cbar.set_label(r"$S^2(m',\theta') = K\overline{K} - C^2$")

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig


def plot_joint_marginals_samples(
    exact_samples,
    flow_samples,
    mprime_exact,
    theta_exact,
    dens_m_exact,
    dens_theta_exact,
    mprime_flow,
    theta_flow,
    dens_m_flow,
    dens_theta_flow,
    *,
    title_left="Isobar-generated events",
    title_right="Flow-generated events",
    sample_label=None,
    savepath=None,
    max_points=None,
    bins=320,
    seed=1234,
):
    left_start = 0.08
    bottom = 0.12
    main_width = 0.32
    main_height = 0.70
    main_gap = 0.03
    marginal_gap = 0.01
    top_height = 0.25
    right_width = 0.10

    lm = [left_start, bottom, main_width, main_height]
    rm = [left_start + main_width + main_gap, bottom, main_width, main_height]
    rtop = [rm[0], rm[1] + rm[3] + marginal_gap, rm[2], top_height]
    rright = [rm[0] + rm[2] + marginal_gap - 0.02, rm[1], right_width, rm[3]]

    exact_plot = _subsample(exact_samples, max_points=max_points, seed=seed)
    flow_plot = _subsample(flow_samples, max_points=max_points, seed=seed + 1)
    _, hist_exact = _point_bin_densities(exact_plot)
    _, hist_flow = _point_bin_densities(flow_plot)
    vmax = np.nanpercentile(np.concatenate([hist_exact.ravel(), hist_flow.ravel()]), 99)

    fig = plt.figure(figsize=(12, 5))
    extent = [0.0, 1.0, 0.0, 1.0]

    ax_joint_l = fig.add_axes(lm)
    _hist2d_panel(
        ax_joint_l,
        exact_plot,
        vmax=vmax,
        title=title_left,
        sample_label=sample_label,
        label_color="white",
        bins=bins,
    )

    ax_joint_r = fig.add_axes(rm)
    _hist2d_panel(
        ax_joint_r,
        flow_plot,
        vmax=vmax,
        title=title_right,
        bins=bins,
    )

    ax_top_r = fig.add_axes(rtop)
    ax_right_r = fig.add_axes(rright)

    left, _, width, _ = ax_joint_r.get_position().bounds
    ax_top_r.set_position([left, rtop[1], width, rtop[3]])
    ax_top_r.set_xlim(extent[0], extent[1])
    ax_right_r.set_ylim(extent[2], extent[3])

    ax_top_r.fill_between(mprime_exact, dens_m_exact, 0, color="black", alpha=0.15, zorder=1)
    ax_top_r.plot(mprime_exact, dens_m_exact, color="black", lw=1.5, zorder=2, label="Isobar model")
    ax_top_r.fill_between(mprime_flow, dens_m_flow, 0, color="red", alpha=0.15, zorder=1)
    ax_top_r.plot(mprime_flow, dens_m_flow, "--", color="red", lw=1.5, zorder=2, label="Flow model")
    ax_top_r.axis("off")
    ax_top_r.legend(loc="upper right", bbox_to_anchor=(1.04, 1.02), fontsize=12, frameon=False)

    ax_right_r.fill_betweenx(theta_exact, 0, dens_theta_exact, color="black", alpha=0.15, zorder=1)
    ax_right_r.plot(dens_theta_exact, theta_exact, color="black", lw=1.5, zorder=2)
    ax_right_r.fill_betweenx(theta_flow, 0, dens_theta_flow, color="red", alpha=0.15, zorder=1)
    ax_right_r.plot(dens_theta_flow, theta_flow, "--", color="red", lw=1.5, zorder=2)
    ax_right_r.axis("off")

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def plot_conditionals_around_main(
    exact_samples,
    flow_samples,
    flow,
    mprime_slices,
    theta_slices,
    exact_conditional_theta_given_mprime,
    exact_conditional_mprime_given_theta,
    conditional_pdf_func,
    *,
    sample_label=None,
    savepath=None,
    bins=320,
):
    n_top = len(mprime_slices)
    n_right = len(theta_slices)

    main = [0.12, 0.12, 0.58, 0.58]
    top_height = 0.16
    top_box_w = 0.16
    top_gap_w = 0.03
    top_gap_above = 0.10
    right_width = 0.16
    right_box_h = 0.16
    right_gap_h = 0.03
    right_gap_side = 0.10

    m_lim = (0.0, 1.0)
    theta_lim = (0.0, 1.0)

    fig = plt.figure(figsize=(10, 8))
    ax_main = fig.add_axes(main)
    exact_plot = _subsample(exact_samples, max_points=None)
    flow_plot = _subsample(flow_samples, max_points=None)
    _, hist_exact = _point_bin_densities(exact_plot, bins=bins)
    _, hist_flow = _point_bin_densities(flow_plot, bins=bins)
    vmax = np.nanpercentile(np.concatenate([hist_exact.ravel(), hist_flow.ravel()]), 99)
    _hist2d_panel(
        ax_main,
        exact_plot,
        vmax=vmax,
        title="",
        sample_label=None,
        label_color="white",
        bins=bins,
    )
    ax_main.set_title("")
    if sample_label is not None:
        ax_main.text(
            0.95,
            0.05,
            sample_label,
            transform=ax_main.transAxes,
            fontsize=17,
            color="white",
            ha="right",
            va="bottom",
            weight="bold",
        )

    for mp in mprime_slices:
        ax_main.axvline(mp, ls="--", color="gray", lw=1.2, alpha=0.8)
    for thp in theta_slices:
        ax_main.axhline(thp, ls="--", color="gray", lw=1.2, alpha=0.8)

    main_left, main_bottom, main_w, main_h = main
    top_y = main_bottom + main_h + top_gap_above
    total_top_width = n_top * top_box_w + (n_top - 1) * top_gap_w
    top_left0 = main_left + 0.5 * (main_w - total_top_width)

    top_axes = []
    for i in range(n_top):
        left = top_left0 + i * (top_box_w + top_gap_w)
        ax = fig.add_axes([left, top_y, top_box_w, top_height])
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$\theta'$")
        if i == 0:
            ax.set_ylabel(r"$p(\theta'\,|\,m')$", fontsize=9)
        top_axes.append(ax)

    right_x = main_left + main_w + right_gap_side
    total_right_height = n_right * right_box_h + (n_right - 1) * right_gap_h
    right_bottom0 = main_bottom + 0.5 * (main_h - total_right_height)

    right_axes = []
    for j in range(n_right):
        bottom = right_bottom0 + (n_right - 1 - j) * (right_box_h + right_gap_h)
        ax = fig.add_axes([right_x, bottom, right_width, right_box_h])
        ax.tick_params(labelsize=8)
        ax.set_ylabel(r"$p(m'\,|\,\theta')$", fontsize=9)
        if j == n_right - 1:
            ax.set_xlabel(r"$m'$")
        else:
            ax.set_xticklabels([])
        right_axes.append(ax)

    for j, ax in enumerate(right_axes):
        theta0 = theta_slices[-(j + 1)]
        mp, p_exact = exact_conditional_mprime_given_theta(theta0)
        xs, p_flow = conditional_pdf_func(fixed_value=theta0, fixed_axis=1)
        ax.plot(mp, p_exact, lw=1.3, color="black")
        ax.plot(xs, p_flow, "--", lw=1.3, color="red")
        ax.text(0.60, 0.93, rf"$\theta' = {theta0:.2f}$", transform=ax.transAxes,
                fontsize=8, color="black", ha="left", va="top")

    for j, ax in enumerate(top_axes):
        m0 = mprime_slices[j]
        thp, p_exact = exact_conditional_theta_given_mprime(m0)
        xs, p_flow = conditional_pdf_func(fixed_value=m0, fixed_axis=0)
        ax.plot(thp, p_exact, lw=1.3, color="black")
        ax.plot(xs, p_flow, "--", lw=1.3, color="red")
        xpos = 0.40 if j == 1 else 0.08
        ax.text(xpos, 0.93, rf"$m' = {m0:.2f}$", transform=ax.transAxes,
                fontsize=8, color="black", ha="left", va="top")

    for i, ax_top in enumerate(top_axes):
        x_main = float(mprime_slices[i])
        for x_frac in (0.0, 1.0):
            cp = ConnectionPatch(
                xyA=(x_main, theta_lim[1]),
                coordsA=ax_main.transData,
                axesA=ax_main,
                xyB=(x_frac, 0.0),
                coordsB=ax_top.transAxes,
                axesB=ax_top,
                color="gray",
                lw=0.75,
                alpha=0.5,
                zorder=1000,
                clip_on=False,
            )
            fig.add_artist(cp)

    for j in range(n_right):
        y_main = float(theta_slices[j])
        ax_right = right_axes[-(j + 1)]
        for y_frac in (0.0, 1.0):
            cp = ConnectionPatch(
                xyA=(m_lim[1], y_main),
                coordsA=ax_main.transData,
                axesA=ax_main,
                xyB=(0.0, y_frac),
                coordsB=ax_right.transAxes,
                axesB=ax_right,
                color="gray",
                lw=0.75,
                alpha=0.5,
                zorder=1000,
                clip_on=False,
            )
            fig.add_artist(cp)

    custom_lines = [
        Line2D([0], [0], color="black", lw=1.3, ls="-"),
        Line2D([0], [0], color="red", lw=1.3, ls="--"),
    ]
    fig.legend(custom_lines, ["Isobar model", "Flow model"], loc="upper right",
               bbox_to_anchor=(0.98, 0.95), fontsize=15, frameon=False)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig
