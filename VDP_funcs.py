"""van-der-pol.py: Plot Van der Pol Equation"""
"""https://github.com/ishidur/Van_der_Pol_visualizer"""
__author__      = "Ryota Ishidu and Brandon Buncher"

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import cm
import matplotlib.colors as Colors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

plt.rc('text', usetex = True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

def VDP_deriv(x, t, mu = 1.0):
    nx0 = mu * x[1] - mu * (x[0] ** 3.0 / 3.0 - x[0])
    nx1 = -x[0] / mu
    res = np.array([nx0, nx1])
    return res

def VDP_solve(t_max = 5000.0, x0 = 0.0, z0 = 0.1, mu = 1.0):
    ts = np.linspace(0.0, t_max, int(t_max) * 1000)
    xs = odeint(VDP_deriv, [x0, z0], ts, (mu,))
    
    return ts, xs

def plot_F(axis, c = (0.2, 0.2, 0.2, 0.8), ls = '-', lw = 1.0):
    F = lambda x: x ** 3.0 / 3.0 - x
    xlim, ylim = axis.get_xlim(), axis.get_ylim()
    xF = np.linspace(np.min(xlim), np.max(xlim), 1000)
    axis.plot(xF, F(xF), c = c, ls = ls, lw = lw, zorder = 0.0)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)

    return

def plot_phase_space(data_in = None, t_max = 5000.0, x0 = 0.0, z0 = 0.1, mu = 1.0, resolution = 30, line_color = 'k', quiv_cmap = cm.gnuplot, arrow_locs = None, arrowsize = 1.0, arrow_fc = None, arrow_ec = None, arrow_style = '->', arrow_lw = 1.0, figsize = (10, 10), cbar_params = None, norm_type = 'linear', filename = None):
    if data_in is not None:
        ts, xs = data_in
    else:
        ts, xs = VDP_solve(t_max, x0, z0, mu)
    xs_peak1 = np.amax([np.amax(xs[:,0]),np.fabs(np.amin(xs[:,0]))]) + 1
    xs_peak2 = np.amax([np.amax(xs[:,1]),np.fabs(np.amin(xs[:,1]))]) + 1
    X, Z = np.meshgrid(np.linspace(-xs_peak1, xs_peak1, resolution), np.linspace(-xs_peak2, xs_peak2, resolution))
    U = mu * Z - mu * (X ** 3.0 / 3.0 - X)
    V = -X / mu #mu * (1.0 - X ** 2.0) * Z - X
    C = np.hypot(U, V)
    fig_phase, ax_phase = plt.subplots(figsize = figsize)
    U_norm = U / np.sqrt(U**2 + V**2)
    V_norm = V / np.sqrt(U**2 + V**2)
    Q = ax_phase.quiver(X, Z, U_norm, V_norm, C, units = 'xy', cmap = quiv_cmap)
    if cbar_params is not None:
        left, width, normed = cbar_params
        if normed:
            if norm_type == 'log':
                norm = Colors.LogNorm(vmin = Q.get_array().min(), vmax = Q.get_array().max())
            else:
                norm = Colors.Normalize(vmin = Q.get_array().min(), vmax = Q.get_array().max())
            Q.set_norm(norm)
        cbar, cbax = cbar_gen(Q, left, width, norm)
    else:
        fig_phase.colorbar(Q)
    line = ax_phase.plot(xs[:, 0], xs[:, 1], c = line_color)[0]
    if arrow_locs is not None:
        arrows = add_arrow_to_line2D(ax_phase, line, arrow_locs = arrow_locs, arrowstyle = arrow_style, arrowsize = arrowsize, facecolor = arrow_fc, edgecolor = arrow_ec, linewidth = arrow_lw)
    ax_phase.set_xlim(-xs_peak1, xs_peak1)
    ax_phase.set_ylim(-xs_peak2, xs_peak2)
    ax_phase.set_xlabel('$x$', fontsize = 20)
    ax_phase.set_ylabel('$z$', fontsize = 20)
    #ax_phase.set_facecolor(facecolor)

    xlim, ylim = ax_phase.get_xlim(), ax_phase.get_ylim()

    ax_phase.plot([*xlim], [0.0, 0.0], 'k-')
    ax_phase.plot([0.0, 0.0], [*ylim], 'k-')

    ax_phase.set_xlim(*xlim)
    ax_phase.set_ylim(*ylim)

    fig_phase.canvas.draw()
    
    if filename is not None:
        plt.savefig(filename)
    return fig_phase, ax_phase

def plot_vib(data_in = None, t_max = 50.0, mu = 1.0, filename = None):
    if data_in is not None:
        ts, cs = data_in
    else:
        ts, xs = VDP_solve(t_max, mu)
    fig_vib, ax_vib = plt.subplots(figsize = (8, 8))
    ax_vib.plot(ts, xs[:,0])
    ax_vib.set_ylabel('$q$')
    ax_vib.set_xlabel('$\tau$')
    if filename is not None:
        plt.savefig(filename)
    return fig_vib, ax_vib





def cbar_gen(im, left, width, norm = None):
    '''
    Add a colorbar with height and bottom corresponding to the height and bottom of axis ax

    im: the image to extract the figure and axes from OR a list containing [fig, ax, sm]
    left: the 'left' parameter in colorbar generation
    width: the 'width' parameter in colorbar generaion
    norm: the matplotlib.colors.Normalize instance;
    if norm == None, the scalar mappable used will be
    that of the colormap from the image.
    '''
    try:
        ax = im.axes
        fig = im.figure
    except Exception:
        fig, ax, sm = im

    fig.canvas.draw()

    ax_c = ax.bbox.get_points()
    fig_c = fig.bbox.get_points()
    ax_wh = ax_c[1] - ax_c[0]
    fig_wh = fig_c[1] - fig_c[0]
    scale = ax_wh[1] / fig_wh[1]
    bottom = ax_c[0, 1] / fig_wh[1]

    cbax = fig.add_axes([left, bottom, width, scale])
    try:
        cbar = fig.colorbar(sm, cax = cbax)
    except Exception:
        sm = cm.ScalarMappable(cmap = im.cmap, norm = norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax = cbax)

    return cbar, cbax




def add_arrow_to_line2D(axes, line, arrow_locs = [0.2, 0.4, 0.6, 0.8], arrowstyle = '->', arrowsize = 1, facecolor = None, edgecolor = None, linewidth = None, transform = None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows

    Found at:
    ---------
    https://stackoverflow.com/questions/26911898/matplotlib-curve-with-arrow-ticks
    """
    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        if edgecolor is None:
            edgecolor = color
        arrow_kw['edgecolor'] = edgecolor
        if facecolor is None:
            facecolor = color
        arrow_kw['facecolor'] = facecolor

    #linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        if linewidth is None:
            linewidth = line.get_linewidth()
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    zorder_max = np.max([child.zorder for child in axes.get_children()])
    for arrow in arrows:
        arrow.set_zorder(zorder_max)
    return arrows



def animate_phase_space_setup(data_in = None, t_max = 5000.0, x0 = 0.0, z0 = 0.1, mu = 1.0, resolution = 30, quiv_cmap = cm.gnuplot, figsize = (10, 10), cbar_params = None, norm_type = 'linear'):
    if data_in is not None:
        ts, xs = data_in
    else:
        ts, xs = VDP_solve(t_max, x0, z0, mu)
    xs_peak1 = np.amax([np.amax(xs[:,0]),np.fabs(np.amin(xs[:,0]))]) + 1
    xs_peak2 = np.amax([np.amax(xs[:,1]),np.fabs(np.amin(xs[:,1]))]) + 1
    X, Z = np.meshgrid(np.linspace(-xs_peak1, xs_peak1, resolution), np.linspace(-xs_peak2, xs_peak2, resolution))
    U = mu * Z - mu * (X ** 3.0 / 3.0 - X)
    V = -X / mu #mu * (1.0 - X ** 2.0) * Z - X
    C = np.hypot(U, V)
    fig_phase, ax_phase = plt.subplots(figsize = figsize)
    U_norm = U / np.sqrt(U**2 + V**2)
    V_norm = V / np.sqrt(U**2 + V**2)
    Q = ax_phase.quiver(X, Z, U_norm, V_norm, C, units = 'xy', cmap = quiv_cmap)
    if cbar_params is not None:
        left, width, normed = cbar_params
        if normed:
            if norm_type == 'log':
                norm = Colors.LogNorm(vmin = Q.get_array().min(), vmax = Q.get_array().max())
            else:
                norm = Colors.Normalize(vmin = Q.get_array().min(), vmax = Q.get_array().max())
            Q.set_norm(norm)
        cbar, cbax = cbar_gen(Q, left, width, norm)
    else:
        fig_phase.colorbar(Q)
    ax_phase.set_xlim(-xs_peak1, xs_peak1)
    ax_phase.set_ylim(-xs_peak2, xs_peak2)
    ax_phase.set_xlabel('$x$', fontsize = 20)
    ax_phase.set_ylabel('$z$', fontsize = 20)

    xlim, ylim = ax_phase.get_xlim(), ax_phase.get_ylim()

    ax_phase.plot([*xlim], [0.0, 0.0], 'k-')
    ax_phase.plot([0.0, 0.0], [*ylim], 'k-')

    ax_phase.set_xlim(*xlim)
    ax_phase.set_ylim(*ylim)

    return fig_phase, ax_phase, ts, xs
