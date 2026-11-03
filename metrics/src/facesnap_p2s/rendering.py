"""Frontal mesh rendering and the per-frame comparison figure.

A small software rasteriser for the shaded mesh panels
plus a scatter layer for the error field. 
"""

import os.path as osp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.titlesize': 21,
    'axes.labelsize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 17,
    'figure.titlesize': 23,
})


FRONT_AXES = {
    'xz': (0, 2, 1, +1),
    '-xz': (0, 2, 1, -1),
    'xy': (0, 1, 2, +1),
    '-xy': (0, 1, 2, -1),
    'yz': (1, 2, 0, +1),
    '-yz': (1, 2, 0, -1),
}

# Fixed frame in mesh-space (mm), 'xy' projection: u=x-axis, v=y-axis.
FIXED_FRAME_XLIM = (-10.0, 10.0)
FIXED_FRAME_YLIM = (-12.0, 14.0)

# Offscreen raster resolution for per-channel compositing
_RASTER_PX = 600


def make_error_cmap(color_floor_frac=0.0):
    """Dark-blue -> green -> yellow -> red ramp."""
    color_floor_frac = float(np.clip(color_floor_frac, 0.0, 0.999))
    base_stops = [(0.0, '#00008B'), (1.0 / 3.0, '#00BB00'),
                  (2.0 / 3.0, '#FFFF00'), (1.0, '#FF0000')]

    if color_floor_frac <= 0.0:
        stops = base_stops
    else:
        scale = 1.0 - color_floor_frac
        stops = [(0.0, '#00008B'), (color_floor_frac, '#00008B')]
        stops += [(color_floor_frac + pos * scale, color) for pos, color in base_stops[1:]]

    return LinearSegmentedColormap.from_list('p2s_err', stops, N=256)


def _compute_vertex_normals(verts, faces):
    """Area-weighted, unit-length vertex normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)

    vn = np.zeros_like(verts, dtype=np.float64)
    np.add.at(vn, faces[:, 0], face_n)
    np.add.at(vn, faces[:, 1], face_n)
    np.add.at(vn, faces[:, 2], face_n)

    mag = np.linalg.norm(vn, axis=1, keepdims=True)
    mag = np.where(mag < 1e-12, 1.0, mag)
    return (vn / mag).astype(np.float32)


def _vertex_shade(verts, faces, depth_axis, sign,
                  ambient=0.28, diffuse=0.70, specular=0.20, shininess=20.0):
    """Per-vertex Blinn-Phong intensity (V,), values typically in [0.15, 1.35]."""
    light_dir = np.zeros(3, dtype=np.float32)
    light_dir[depth_axis] = -sign
    view_dir = light_dir.copy()
    half = light_dir + view_dir
    hn = np.linalg.norm(half)
    if hn > 1e-12:
        half = half / hn

    vn = _compute_vertex_normals(verts, faces)
    diff = np.clip((vn * light_dir).sum(axis=1), 0.0, 1.0)
    spec = np.clip((vn * half).sum(axis=1), 0.0, 1.0) ** shininess
    return (ambient + diffuse * diff + specular * spec).astype(np.float64)


def _depth_sorted_faces(verts, faces, depth_axis, sign):
    """Return faces sorted back-to-front by mean depth."""
    d = verts[:, depth_axis] * sign
    order = np.argsort(-d[faces].mean(axis=1))
    return faces[order]


def _render_rgb_gouraud(verts, faces, vertex_rgb, depth_axis, sign, px=_RASTER_PX):
    """Render mesh with per-vertex RGB Gouraud shading into an (H, W, 3) uint8 image.

    Uses matplotlib.tri.tripcolor independently per channel on a small offscreen
    figure so each channel is Gouraud-interpolated per pixel — fully smooth, no edges.
    """
    # FRONT_AXES 'xy': iu=0 (x), iv=1 (y)
    u = verts[:, 0].astype(np.float64)
    v = verts[:, 1].astype(np.float64)

    sorted_f = _depth_sorted_faces(verts, faces, depth_axis, sign)

    triang = mtri.Triangulation(u, v, sorted_f)

    grey_cmap = LinearSegmentedColormap.from_list('_g01', [(0, 0, 0), (1, 1, 1)], N=512)

    dpi = 100
    fw = fh = px / dpi
    channels = []
    for ch in range(3):
        vals = np.clip(vertex_rgb[:, ch], 0.0, 1.0)
        fig_off = plt.figure(figsize=(fw, fh), dpi=dpi)
        ax_off = fig_off.add_axes([0, 0, 1, 1])
        ax_off.set_facecolor('white')
        ax_off.set_xlim(*FIXED_FRAME_XLIM)
        ax_off.set_ylim(*FIXED_FRAME_YLIM)
        ax_off.set_aspect('equal')
        ax_off.axis('off')
        ax_off.tripcolor(triang, vals, shading='gouraud',
                         cmap=grey_cmap, vmin=0.0, vmax=1.0)
        fig_off.canvas.draw()
        buf = np.frombuffer(fig_off.canvas.tostring_rgb(), dtype=np.uint8)
        h_px = fig_off.canvas.get_width_height()[1]
        w_px = fig_off.canvas.get_width_height()[0]
        buf = buf.reshape(h_px, w_px, 3)
        channels.append(buf[:, :, 0])  # greyscale channel carries the scalar
        plt.close(fig_off)

    return np.stack(channels, axis=2)  # (H, W, 3) uint8


def render_mesh_frontal(ax, verts, faces, front_axis='xy',
                        vertex_colors=None, base_color=(0.86, 0.86, 0.86),
                        title='', depth_sign=None):
    """Gouraud-shaded mesh render on `ax`.

    For a grey mesh (vertex_colors=None): single tripcolor pass with a grey cmap —
    intensity smoothly interpolated per pixel, no polygon edges visible.
    For coloured meshes: renders each RGB channel separately and composites via imshow.
    """
    iu, iv, id_, default_sign = FRONT_AXES[front_axis]
    sign = default_sign if depth_sign is None else depth_sign

    intensity = _vertex_shade(verts, faces, id_, sign)

    if vertex_colors is None:
        # Single-channel grey path: encode shaded grey as a scalar, use tripcolor shading='gouraud'
        # Build a 2-stop colormap from black to base_color scaled to the max possible intensity.
        base = np.asarray(base_color, dtype=np.float64)
        i_max = 1.35  # theoretical max of ambient+diffuse+specular with our defaults
        dark = tuple(np.clip(base * 0.0, 0, 1))   # black (ambient * 0 extreme)
        light = tuple(np.clip(base * i_max, 0, 1))
        grey_cmap = LinearSegmentedColormap.from_list('_grey', [dark, light], N=512)

        sorted_f = _depth_sorted_faces(verts, faces, id_, sign)
        u = verts[:, iu].astype(np.float64)
        v = verts[:, iv].astype(np.float64)
        triang = mtri.Triangulation(u, v, sorted_f)

        ax.set_facecolor('white')
        ax.tripcolor(triang, intensity, shading='gouraud',
                     cmap=grey_cmap, vmin=0.0, vmax=i_max)
    else:
        vc = np.asarray(vertex_colors, dtype=np.float64)
        vertex_rgb = np.clip(vc * intensity[:, None], 0.0, 1.0)
        rgb_img = _render_rgb_gouraud(verts, faces, vertex_rgb, id_, sign)
        ax.imshow(rgb_img, origin='upper',
                  extent=[FIXED_FRAME_XLIM[0], FIXED_FRAME_XLIM[1],
                          FIXED_FRAME_YLIM[0], FIXED_FRAME_YLIM[1]],
                  aspect='equal', interpolation='bilinear')

    ax.set_title(title)
    return verts[:, [iu, iv]]


def _setup_mesh_ax(ax):
    ax.set_xlim(*FIXED_FRAME_XLIM)
    ax.set_ylim(*FIXED_FRAME_YLIM)
    ax.set_aspect('equal')
    ax.set_axis_off()


def create_multi_method_figure(scan_data, methods_data, method_names, args, save_path,
                               method_groups=None):
    """Single-row layout, one panel per thing being shown.

    Raw scan | <method> mesh | <method> heat map | <method2> mesh | ...

    i.e. 1 + 2*n_methods panels in a row.  Columns are separated by thin
    spacers.

    method_groups: list of (label, first_method_idx, last_method_idx) using 0-based
        method indices.  Adjacent groups are separated by a wider spacer and get a
        bracket label drawn above the top row.
    """
    n_methods = len(method_names)
    # scan panel, then a (grey mesh, heat map) pair per method
    n_cols = 1 + 2 * n_methods
    n_cols_total = n_cols * 2 - 1   # interleave spacer columns between each pair

    # Build set of method-column boundaries that sit between groups.
    group_dividers = set()  # logical column indices i such that spacer after col i is wide
    if method_groups:
        for k in range(len(method_groups) - 1):
            last_of_group = method_groups[k][2]   # last method idx (0-based)
            group_dividers.add(2 + 2 * last_of_group)  # last panel of that method

    col_w        = 1.0
    spacer_w     = 0.06
    spacer_group = 0.32   # wide spacer between groups
    width_ratios = []
    for i in range(n_cols):
        width_ratios.append(col_w)
        if i < n_cols - 1:
            width_ratios.append(spacer_group if i in group_dividers else spacer_w)

    fig_w = n_cols * 3.6
    # Each content cell must be exactly cell_w wide and cell_w * data_aspect tall
    # so that set_aspect('equal') fills the cell with no padding.
    cell_w = fig_w / sum(width_ratios)
    data_aspect = ((FIXED_FRAME_YLIM[1] - FIXED_FRAME_YLIM[0]) /
                   (FIXED_FRAME_XLIM[1] - FIXED_FRAME_XLIM[0]))
    fig_h = cell_w * data_aspect
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, n_cols_total, width_ratios=width_ratios,
                          wspace=0.0)

    axes = []
    col_idx = 0
    for i in range(n_cols):
        axes.append(fig.add_subplot(gs[0, col_idx]))
        col_idx += 1
        if i < n_cols - 1:
            col_idx += 1  # skip spacer

    # The per-method code below was written against a 2-row grid.  Alias the
    # single row onto it: for method i, "top" is its grey mesh panel and "bot"
    # is its heat-map panel.  Index 0 of both is the scan panel.
    top_axes = [axes[0]] + [axes[1 + 2 * i] for i in range(n_methods)]
    bot_axes = [axes[0]] + [axes[2 + 2 * i] for i in range(n_methods)]

    cmap_span = max(args.err_max - args.err_min, 1e-12)
    color_floor_frac = (args.color_floor - args.err_min) / cmap_span
    cmap = make_error_cmap(color_floor_frac=color_floor_frac)
    norm = plt.Normalize(vmin=args.err_min, vmax=args.err_max)

    render_axis = 'xy'
    depth_sign = -1
    iu, iv, id_, _ = FRONT_AXES[render_axis]

    # ------------------------------------------------------------------ #
    # Panel 0: the ground-truth scan                                      #
    # ------------------------------------------------------------------ #
    ax_scan = axes[0]

    render_mesh_frontal(
        ax_scan, scan_data['scan_verts_aligned'], scan_data['raw_faces'],
        front_axis=render_axis, title='Raw scan', depth_sign=depth_sign,
    )
    _setup_mesh_ax(ax_scan)

    # ------------------------------------------------------------------ #
    # Per method: grey mesh panel, then heat-map panel                    #
    # ------------------------------------------------------------------ #
    # Determine which names are multi-line (split at '+') so single-line names
    # can be padded with a leading newline to stay vertically centred.
    def _split_name(name):
        """Split 'A+B Suffix' into 'A+\nB Suffix' at the last '+' if name contains '+'."""
        if '+' in name:
            idx = name.rindex('+')
            return name[:idx + 1] + '\n' + name[idx + 1:].lstrip()
        return name

    any_multiline = any('+' in n for n in method_names)

    for i, (method_data, method_name) in enumerate(zip(methods_data, method_names)):
        ci = 1 + i
        ax_mesh = top_axes[ci]
        ax_pts  = bot_axes[ci]

        mesh_v = method_data['mesh_verts_aligned']
        mesh_f = method_data['mesh_faces']
        err    = method_data['err']

        display_name = _split_name(method_name)
        if any_multiline and '\n' not in display_name:
            display_name = '\n' + display_name  # pad to align vertical centre

        render_mesh_frontal(
            ax_mesh, mesh_v, mesh_f, front_axis=render_axis,
            title=display_name, depth_sign=depth_sign,
        )
        _setup_mesh_ax(ax_mesh)

        if args.color_mode == 'vertex':
            vert_rgba = cmap(norm(np.clip(err, args.err_min, args.err_max)))
            vert_rgb  = vert_rgba[:, :3].astype(np.float32)

            scan_pts_aligned = scan_data['scan_pts_aligned']
            region_min = scan_pts_aligned.min(axis=0) - args.region_padding
            region_max = scan_pts_aligned.max(axis=0) + args.region_padding
            vert_in_region = np.all(
                (mesh_v >= region_min) & (mesh_v <= region_max), axis=1
            )
            base_color_arr = np.full((len(mesh_v), 3), 0.78, dtype=np.float32)
            base_color_arr[vert_in_region] = vert_rgb[vert_in_region]

            render_mesh_frontal(
                ax_pts, mesh_v, mesh_f, front_axis=render_axis,
                vertex_colors=base_color_arr,
                title='P2S error', depth_sign=depth_sign,
            )
        else:
            render_mesh_frontal(
                ax_pts, mesh_v, mesh_f, front_axis=render_axis,
                title='P2S error', depth_sign=depth_sign,
            )
            scan_pts_aligned = scan_data['scan_pts_aligned']
            pts_uv = scan_pts_aligned[:, [iu, iv]]
            pts_depth = scan_pts_aligned[:, id_] * depth_sign
            order = np.argsort(-pts_depth)
            ax_pts.scatter(
                pts_uv[order, 0], pts_uv[order, 1],
                c=np.clip(err[order], args.err_min, args.err_max),
                cmap=cmap, norm=norm,
                s=args.point_size, marker='.', linewidths=0,
                alpha=args.point_alpha, zorder=3,
            )

        _setup_mesh_ax(ax_pts)

    # Flush layout once so all get_position() calls below are accurate.
    fig.canvas.draw()

    # --- Group brackets + dividers ---
    # Collect all annotation artists so they can be hidden for the unlabeled save.
    label_artists = []   # method title Text objects from ax.set_title()
    bracket_artists = [] # bracket lines + labels added to the figure (hidden in unlabeled)
    divider_artists = [] # vertical dashed dividers (kept in unlabeled)

    for ax in top_axes[1:]:
        label_artists.append(ax.title)

    if method_groups:
        trans = fig.transFigure

        for g_idx, (label, m_first, m_last) in enumerate(method_groups):
            ax_l = top_axes[1 + m_first]
            ax_r = top_axes[1 + m_last]
            x0 = ax_l.get_position().x0
            x1 = ax_r.get_position().x1
            y_top = ax_l.get_position().y1
            bracket_y   = y_top + 0.085
            label_y     = y_top + 0.095
            tick_h      = 0.010

            for coords in [([x0, x1], [bracket_y, bracket_y]),
                           ([x0, x0], [bracket_y - tick_h, bracket_y]),
                           ([x1, x1], [bracket_y - tick_h, bracket_y])]:
                artist = mlines.Line2D(coords[0], coords[1],
                                       transform=trans, color='#444444',
                                       linewidth=0.9, clip_on=False)
                fig.add_artist(artist)
                bracket_artists.append(artist)

            txt = fig.text((x0 + x1) / 2, label_y, label,
                           ha='center', va='bottom',
                           fontsize=plt.rcParams['axes.titlesize'],
                           color='#222222', transform=trans)
            bracket_artists.append(txt)

            if g_idx > 0:
                ax_prev = top_axes[1 + method_groups[g_idx - 1][2]]
                ax_cur  = top_axes[1 + m_first]
                gap_cx = (ax_prev.get_position().x1 + ax_cur.get_position().x0) / 2
                y_bottom = bot_axes[0].get_position().y0
                line = mlines.Line2D(
                    [gap_cx, gap_cx], [y_bottom, y_top],
                    transform=trans, color='#bbbbbb', linewidth=0.8,
                    linestyle='--', clip_on=False)
                fig.add_artist(line)
                divider_artists.append(line)

    # Place a narrow colorbar axis to the right of the bottom row, outside the last
    # content column, so it never affects any cell's size.
    last_bot = bot_axes[-1].get_position()
    cbar_w   = 0.006
    cbar_pad = 0.008
    cbar_h   = last_bot.height * 0.80
    cbar_ax  = fig.add_axes([
        last_bot.x1 + cbar_pad,
        last_bot.y0 + (last_bot.height - cbar_h) / 2,
        cbar_w,
        cbar_h,
    ])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Error (mm)')

    # --- Labeled save ---
    plt.savefig(save_path, dpi=args.output_dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved comparison figure to {save_path}')

    # --- Unlabeled save (no method names, no brackets) ---
    # for a in label_artists + bracket_artists:
    #     a.set_visible(False)
    # base, ext = osp.splitext(save_path)
    # unlabeled_path = base + '_unlabeled' + ext
    # plt.savefig(unlabeled_path, dpi=args.output_dpi, bbox_inches='tight', facecolor='white')
    # print(f'Saved unlabeled figure to {unlabeled_path}')

    plt.close(fig)
