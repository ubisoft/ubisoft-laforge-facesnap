"""Mesh and scan I/O for the FaceSnap P2S benchmark.

Readers for everything the metric needs, on numpy alone:

* :func:`load_scan`   -- ground-truth scan, released ``.ply`` or legacy pickle
* :func:`load_obj`    -- a method's registered mesh
* :func:`load_scan_transform` / :func:`save_scan_transform`
                      -- the per-frame raw -> canonical scan transform

Two conventions here are load-bearing:

* **Quad triangulation.** If your registered mesh is a quad mesh, each quad
  ``[a, b, c, d]`` becomes ``[a, b, c]`` and ``[c, d, a]``.
* **PLY axis convention.** The released scans are Blender Z-up while the scan
  transform is in the capture rig's Y-up frame.  See :func:`ply_axes_to_raw`.
"""

import json
import os.path as osp
import pickle

import numpy as np


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def load_scan_cache(path):
    """Read a legacy ``.abc.cache`` scan pickle.

    A pickle file holding ``{'verts', 'faces', 'uvs', 'landmarks'}`` as numpy arrays, storing
    an unindexed triangle soup (3 vertices per face).  Kept so internal
    captures still load; released data is ``.ply`` -- see :func:`load_scan`.

    Returns a dict with:
      verts     (N, 3) float32  -- raw scan space
      faces     (F, 3) int64    -- triangles
      landmarks (51, 3) float32 or None
      uvs       (N, 2) float32 or None
    """
    with open(path, 'rb') as fh:
        dic = pickle.load(fh)

    for key in ('verts', 'faces'):
        if key not in dic:
            raise KeyError(f"{path!r} has no {key!r} (keys: {sorted(dic)})")

    out = {
        'verts': np.ascontiguousarray(dic['verts'], dtype=np.float32),
        'faces': np.ascontiguousarray(dic['faces'], dtype=np.int64),
        'landmarks': None,
        'uvs': None,
    }
    if dic.get('landmarks') is not None:
        out['landmarks'] = np.ascontiguousarray(dic['landmarks'], dtype=np.float32)
    if dic.get('uvs') is not None:
        # V is flipped on load to match the capture pipeline's convention.
        # The metric never touches UVs; this is only for parity.
        uvs = np.ascontiguousarray(dic['uvs'], dtype=np.float32)
        uvs[..., 1] = 1.0 - uvs[..., 1]
        out['uvs'] = uvs

    if out['verts'].ndim != 2 or out['verts'].shape[-1] != 3:
        raise ValueError(f"bad verts shape {out['verts'].shape}")
    if out['faces'].ndim != 2 or out['faces'].shape[-1] != 3:
        raise ValueError(f"bad faces shape {out['faces'].shape}")
    return out


# ---------------------------------------------------------------------------
# Scan transform (replaces reading FaceSnap's stage3_output.pickle)
# ---------------------------------------------------------------------------

def load_scan_transform(path):
    """Load ``refined_scan_{scale,trans,rot}`` from JSON or a stage3 pickle.

    JSON schema (what the benchmark ships)::

        {"scale": 0.47893092,
         "rot_euler": [-1.3256601, -0.7435547, -0.18129608],
         "trans": [1.477164, -14.108302, 8.043725]}

    ``rot_euler`` is in radians and is composed as Rz @ Ry @ Rx.

    Returns (scale: float, rot_euler: (3,) float32, trans: (3,) float32).
    """
    if path.endswith('.json'):
        with open(path, 'r') as fh:
            dic = json.load(fh)
        scale = float(np.asarray(dic['scale']).reshape(-1)[0])
        rot = np.asarray(dic['rot_euler'], dtype=np.float32).reshape(3)
        trans = np.asarray(dic['trans'], dtype=np.float32).reshape(3)
        return scale, rot, trans

    with open(path, 'rb') as fh:
        dic = pickle.load(fh)
    scale = float(np.asarray(dic['refined_scan_scale']).reshape(-1)[0])
    rot = np.asarray(dic['refined_scan_rot'], dtype=np.float32).reshape(3)
    trans = np.asarray(dic['refined_scan_trans'], dtype=np.float32).reshape(3)
    return scale, rot, trans


def save_scan_transform(path, scale, rot_euler, trans):
    """Write the transform as JSON so the released data carries no pickle."""
    with open(path, 'w') as fh:
        json.dump({
            'scale': float(scale),
            'rot_euler': [float(v) for v in np.asarray(rot_euler).reshape(3)],
            'trans': [float(v) for v in np.asarray(trans).reshape(3)],
            '_comment': 'rot_euler in radians, composed Rz @ Ry @ Rx; '
                        'aligned = R @ (scale * v_raw) + trans',
        }, fh, indent=2)


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

def load_obj(path):
    """Minimal OBJ reader returning ``{'verts': (N,3), 'tris': (T,3)}``.

    Handles ``v`` lines and ``f`` lines with 3 or more corners in any of the
    ``i``, ``i/j``, ``i//k``, ``i/j/k`` forms.  Negative (relative) indices are
    supported.  Polygons with more than 3 corners are split as corners
    ``[a, b, c, d]`` -> ``[a, b, c]`` + ``[c, d, a]``; for n > 4 this
    generalises to a fan from the first corner.
    """
    verts = []
    faces = []

    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            if not line:
                continue
            tag = line[:2]
            if tag == 'v ':
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == 'f ':
                idx = []
                for tok in line.split()[1:]:
                    first = tok.split('/')[0]
                    if not first:
                        continue
                    idx.append(int(first))
                if len(idx) < 3:
                    continue
                faces.append(idx)

    if not verts:
        raise ValueError(f"no vertices found in {path!r}")

    verts = np.asarray(verts, dtype=np.float32)
    n_verts = len(verts)

    tris = []
    for idx in faces:
        # OBJ is 1-based; negative indices count back from the end.
        conv = [(i - 1) if i > 0 else (n_verts + i) for i in idx]
        if len(conv) == 3:
            tris.append(conv)
        elif len(conv) == 4:
            a, b, c, d = conv
            tris.append([a, b, c])
            tris.append([c, d, a])
        else:
            a = conv[0]
            for k in range(1, len(conv) - 1):
                tris.append([a, conv[k], conv[k + 1]])

    tris = np.asarray(tris, dtype=np.int64)
    if tris.size and (tris.min() < 0 or tris.max() >= n_verts):
        raise ValueError(
            f"{path!r}: face index out of range "
            f"[{tris.min()}, {tris.max()}] for {n_verts} vertices"
        )
    return {'verts': verts, 'tris': tris}


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

_PLY_DTYPES = {
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
}


def load_ply(path):
    """Read a PLY mesh.  Returns ``{'verts', 'faces', 'uvs'}``.

    Supports ``binary_little_endian`` and ``ascii``, arbitrary vertex property
    layouts (``x/y/z`` located by name, ``s/t`` or ``u/v`` taken as UVs when
    present), and face lists with any index type.  Polygons with more than 3
    corners are fanned from the first corner, matching ``load_obj``.

    This is the released scan format: shared vertices plus UVs, where the
    legacy ``.abc.cache`` stored an unindexed triangle soup of the same
    surface.
    """
    with open(path, 'rb') as fh:
        if fh.readline().strip() != b'ply':
            raise ValueError(f'{path!r} is not a PLY file')

        fmt = None
        elements = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f'{path!r}: unexpected EOF in header')
            tok = line.decode('ascii', 'replace').split()
            if not tok or tok[0] == 'comment':
                continue
            if tok[0] == 'format':
                fmt = tok[1]
            elif tok[0] == 'element':
                elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == 'property':
                if not elements:
                    raise ValueError(f'{path!r}: property before element')
                if tok[1] == 'list':
                    elements[-1][2].append((tok[4], ('list', tok[2], tok[3])))
                else:
                    elements[-1][2].append((tok[2], tok[1]))
            elif tok[0] == 'end_header':
                break

        if fmt not in ('binary_little_endian', 'ascii'):
            raise NotImplementedError(f'{path!r}: unsupported PLY format {fmt!r}')
        body = fh.read()

    verts = uvs = None
    faces = []
    offset = 0
    ascii_tokens = body.split() if fmt == 'ascii' else None
    ascii_pos = 0

    for name, count, props in elements:
        is_list = any(isinstance(t, tuple) for _, t in props)

        if not is_list:
            if fmt == 'binary_little_endian':
                dt = np.dtype([(pn, '<' + _PLY_DTYPES[pt]) for pn, pt in props])
                arr = np.frombuffer(body, dtype=dt, count=count, offset=offset)
                offset += dt.itemsize * count
            else:
                ncol = len(props)
                flat = np.asarray(ascii_tokens[ascii_pos:ascii_pos + count * ncol],
                                  dtype=np.float64).reshape(count, ncol)
                ascii_pos += count * ncol
                arr = {pn: flat[:, i] for i, (pn, _) in enumerate(props)}
            if name == 'vertex':
                verts = np.stack([np.asarray(arr['x'], dtype=np.float32),
                                  np.asarray(arr['y'], dtype=np.float32),
                                  np.asarray(arr['z'], dtype=np.float32)], axis=1)
                have = {pn for pn, _ in props}
                for a, b in (('s', 't'), ('u', 'v'), ('texture_u', 'texture_v')):
                    if a in have and b in have:
                        uvs = np.stack([np.asarray(arr[a], dtype=np.float32),
                                        np.asarray(arr[b], dtype=np.float32)],
                                       axis=1)
                        break
            continue

        _, cnt_t, idx_t = next(t for _, t in props if isinstance(t, tuple))
        if fmt == 'ascii':
            for _ in range(count):
                n = int(ascii_tokens[ascii_pos]); ascii_pos += 1
                faces.append([int(v) for v in ascii_tokens[ascii_pos:ascii_pos + n]])
                ascii_pos += n
        else:
            cnt_dt = np.dtype('<' + _PLY_DTYPES[cnt_t])
            idx_dt = np.dtype('<' + _PLY_DTYPES[idx_t])
            rec = np.dtype([('n', cnt_dt), ('i', idx_dt, 3)])
            fast = False
            if count and rec.itemsize * count <= len(body) - offset:
                probe = np.frombuffer(body, dtype=rec, count=count, offset=offset)
                if np.all(probe['n'] == 3):
                    faces = probe['i'].astype(np.int64)
                    offset += rec.itemsize * count
                    fast = True
            if not fast:
                for _ in range(count):
                    n = int(np.frombuffer(body, dtype=cnt_dt, count=1,
                                          offset=offset)[0])
                    offset += cnt_dt.itemsize
                    idx = np.frombuffer(body, dtype=idx_dt, count=n, offset=offset)
                    offset += idx_dt.itemsize * n
                    faces.append(idx.astype(np.int64).tolist())

    if verts is None:
        raise ValueError(f'{path!r}: no vertex element found')

    if isinstance(faces, np.ndarray):
        tris = faces
    else:
        tris = []
        for idx in faces:
            if len(idx) == 3:
                tris.append(list(idx))
            elif len(idx) == 4:
                a, b, c, d = idx
                tris.append([a, b, c])
                tris.append([c, d, a])
            elif len(idx) > 4:
                a = idx[0]
                for k in range(1, len(idx) - 1):
                    tris.append([a, idx[k], idx[k + 1]])
        tris = np.asarray(tris, dtype=np.int64)

    tris = np.ascontiguousarray(tris, dtype=np.int64)
    if tris.size and (tris.min() < 0 or tris.max() >= len(verts)):
        raise ValueError(f'{path!r}: face index out of range '
                         f'[{tris.min()}, {tris.max()}] for {len(verts)} verts')
    return {'verts': np.ascontiguousarray(verts, dtype=np.float32),
            'faces': tris,
            'uvs': uvs}


def ply_axes_to_raw(verts):
    """Rotate released-PLY coordinates into the raw scan frame.

    The released PLYs were written by Blender in a Z-up frame, while the scan
    transform (``refined_scan_*``) and the legacy ``.abc.cache`` are in the
    capture rig's Y-up frame.  The mapping is::

        raw_x =  ply_x
        raw_y =  ply_z
        raw_z = -ply_y


    Skipping this step would leave the scan rotated relative to the transform,
    and since the face bounding box is applied in canonical space the clip
    would then cut the wrong region.
    """
    out = np.empty_like(verts)
    out[:, 0] = verts[:, 0]
    out[:, 1] = verts[:, 2]
    out[:, 2] = -verts[:, 1]
    return out


def load_scan(path, ply_axis_fix=True):
    """Load a scan from the released ``.ply`` or a legacy ``.abc.cache``.

    Returns the same dict shape either way: ``verts`` (N,3) float32, ``faces``
    (F,3) int64, plus ``uvs`` / ``landmarks`` when available.

    For ``.ply`` inputs the Blender Z-up convention is undone by default so the
    result is in the raw scan frame the transform expects -- see
    :func:`ply_axes_to_raw`.  Pass ``ply_axis_fix=False`` to read the file
    verbatim.
    """
    if path.lower().endswith('.ply'):
        d = load_ply(path)
        if ply_axis_fix:
            d['verts'] = np.ascontiguousarray(ply_axes_to_raw(d['verts']))
        d.setdefault('landmarks', None)
        return d
    return load_scan_cache(path)
