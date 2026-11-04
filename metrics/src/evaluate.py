"""Point-to-surface (P2S) evaluation of one registered mesh against one scan.

Needs a local clone of TEMPEH (https://github.com/TimoBolkart/TEMPEH) --
see README.md -- pointed at via --tempeh-path or the TEMPEH_PATH env var.

Run with no arguments (plus --tempeh-path) to evaluate the bundled demo
frame (Subject8870559, Frame044172)::

    python evaluate.py --tempeh-path /path/to/TEMPEH --out ./demo_out

Writes ``scores.json`` and a comparison figure ``<Frame>.png``.  The bundled
demo mesh is a copy of the ground-truth scan itself (see demo_data/README.md)
-- it's a smoke test for the pipeline, not a method evaluation, so the
near-zero error it reports is expected and not meaningful.

To evaluate your own frame, give it the four things the metric needs -- a scan,
that scan's transform to go to a known canonical space , your mesh, and your mesh's unit scale::

    python evaluate.py \
        --scan       <frame>_scan.ply \
        --transform  <frame>_scan_transform.json \
        --mesh       <your_method>/<frame>.obj \
        --mesh-scale 0.01860523201775983 \
        --clip-bbox  -22 15 -17 6.5 4 13 \
        --out        ./my_results

``--mesh-scale`` is the divisor that takes your mesh into the scan's raw units
(``raw = mesh_verts / scale``); ``--clip-bbox`` is the subject's face region in
canonical space.  Both are per-subject constants published with the benchmark.
See README.md for how the two coordinate frames fit together.
"""

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from facesnap_p2s import PointToSurfaceLoss, load_obj, load_scan_and_sample  # noqa: E402
from facesnap_p2s.rendering import create_multi_method_figure                # noqa: E402

# --- bundled demo: Subject8870559 / ROM_1 / Frame044172 ---------------------
# Everything below ships in demo_data. See demo_data/README.md for what each file is.
_HERE = osp.dirname(osp.abspath(__file__))
_DEMO = osp.join(_HERE, 'demo_data')

DEMO_SCAN = osp.join(_DEMO, 'Frame044172_scan.ply')
DEMO_TRANSFORM = osp.join(_DEMO, 'Frame044172_scan_transform.json')
DEMO_MESH = osp.join(_DEMO, 'Frame044172_duplicate.obj')

DEMO_MESH_SCALE = 1.0   # dummy mesh is a copy of the raw scan -- already in raw units
DEMO_CLIP_BBOX = [-22, 15, -17, 6.5, 4, 13]   # Subject8870559's face region

# Note: the dummy mesh is a copy of the ground-truth scan, so the error it
# reports is near-zero by construction. It's here to exercise the pipeline
# end to end, not to represent any method's accuracy.


def frame_name(scan_path):
    """``FrameNNNNNN`` from a scan filename, whatever the suffix.

    Handles ``Frame044172.ply``, ``Frame044172_scan.ply`` and
    ``Frame044172.abc.cache`` alike.  The name keys the self-check and names
    the output figure, so a stray suffix must not change it.
    """
    stem = osp.basename(scan_path).split('.')[0]
    for suffix in ('_scan', '_gt', '_mesh'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def compute_p2s_error(scan_data, mesh_v_raw, mesh_faces, device, color_mode, tempeh_path):
    """P2S error for one mesh against one sampled scan.

    ``mesh_v_raw`` is in raw scan space, already divided by the method's unit
    scale.  Both operands are then multiplied by the scan's ``scale``, which
    keeps the pair consistent and puts the distances in the scan's millimetres.

    ``color_mode`` picks the direction:
      ``scan_points``  scan samples -> mesh surface  (the published metric)
      ``vertex``       mesh vertices -> scan surface (diagnostic)

    Returns ``(mesh_verts_aligned, err)`` -- the mesh in canonical space for
    the figure, and the per-point error in mm.
    """
    xf = scan_data['transform']
    scale = xf.scale

    mv_t = torch.as_tensor(mesh_v_raw, dtype=torch.float32,
                           device=device).unsqueeze(0)
    mesh_verts_aligned = (
        xf.to_aligned(mv_t)[0].detach().cpu().numpy().astype(np.float32)
    )

    p2s = PointToSurfaceLoss(tempeh_path=tempeh_path)
    mf_t = torch.as_tensor(mesh_faces, dtype=torch.int64, device=device)
    mv_scaled = mv_t * scale

    if color_mode == 'scan_points':
        pts_t = torch.as_tensor(scan_data['scan_pts_raw'], dtype=torch.float32,
                                device=device).unsqueeze(0) * scale
        _, err = p2s(pts_t, mv_scaled, mf_t)
    else:
        rv_t = torch.as_tensor(scan_data['raw_verts'], dtype=torch.float32,
                               device=device).unsqueeze(0) * scale
        _, err = p2s(mv_scaled, rv_t, mf_t)

    err = err.squeeze(0).detach().cpu().numpy()
    print(f'  P2S (mode={color_mode}): min={err.min():.4f} max={err.max():.4f} '
          f'mean={err.mean():.4f} median={float(np.median(err)):.4f}')
    return mesh_verts_aligned, err


def main(args):
    os.makedirs(args.out, exist_ok=True)

    print(f'Scan        : {args.scan}')
    print(f'Transform   : {args.transform}')
    print(f'Mesh        : {args.mesh}  (scale {args.mesh_scale})')
    print(f'Clip bbox   : {args.clip_bbox}')
    print(f'TEMPEH path : {args.tempeh_path}')

    print('\nLoading scan and sampling...')
    scan_data = load_scan_and_sample(
        args.scan, args.transform, args.sample_points,
        args.clip_bbox, device=args.device, seed=args.seed,
    )
    print(f'  {scan_data["transform"]}')
    print(f'  scan      : {len(scan_data["raw_verts"])} verts, '
          f'{len(scan_data["raw_faces"])} tris')
    print(f'  sampled   : {len(scan_data["scan_pts_aligned"])} points kept '
          f'of {args.sample_points} after clip')
    if len(scan_data['scan_pts_aligned']) == 0:
        raise SystemExit('No points survived the clip -- check --clip-bbox.')

    print(f'\nLoading mesh {osp.basename(args.mesh)}...')
    md = load_obj(args.mesh)
    mesh_v = (md['verts'] / np.float32(args.mesh_scale)).astype(np.float32)
    mesh_f = md['tris']
    print(f'  mesh      : {len(mesh_v)} verts, {len(mesh_f)} tris '
          f'(after quad split)')

    mesh_verts_aligned, err = compute_p2s_error(
        scan_data, mesh_v, mesh_f, args.device, args.color_mode, args.tempeh_path)

    scores = {
        'frame': frame_name(args.scan),
        'method': args.method_name,
        'color_mode': args.color_mode,
        'sample_points': args.sample_points,
        'points_scored': int(err.size),
        'mesh_scale': args.mesh_scale,
        'clip_bbox': list(args.clip_bbox),
        'mean': float(err.mean()),
        'median': float(np.median(err)),
        'std': float(err.std()),
    }
    scores_path = osp.join(args.out, 'scores.json')
    with open(scores_path, 'w') as fh:
        json.dump(scores, fh, indent=2)
    print(f'\nWrote {scores_path}')

    if not args.no_figure:
        fig_path = osp.join(args.out, f'{scores["frame"]}.png')
        print('Rendering figure...')
        create_multi_method_figure(
            scan_data,
            [{'mesh_verts_aligned': mesh_verts_aligned,
              'mesh_faces': mesh_f,
              'err': err}],
            [args.method_name], args, fig_path, method_groups=None,
        )
        print(f'Wrote {fig_path}')

    print('\nDone.')


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('inputs (default to the bundled demo frame)')
    g.add_argument('--scan', default=DEMO_SCAN,
                   help='ground-truth scan: released .ply, or legacy .abc.cache')
    g.add_argument('--transform', default=DEMO_TRANSFORM,
                   help="that scan's raw -> canonical transform (.json)")
    g.add_argument('--mesh', default=DEMO_MESH,
                   help='the registered mesh to score (.obj)')
    g.add_argument('--mesh-scale', type=float, default=DEMO_MESH_SCALE,
                   help="the method's unit divisor: raw = mesh_verts / scale")
    g.add_argument('--clip-bbox', type=float, nargs=6, default=DEMO_CLIP_BBOX,
                   metavar=('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'),
                   help="the subject's face region, in canonical space")
    g.add_argument('--method-name', default='Dummy',
                   help='label for the figure and scores.json')

    g = p.add_argument_group('metric')
    g.add_argument('--sample-points', type=int, default=300000,
                   help='surface samples drawn before clipping (default: '
                        '300000, the published setting)')
    g.add_argument('--color-mode', choices=['scan_points', 'vertex'],
                   default='scan_points',
                   help='scan_points: scan -> mesh (the published metric); '
                        'vertex: mesh -> scan (diagnostic)')
    g.add_argument('--seed', type=int, default=0,
                   help='torch seed. NB: it does not fully pin the result -- '
                        "kaolin's area sampling draws faces with "
                        'torch.multinomial, which is non-deterministic on CUDA')
    g.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    g.add_argument('--tempeh-path', default=os.environ.get('TEMPEH_PATH'),
                   help='path to a local clone of '
                        'https://github.com/TimoBolkart/TEMPEH (or set '
                        'TEMPEH_PATH)')

    g = p.add_argument_group('output')
    g.add_argument('--out', default='./demo_out',
                   help='output directory for scores.json and the figure')
    g.add_argument('--no-figure', action='store_true',
                   help='compute scores only, skip rendering')
    g.add_argument('--err-min', '--err_min', dest='err_min', type=float,
                   default=0.0, help='colour-scale floor, mm')
    g.add_argument('--err-max', '--err_max', dest='err_max', type=float,
                   default=0.25, help='colour-scale ceiling, mm')
    g.add_argument('--output-dpi', '--output_dpi', dest='output_dpi', type=int,
                   default=100)

    # Figure knobs the renderer reads off args; rarely worth changing.
    p.add_argument('--color-floor', '--color_floor', dest='color_floor',
                   type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument('--region-padding', '--region_padding',
                   dest='region_padding', type=float, default=0.0,
                   help=argparse.SUPPRESS)
    p.add_argument('--point-size', '--point_size', dest='point_size',
                   type=float, default=2.0, help=argparse.SUPPRESS)
    p.add_argument('--point-alpha', '--point_alpha', dest='point_alpha',
                   type=float, default=0.9, help=argparse.SUPPRESS)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    if not args.tempeh_path:
        build_parser().error('--tempeh-path is required (or set TEMPEH_PATH)')
    main(args)
