"""Point-to-surface (P2S) evaluation for the FaceSnap 4D face benchmark.

The metric: sample the ground-truth scan surface uniformly by area, keep the
samples inside the subject's face bounding box, and measure each one's distance
to the method's registered mesh.  Reported in millimetres.

Layout::

    scan_io.py     readers -- scan (.ply / .abc.cache), mesh (.obj), transform
    transforms.py  the raw <-> aligned (canonical) coordinate map
    scan.py        BenchmarkScan: load, pose, area-sample, clip
    p2sloss.py     the point-to-surface distance itself (kaolin)
    rendering.py   the per-frame comparison figure (numpy + matplotlib)

Typical use is through ``evaluate.py`` 

    from facesnap_p2s import load_scan_and_sample, PointToSurfaceLoss

Importing this package pulls in torch and kaolin.  :mod:`facesnap_p2s.rendering`
and :mod:`facesnap_p2s.scan_io` are importable on their own without either.
"""

from .p2sloss import PointToSurfaceLoss
from .scan import BenchmarkScan, load_scan_and_sample
from .scan_io import (
    load_obj,
    load_ply,
    load_scan,
    load_scan_cache,
    load_scan_transform,
    ply_axes_to_raw,
    save_scan_transform,
)
from .transforms import ScanTransform, apply_rigid, euler_to_matrix, invert_rigid

__all__ = [
    'BenchmarkScan',
    'PointToSurfaceLoss',
    'ScanTransform',
    'apply_rigid',
    'euler_to_matrix',
    'invert_rigid',
    'load_obj',
    'load_ply',
    'load_scan',
    'load_scan_and_sample',
    'load_scan_cache',
    'load_scan_transform',
    'ply_axes_to_raw',
    'save_scan_transform',
]
