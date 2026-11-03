"""One frame of ground-truth scan: load, pose, sample, clip.

:class:`BenchmarkScan` owns the three steps that turn a scan file into the
point set the metric scores:

1. load raw vertices and triangles (:mod:`.scan_io`),
2. pose them into canonical space (:mod:`.transforms`),
3. sample the surface uniformly by area, then clip to the face bounding box.

"""

import numpy as np
import torch

from .scan_io import load_scan
from .transforms import ScanTransform


class BenchmarkScan:
    """One frame of ground-truth scan, with its raw->aligned transform."""

    def __init__(self, cache_path, transform, device='cuda'):
        """``transform`` is a :class:`ScanTransform` (or None to stay in raw)."""
        self.device = device
        self.path = cache_path

        dic = load_scan(cache_path)
        self.raw_verts = torch.as_tensor(dic['verts'], dtype=torch.float32,
                                         device=device)
        self.faces = torch.as_tensor(dic['faces'], dtype=torch.int64,
                                     device=device)
        self.landmarks = (
            torch.as_tensor(dic['landmarks'], dtype=torch.float32, device=device)
            if dic['landmarks'] is not None else None
        )
        self.transform = transform

    # -- frames ------------------------------------------------------------

    @property
    def aligned_verts(self):
        """Scan vertices in aligned (canonical) space.

        ``R @ (scale * v_raw) + t`` -- see ``transforms.ScanTransform.to_aligned``
        
        """
        if self.transform is None:
            return self.raw_verts
        return self.transform.to_aligned(self.raw_verts)

    # -- sampling ----------------------------------------------------------

    def sample_uniform(self, n_samples, bbox=None, seed=None):
        """Uniformly sample the *aligned* surface, then clip to ``bbox``.

        Sampling is area-weighted over the whole scan and only then clipped, so
        the surviving points are uniform over the face region.  ``bbox`` is
        ``[x_min, x_max, y_min, y_max, z_min, z_max]`` in aligned space.

        Returns (points_aligned (M, 3) tensor, kept_mask (n_samples,) tensor).
        """
        from kaolin.ops.mesh import sample_points

        if seed is not None:
            torch.manual_seed(seed)

        # Transform to canonical space FIRST, then sample, then clip.  The
        # bbox is expressed in this canonical frame, so the clip is only valid
        # on aligned coordinates -- clipping raw coordinates with the same
        # numbers would cut an arbitrary slab out of a mis-posed head.
        verts = self.aligned_verts
        pts, _ = sample_points(verts.unsqueeze(0), self.faces, n_samples)
        pts = pts[0]

        if bbox is None:
            return pts, torch.ones(len(pts), dtype=torch.bool, device=pts.device)

        mask = self.bbox_mask(pts, bbox)
        return pts[mask], mask

    @staticmethod
    def bbox_mask(points, bbox):
        """Strict-inequality AABB mask.

        ``points`` must already be in canonical/aligned space.  ``bbox`` is
        ``[x_min, x_max, y_min, y_max, z_min, z_max]``, hand-calibrated per
        subject -- this is the face region, and everything outside it (cap,
        shoulders, background) is dropped before scoring. 
        """
        x0, x1, y0, y1, z0, z1 = (float(v) for v in bbox)
        return (
            (points[:, 0] > x0) & (points[:, 0] < x1) &
            (points[:, 1] > y0) & (points[:, 1] < y1) &
            (points[:, 2] > z0) & (points[:, 2] < z1)
        )

    # -- convenience -------------------------------------------------------

    def numpy_aligned(self):
        """Aligned vertices + faces as numpy, for the renderer."""
        return (
            self.aligned_verts.detach().cpu().numpy().astype(np.float32),
            self.faces.detach().cpu().numpy().astype(np.int64),
        )

    def __repr__(self):
        return (f"BenchmarkScan({len(self.raw_verts)} verts, "
                f"{len(self.faces)} tris)")


def load_scan_and_sample(cache_path, transform_path, sample_points, clip_bbox,
                         device='cuda', seed=None):
    """Load one frame, pose it, sample it, and return it in both frames.

    The metric wants both raw coordinates and the renderer wants aligned ones :
      raw_verts / raw_faces           -- raw space, numpy
      scan_pts_raw                    -- sampled+clipped points, raw space
      scan_verts_aligned              -- scan vertices, aligned space
      scan_pts_aligned                -- sampled+clipped points, aligned space
      transform                       -- the ScanTransform
    """
    from .scan_io import load_scan_transform

    scale, rot, trans = load_scan_transform(transform_path)
    xf = ScanTransform(scale, rot, trans, device=device)

    scan = BenchmarkScan(cache_path, xf, device=device)
    pts_aligned, _ = scan.sample_uniform(sample_points, bbox=clip_bbox, seed=seed)
    pts_raw = xf.to_raw(pts_aligned)

    verts_aligned, faces_np = scan.numpy_aligned()

    return {
        'raw_verts': scan.raw_verts.detach().cpu().numpy().astype(np.float32),
        'raw_faces': faces_np,
        'scan_pts_raw': pts_raw.detach().cpu().numpy().astype(np.float32),
        'scan_verts_aligned': verts_aligned,
        'scan_pts_aligned': pts_aligned.detach().cpu().numpy().astype(np.float32),
        'transform': xf,
        'scan': scan,
    }
