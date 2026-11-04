"""The two coordinate frames the benchmark works in, and the map between them.

Frames
------
raw      : the scan's native capture coordinate system.  Meshes from other methods
           may need only a per-method unit divisor to
           land here, which is what ``--mesh-scale`` is.
aligned  : raw scaled, then rigidly posed into the canonical frontal frame::

               v_aligned = R @ (scale * v_raw) + trans

           This is the frame the face bounding box is expressed in, and the
           frame the renderer draws.

The rotation is 3 Euler angles in radians, composed ``Rz @ Ry @ Rx``.  Op order
and dtype (float32 throughout) are fixed deliberately -- changing either moves
the last digits of the reported error.
"""

import torch


def euler_to_matrix(rot_euler):
    """Euler angles -> rotation matrix.

    ``rot_euler``: (B, 3) radians.  Returns (B, 3, 3), composed
    ``Rz @ Ry @ Rx``.
    """
    assert rot_euler.dim() == 2 and rot_euler.shape[-1] == 3
    B = rot_euler.shape[0]
    dev, dt = rot_euler.device, rot_euler.dtype

    cx, sx = torch.cos(rot_euler[:, 0]), torch.sin(rot_euler[:, 0])
    cy, sy = torch.cos(rot_euler[:, 1]), torch.sin(rot_euler[:, 1])
    cz, sz = torch.cos(rot_euler[:, 2]), torch.sin(rot_euler[:, 2])

    one = torch.ones(B, device=dev, dtype=dt)
    zero = torch.zeros(B, device=dev, dtype=dt)

    Rx = torch.stack([one, zero, zero,
                      zero, cx, -sx,
                      zero, sx, cx], dim=-1).view(B, 3, 3)
    Ry = torch.stack([cy, zero, sy,
                      zero, one, zero,
                      -sy, zero, cy], dim=-1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, zero,
                      sz, cz, zero,
                      zero, zero, one], dim=-1).view(B, 3, 3)

    return torch.matmul(Rz, torch.matmul(Ry, Rx))


def apply_rigid(verts, rot, trans):
    """``v' = R @ v + t``.

    ``verts``: (B, N, 3); ``rot``: (B, 3) euler or (B, 3, 3) matrix;
    ``trans``: (B, 3).  Returns (B, N, 3).

    Expressed as a row-vector matmul rather than the homogeneous [3, 4] form,
    which keeps it a single fused ``baddbmm``.
    """
    assert verts.dim() == 3 and verts.shape[-1] == 3
    R = euler_to_matrix(rot) if rot.dim() == 2 else rot
    return torch.baddbmm(trans.unsqueeze(1), verts, R.transpose(1, 2))


def invert_rigid(rot, trans):
    """Inverse of (R, t): returns (R^T, -R^T t)."""
    R = euler_to_matrix(rot) if rot.dim() == 2 else rot
    R_inv = R.transpose(1, 2)
    t_inv = -torch.matmul(R_inv, trans.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv


class ScanTransform:
    """The per-frame scan transform: one scale, one rotation, one translation.

    Holds the map taking the scan from raw to aligned (canonical) space and
    applies it in either direction.  Loaded from the
    ``*_scan_transform.json``.
    """

    def __init__(self, scale, rot_euler, trans, device='cuda'):
        self.device = device
        self.scale = torch.as_tensor(scale, dtype=torch.float32,
                                     device=device).reshape(1)
        self.rot = torch.as_tensor(rot_euler, dtype=torch.float32,
                                   device=device).reshape(1, 3)
        self.trans = torch.as_tensor(trans, dtype=torch.float32,
                                     device=device).reshape(1, 3)

    def to_aligned(self, verts):
    
        v = verts.unsqueeze(0) if verts.dim() == 2 else verts
        out = apply_rigid(v * self.scale, self.rot, self.trans)
        return out.squeeze(0) if verts.dim() == 2 else out

    def to_raw(self, verts):
        """aligned -> raw.  ``verts``: (N, 3) or (1, N, 3)."""
        v = verts.unsqueeze(0) if verts.dim() == 2 else verts
        R_inv, t_inv = invert_rigid(self.rot, self.trans)
        out = apply_rigid(v, R_inv, t_inv) / self.scale
        return out.squeeze(0) if verts.dim() == 2 else out

    def __repr__(self):
        return (f"ScanTransform(scale={self.scale.item():.8f}, "
                f"rot={self.rot.flatten().tolist()}, "
                f"trans={self.trans.flatten().tolist()})")
