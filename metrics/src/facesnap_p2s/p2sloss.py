"""
Point-to-surface (P2S) distance.

Loads the distance class straight out of a local clone of TEMPEH (Bolkart
et al., CVPR 2023) https://github.com/TimoBolkart/TEMPEH.

This does NOT install or import TEMPEH as a package -- it loads the
file, ``utils/point_to_surface_loss.py``, directly off disk by path provided

TEMPEH is MPG-licensed, research-only, and not redistributed here -- you
clone it yourself and point this module at your clone:

    git clone https://github.com/TimoBolkart/TEMPEH.git /path/to/TEMPEH
    export TEMPEH_PATH=/path/to/TEMPEH        # or pass tempeh_path= explicitly

See README.md for the full walkthrough.

Note: TEMPEH's class always runs its GMO robustifier and returns *squared*
distances (there is no un-robustified, Euclidean "eval mode" in the
original). To get the plain Euclidean-distance metric this benchmark
reports, we construct it with ``gmo_sigma=0`` (robustifier becomes the
identity) and take the square root ourselves below.
"""

import importlib.util
import os

import torch
from torch import nn


def _load_tempeh_point_to_surface_loss(tempeh_path):
    file_path = os.path.join(tempeh_path, "utils", "point_to_surface_loss.py")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Could not find utils/point_to_surface_loss.py under {tempeh_path!r}. "
            "Clone TEMPEH (https://github.com/TimoBolkart/TEMPEH) and point "
            "TEMPEH_PATH (or tempeh_path=) at the clone's root directory."
        )
    spec = importlib.util.spec_from_file_location(
        "tempeh_point_to_surface_loss", file_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PointToSurfaceLoss


class PointToSurfaceLoss(nn.Module):
    """For each point, the distance to the closest point on a mesh surface,
    backed by TEMPEH's unmodified point-to-surface distance class.
    """

    def __init__(self, tempeh_path=None):
        super().__init__()
        tempeh_path = tempeh_path or os.environ.get("TEMPEH_PATH")
        if not tempeh_path:
            raise RuntimeError(
                "Set TEMPEH_PATH to your local TEMPEH clone, or pass "
                "tempeh_path= explicitly. See README.md."
            )
        tempeh_cls = _load_tempeh_point_to_surface_loss(tempeh_path)
        self._impl = tempeh_cls(gmo_sigma=0.0)  # disables the GMO robustifier

    def forward(self, points, mesh_vertices, mesh_faces):
        """
        Args:
            points: (batch_size, num_points, 3)
            mesh_vertices: (batch_size, num_vertices, 3)
            mesh_faces: (num_faces, 3)

        Returns:
            mean_distance: scalar mean distance over all points
            distances: (batch_size, num_points) distance per point
        """
        _, squared_distances = self._impl(
            points, mesh_vertices, mesh_faces, return_distances=True
        )
        distances = torch.sqrt(squared_distances)
        return distances.mean(), distances
