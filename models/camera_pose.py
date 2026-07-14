"""
models/camera_pose.py
=======================
Construit une "caméra virtuelle" complète : intrinsèques (K) + plan du sol
en repère caméra. C'est l'objet central que consomment bev_transform.py et
main.py pour projeter n'importe quel pixel vers le monde.

Aucune calibration réelle n'est disponible -> K est dérivée d'un FOV
supposé (utils.config.assumed_horizontal_fov_deg), et le plan du sol est
estimé automatiquement (depth + segmentation + RANSAC).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from geometry.projection import intrinsics_from_fov, build_pointcloud
from geometry.ground_plane import fit_plane_ransac, camera_height_above_plane


@dataclass
class VirtualCamera:
    K: np.ndarray            # (3,3) intrinsèques
    plane_cam: np.ndarray    # (4,) équation du plan sol en repère caméra
    height_m: float          # hauteur caméra au-dessus du sol (mètres)
    inlier_ratio: float      # qualité du fit (diagnostic)


def estimate_virtual_camera(
    image_shape: tuple[int, int],
    depth_map: np.ndarray,
    floor_mask: np.ndarray,
    hfov_deg: float,
    ransac_iterations: int,
    ransac_distance_threshold: float,
    ransac_min_inlier_ratio: float,
    pointcloud_stride: int,
) -> VirtualCamera:
    """
    Pipeline complet d'auto-calibration :
      1. K depuis le FOV supposé.
      2. Nuage de points 3D restreint au sol segmenté.
      3. Plan sol par RANSAC + raffinement SVD.

    Raises:
        RuntimeError si le plan sol n'est pas trouvable (propagé depuis
        ground_plane.fit_plane_ransac) : mieux vaut échouer explicitement
        que produire une BEV silencieusement fausse.
    """
    h, w = image_shape[:2]
    K = intrinsics_from_fov(w, h, hfov_deg)

    floor_points = build_pointcloud(
        depth_map, K, mask=floor_mask, stride=pointcloud_stride,
    )

    plane, inliers = fit_plane_ransac(
        floor_points,
        iterations=ransac_iterations,
        distance_threshold=ransac_distance_threshold,
        min_inlier_ratio=ransac_min_inlier_ratio,
    )

    height = camera_height_above_plane(plane)
    inlier_ratio = float(inliers.sum()) / max(len(floor_points), 1)

    return VirtualCamera(K=K, plane_cam=plane, height_m=height, inlier_ratio=inlier_ratio)
