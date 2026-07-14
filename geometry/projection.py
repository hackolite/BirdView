"""
geometry/projection.py
=======================
Primitives de géométrie projective (modèle sténopé / pinhole).

Ce module ne connaît rien du "sol" ou du "BEV" : il fournit uniquement les
briques de base (rétro-projection, rayons caméra, construction du nuage de
points) réutilisées par ground_plane.py et bev_transform.py.

Convention : repère caméra standard vision par ordinateur
  X -> droite, Y -> bas, Z -> devant la caméra (profondeur).
"""

from __future__ import annotations
import numpy as np


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """
    Construit une matrice intrinsèque K à partir d'un champ de vision
    horizontal supposé (aucune calibration caméra disponible -> c'est le
    seul a priori géométrique du pipeline).

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    On suppose des pixels carrés (fx = fy) et le centre optique au centre
    de l'image, hypothèses standards pour une caméra "générique".
    """
    hfov_rad = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # pixels carrés
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def pixel_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """
    Direction du rayon caméra passant par le pixel (u, v), en repère caméra.
    Non normalisé à 1 (norme arbitraire) : renvoie (x, y, 1) dans le plan
    focal, ce qui suffit pour toute intersection rayon/plan ultérieure.
    """
    K_inv = np.linalg.inv(K)
    pixel_h = np.array([u, v, 1.0])
    ray = K_inv @ pixel_h
    return ray  # ray[2] == 1.0 par construction


def backproject(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """
    Rétro-projette un pixel (u, v) avec une profondeur métrique connue
    vers un point 3D en repère caméra.
    """
    ray = pixel_ray(u, v, K)
    return ray * depth  # (X, Y, Z) avec Z = depth


def build_pointcloud(depth_map: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None,
                      stride: int = 4, max_depth: float = 20.0) -> np.ndarray:
    """
    Construit un nuage de points 3D (repère caméra) à partir d'une depth map.

    Args:
        depth_map: (H, W) profondeur en mètres.
        K: matrice intrinsèque (3,3).
        mask: (H, W) booléen optionnel, ex. floor_mask -> ne garder que le sol.
        stride: sous-échantillonnage spatial (perf : pas besoin de tous les pixels).
        max_depth: filtre les profondeurs aberrantes (souvent bruitées en monoculaire).

    Returns:
        (N, 3) points 3D en repère caméra.
    """
    h, w = depth_map.shape[:2]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us = us.ravel()
    vs = vs.ravel()
    depths = depth_map[vs, us]

    valid = (depths > 0.05) & (depths < max_depth) & np.isfinite(depths)
    if mask is not None:
        valid &= mask[vs, us]

    us, vs, depths = us[valid], vs[valid], depths[valid]
    if len(us) == 0:
        return np.empty((0, 3), dtype=np.float64)

    K_inv = np.linalg.inv(K)
    pixels_h = np.stack([us, vs, np.ones_like(us)], axis=1).astype(np.float64)  # (N,3)
    rays = pixels_h @ K_inv.T  # (N,3), ray[:,2] == 1
    points = rays * depths[:, None]
    return points
