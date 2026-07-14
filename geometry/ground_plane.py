"""
geometry/ground_plane.py
=========================
Estimation automatique du plan du sol, SANS aucun point cliqué manuellement.

Principe :
1. La depth map (Depth Anything V2) + les intrinsèques virtuels donnent un
   nuage de points 3D en repère caméra (voir projection.py).
2. On restreint ce nuage aux pixels classés "sol" par la segmentation
   sémantique (floor_mask) -> on ne fit le plan que sur des points qui
   ont statistiquement de bonnes chances d'appartenir réellement au sol.
3. RANSAC : on tire aléatoirement des triplets de points, on calcule le
   plan qu'ils définissent, on compte les inliers (points à moins de
   `distance_threshold` du plan), on garde le meilleur plan.
4. Raffinement : régression totale des moindres carrés (SVD) sur les
   inliers du meilleur plan RANSAC -> plan final plus précis et moins
   sensible au bruit de la depth map.

Sortie : équation de plan (a, b, c, d) telle que  a*x + b*y + c*z + d = 0
en repère caméra, avec (a,b,c) la normale (non nécessairement unitaire
avant normalisation, mais on la normalise en sortie).
"""

from __future__ import annotations
import numpy as np


def _plane_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray | None:
    """Calcule (a,b,c,d) du plan passant par 3 points 3D. None si dégénéré."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None  # points colinéaires
    normal = normal / norm
    d = -np.dot(normal, p1)
    return np.array([normal[0], normal[1], normal[2], d])


def _point_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Distance signée des points au plan (plan supposé normalisé : |a,b,c|=1)."""
    a, b, c, d = plane
    return points @ np.array([a, b, c]) + d


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 500,
    distance_threshold: float = 0.05,
    min_inlier_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ajuste un plan 3D par RANSAC.

    Returns:
        plane: (4,) -> (a, b, c, d), normale normalisée, orientée vers la caméra
               (c < 0, cf. _orient_normal_towards_camera).
        inlier_mask: (N,) booléen, points utilisés dans le plan final.

    Raises:
        RuntimeError si aucun plan avec suffisamment d'inliers n'est trouvé
        (scène sans sol visible détectable -> il faut alerter l'utilisateur
        plutôt que de renvoyer un plan n'importe quoi).
    """
    n = points.shape[0]
    if n < 50:
        raise RuntimeError(
            f"Pas assez de points sol candidats ({n}) pour un fit RANSAC fiable. "
            "Vérifier floor_mask / la depth map."
        )

    rng = np.random.default_rng(seed)
    best_plane = None
    best_inliers = None
    best_count = 0

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        plane = _plane_from_three_points(points[idx[0]], points[idx[1]], points[idx[2]])
        if plane is None:
            continue
        dist = np.abs(_point_plane_distance(points, plane))
        inliers = dist < distance_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_plane = plane
            best_inliers = inliers

    if best_plane is None or best_count < min_inlier_ratio * n:
        raise RuntimeError(
            f"RANSAC n'a pas trouvé de plan sol cohérent "
            f"({best_count}/{n} inliers, seuil={min_inlier_ratio:.0%}). "
            "La scène est peut-être trop encombrée ou floor_mask est vide."
        )

    # --- Raffinement par moindres carrés totaux (SVD) sur les inliers ---
    refined_plane = _refine_plane_svd(points[best_inliers])
    dist = np.abs(_point_plane_distance(points, refined_plane))
    final_inliers = dist < distance_threshold

    refined_plane = _orient_normal_towards_camera(refined_plane)
    return refined_plane, final_inliers


def _refine_plane_svd(points: np.ndarray) -> np.ndarray:
    """
    Ajustement de plan par moindres carrés totaux : le centroïde est sur le
    plan, la normale est le vecteur singulier associé à la plus petite
    valeur singulière (direction de variance minimale du nuage de points).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]  # plus petite valeur singulière
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d])


def _orient_normal_towards_camera(plane: np.ndarray) -> np.ndarray:
    """
    Convention : on oriente la normale du plan sol pour qu'elle pointe
    "vers le haut" du point de vue caméra (composante Y négative en repère
    caméra standard où Y pointe vers le bas). Ça évite les ambiguïtés de
    signe en aval (calcul de hauteur caméra, alignement du repère monde).
    """
    a, b, c, d = plane
    normal = np.array([a, b, c])
    if normal[1] > 0:  # Y positif = pointe vers le bas -> on flip
        normal = -normal
        d = -d
    return np.array([normal[0], normal[1], normal[2], d])


def camera_height_above_plane(plane: np.ndarray) -> float:
    """
    Distance de l'origine caméra (0,0,0) au plan sol.
    Pour un plan normalisé (|a,b,c| = 1) : distance = |d|.
    """
    a, b, c, d = plane
    normal_norm = np.linalg.norm([a, b, c])
    return abs(d) / normal_norm
