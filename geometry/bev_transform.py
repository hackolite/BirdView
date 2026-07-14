"""
geometry/bev_transform.py
==========================
Cœur géométrique du projet : transforme l'image caméra en vue "Bird's Eye
View" SANS jamais demander 4 points à l'utilisateur.

Idée clé
--------
Une fois le plan du sol connu (ground_plane.py), la relation entre les
pixels de l'image et les coordonnées monde (X, Y) sur le sol est, par
construction géométrique, une HOMOGRAPHIE (car le sol est un plan).
On peut donc calculer cette homographie *analytiquement* à partir de :
  - la matrice intrinsèque virtuelle K,
  - l'équation du plan sol en repère caméra,
sans jamais faire cliquer 4 points à l'utilisateur.

C'est l'équivalent "auto-calibré" de cv2.getPerspectiveTransform().

Étapes de la dérivation (voir docstring de `_ground_frame_rotation`) :
1. On calcule une rotation R_align qui aligne la normale du plan sol avec
   l'axe Z du "monde" -> on obtient un repère où le sol est horizontal.
2. On en déduit la pose caméra->monde (rotation + translation).
3. L'homographie image -> monde s'écrit alors H = K @ [r1 r2 t] (colonnes
   1, 2 de la rotation + translation), inversée pour aller image->monde.
"""

from __future__ import annotations
import numpy as np
import cv2

from .projection import pixel_ray


def _skew(v: np.ndarray) -> np.ndarray:
    """Matrice antisymétrique associée au produit vectoriel par v."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Renvoie la rotation R (3x3) minimale telle que R @ a = b (a, b unitaires).
    Utilise la formule de Rodrigues sous forme fermée (via le produit
    vectoriel), robuste sauf quand a et b sont exactement opposés.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)  # déjà alignés
        # a et b opposés : rotation de 180° autour d'un axe orthogonal à a
        orthogonal = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, orthogonal)
        axis = axis / np.linalg.norm(axis)
        return cv2.Rodrigues(axis * np.pi)[0]

    vx = _skew(v)
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def _ground_frame_rotation(plane_cam: np.ndarray) -> tuple[np.ndarray, float]:
    """
    À partir du plan sol (a,b,c,d) en repère caméra, calcule :
      - R_align (3x3) : rotation telle que R_align @ normal_cam = [0,0,1].
        Appliquer R_align à un point du repère caméra donne ses coordonnées
        dans un repère "monde tourné" où le sol est le plan Z = z_ground.
      - z_ground : altitude du plan sol dans ce repère tourné.

    Repère caméra utilisé : X droite, Y bas, Z devant (profondeur), donc la
    normale du sol (orientée vers la caméra, cf. ground_plane.py) a une
    composante Y négative typiquement -> on l'aligne avec +Z du monde.
    """
    a, b, c, d = plane_cam
    normal = np.array([a, b, c])
    normal_unit = normal / np.linalg.norm(normal)

    R_align = _rotation_aligning_vectors(normal_unit, np.array([0.0, 0.0, 1.0]))

    # Le plan en repère caméra : normal_unit . X_cam + d/|normal| = 0
    # Après rotation : [0,0,1] . X_rot + d/|normal| = 0  =>  z_ground = -d/|normal|
    z_ground = -d / np.linalg.norm(normal)
    return R_align, z_ground


def compute_homography_image_to_world(K: np.ndarray, plane_cam: np.ndarray) -> np.ndarray:
    """
    Calcule l'homographie H (3x3) qui envoie un pixel homogène (u, v, 1)
    vers des coordonnées monde homogènes (X, Y, 1) sur le plan du sol.

    Dérivation :
        R_align, z_ground = _ground_frame_rotation(plane_cam)
        # Rotation monde -> caméra : R_wc = R_align^T
        # Translation monde -> caméra (origine monde posée sur le sol,
        # directement sous la caméra dans le repère tourné) :
        #   t_wc = R_align^T @ [0, 0, z_ground]
        # Homographie monde -> image (points du plan Z_world=0) :
        #   H_w2i = K @ [r1 r2 t_wc]   (r1, r2 = 2 premières colonnes de R_wc)
        # Homographie image -> monde : H = inv(H_w2i)
    """
    R_align, z_ground = _ground_frame_rotation(plane_cam)
    R_wc = R_align.T
    t_wc = R_align.T @ np.array([0.0, 0.0, z_ground])

    r1 = R_wc[:, 0]
    r2 = R_wc[:, 1]
    extrinsics_planar = np.column_stack([r1, r2, t_wc])  # (3,3)

    H_world_to_image = K @ extrinsics_planar
    H_image_to_world = np.linalg.inv(H_world_to_image)
    return H_image_to_world


def image_point_to_bev(u: float, v: float, H_image_to_world: np.ndarray) -> tuple[float, float]:
    """Applique l'homographie à un pixel unique -> coordonnées monde (X, Y) en mètres."""
    p = np.array([u, v, 1.0])
    world = H_image_to_world @ p
    world = world / world[2]
    return float(world[0]), float(world[1])


def image_points_to_bev(points_uv: np.ndarray, H_image_to_world: np.ndarray) -> np.ndarray:
    """
    Version vectorisée : (N,2) pixels -> (N,2) coordonnées monde (mètres).
    Utilise cv2.perspectiveTransform pour l'application matricielle
    (opération géométrique standard OpenCV).
    """
    pts = points_uv.reshape(-1, 1, 2).astype(np.float64)
    world = cv2.perspectiveTransform(pts, H_image_to_world)
    return world.reshape(-1, 2)


def image_to_bev(u: float, v: float, K: np.ndarray, plane_cam: np.ndarray) -> tuple[float, float]:
    """
    Fonction de haut niveau demandée par le cahier des charges :
    pixel (u, v) -> coordonnées monde (X, Y) sur le plan du sol,
    par intersection rayon caméra / plan (plus robuste que de relire un
    seul pixel de la depth map, qui peut être bruité localement).
    """
    ray = pixel_ray(u, v, K)  # direction (x, y, 1) en repère caméra
    a, b, c, d = plane_cam
    denom = a * ray[0] + b * ray[1] + c * ray[2]
    if abs(denom) < 1e-8:
        raise ValueError("Le rayon est parallèle au plan sol (pixel probablement au-dessus de l'horizon).")
    t = -d / denom
    if t <= 0:
        raise ValueError("Intersection derrière la caméra : pixel invalide pour une projection sol.")
    point_cam = ray * t
    R_align, z_ground = _ground_frame_rotation(plane_cam)
    point_rot = R_align @ point_cam
    return float(point_rot[0]), float(point_rot[1])


def warp_image_to_bev(
    image: np.ndarray,
    K: np.ndarray,
    plane_cam: np.ndarray,
    canvas_size: int = 800,
    meters_per_pixel: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warpe l'image entière vers une vue du dessus (pour la fenêtre de
    visualisation). Combine l'homographie analytique (image -> monde en
    mètres) avec une similarité monde -> pixels du canvas (centrage +
    mise à l'échelle), puis applique cv2.warpPerspective.
    """
    H_image_to_world = compute_homography_image_to_world(K, plane_cam)

    # Similarité monde (mètres) -> pixels canvas : origine au centre du
    # canvas, axe Y inversé pour un affichage "vu du dessus" intuitif.
    center = canvas_size / 2.0
    S = np.array([
        [1.0 / meters_per_pixel, 0.0, center],
        [0.0, -1.0 / meters_per_pixel, center],
        [0.0, 0.0, 1.0],
    ])

    H_image_to_canvas = S @ H_image_to_world
    bev = cv2.warpPerspective(image, H_image_to_canvas, (canvas_size, canvas_size))
    return bev, H_image_to_canvas
