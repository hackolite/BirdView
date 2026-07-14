"""
visualization/display.py
==========================
Deux fenêtres de sortie :
  1. Image originale annotée (bounding boxes + masque du sol en surimpression).
  2. Vue BEV : selon le mode, soit heatmap d'occupation cumulée (BevRenderer),
     soit trajectoires individuelles colorées par ID (TrackingRenderer).
"""

from __future__ import annotations
from collections import defaultdict, deque
import numpy as np
import cv2


# Palette de couleurs (BGR) pour distinguer les tracks (20 couleurs cycliques).
_TRACK_PALETTE = [
    (255, 80,  80),  (80,  255, 80),  (80,  80,  255), (255, 255, 80),
    (255, 80,  255), (80,  255, 255), (200, 120, 60),  (60,  200, 120),
    (120, 60,  200), (200, 200, 60),  (200, 60,  200), (60,  200, 200),
    (255, 160, 100), (100, 255, 160), (160, 100, 255), (255, 200, 50),
    (50,  255, 200), (200, 50,  255), (180, 255, 100), (100, 180, 255),
]


def _track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def draw_detections_and_floor(
    image: np.ndarray,
    detections: list,
    floor_mask: np.ndarray | None = None,
    floor_color: tuple[int, int, int] = (0, 200, 0),
    floor_alpha: float = 0.3,
) -> np.ndarray:
    """Dessine les bounding boxes + le masque du sol en surimpression semi-transparente."""
    vis = image.copy()

    if floor_mask is not None:
        overlay = vis.copy()
        overlay[floor_mask] = floor_color
        vis = cv2.addWeighted(overlay, floor_alpha, vis, 1 - floor_alpha, 0)

    for det in detections:
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        color = _track_color(det.track_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        fx, fy = map(int, det.foot_point)
        cv2.circle(vis, (fx, fy), 5, (255, 0, 0), -1)
        label = f"#{det.track_id} {det.confidence:.2f}" if det.track_id is not None else f"{det.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


class BevRenderer:
    """
    Maintient l'état de la heatmap d'occupation entre les frames (mode vidéo).
    """

    def __init__(self, canvas_size: int = 800, meters_per_pixel: float = 0.02,
                 decay: float = 0.98, gaussian_sigma_px: int = 15):
        self.canvas_size = canvas_size
        self.meters_per_pixel = meters_per_pixel
        self.decay = decay
        self.gaussian_sigma_px = gaussian_sigma_px
        self.heatmap = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    def world_to_canvas(self, world_xy: np.ndarray) -> np.ndarray:
        """(N,2) coords monde (mètres) -> (N,2) pixels canvas (centré, Y inversé)."""
        center = self.canvas_size / 2.0
        px = world_xy[:, 0] / self.meters_per_pixel + center
        py = -world_xy[:, 1] / self.meters_per_pixel + center
        return np.stack([px, py], axis=1)

    def update_heatmap(self, world_points: np.ndarray):
        """Ajoute les positions courantes à la heatmap avec décroissance temporelle."""
        self.heatmap *= self.decay

        if len(world_points) == 0:
            return

        canvas_pts = self.world_to_canvas(world_points)
        impulse = np.zeros_like(self.heatmap)
        for x, y in canvas_pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.canvas_size and 0 <= yi < self.canvas_size:
                impulse[yi, xi] += 1.0

        impulse = cv2.GaussianBlur(
            impulse, (0, 0), sigmaX=self.gaussian_sigma_px, sigmaY=self.gaussian_sigma_px
        )
        self.heatmap += impulse

    def render(self, world_points: np.ndarray, grid_spacing_m: float = 1.0) -> np.ndarray:
        """Construit l'image finale : heatmap (colormap) + grille + points personnes."""
        normalized = self.heatmap / (self.heatmap.max() + 1e-6)
        heatmap_u8 = (normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        # Fond neutre là où la heatmap est ~vide, pour ne pas tout peindre en bleu
        canvas = np.full((self.canvas_size, self.canvas_size, 3), 30, dtype=np.uint8)
        mask = normalized > 0.02
        canvas[mask] = heatmap_color[mask]

        self._draw_grid(canvas, grid_spacing_m)

        if len(world_points) > 0:
            canvas_pts = self.world_to_canvas(world_points)
            for x, y in canvas_pts:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < self.canvas_size and 0 <= yi < self.canvas_size:
                    cv2.circle(canvas, (xi, yi), 6, (255, 255, 255), -1)
                    cv2.circle(canvas, (xi, yi), 6, (0, 0, 0), 1)

        cv2.putText(canvas, f"Personnes: {len(world_points)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return canvas

    def _draw_grid(self, canvas: np.ndarray, spacing_m: float):
        """Grille de repère (tous les `spacing_m` mètres) pour donner une échelle visuelle."""
        step_px = int(spacing_m / self.meters_per_pixel)
        if step_px <= 0:
            return
        for x in range(0, self.canvas_size, step_px):
            cv2.line(canvas, (x, 0), (x, self.canvas_size), (60, 60, 60), 1)
        for y in range(0, self.canvas_size, step_px):
            cv2.line(canvas, (0, y), (self.canvas_size, y), (60, 60, 60), 1)
        center = self.canvas_size // 2
        cv2.line(canvas, (center, 0), (center, self.canvas_size), (100, 100, 100), 1)
        cv2.line(canvas, (0, center), (self.canvas_size, center), (100, 100, 100), 1)


class TrackingRenderer:
    """
    Vue BEV en mode tracking : affiche les trajectoires individuelles de chaque
    personne avec une couleur unique par ID, et conserve un historique glissant
    des positions (trail).

    Contrairement à BevRenderer, pas de heatmap : on trace directement les
    chemins parcourus par chaque track.
    """

    def __init__(self, canvas_size: int = 800, meters_per_pixel: float = 0.02,
                 trail_length: int = 60):
        """
        Args:
            canvas_size:      Côté du canvas carré en pixels.
            meters_per_pixel: Résolution spatiale de la vue du dessus.
            trail_length:     Nombre de positions passées conservées par track.
        """
        self.canvas_size = canvas_size
        self.meters_per_pixel = meters_per_pixel
        self.trail_length = trail_length
        # dict track_id -> deque de positions canvas (xi, yi)
        self._trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))

    def world_to_canvas(self, wx: float, wy: float) -> tuple[int, int]:
        """Coordonnées monde (mètres) -> pixel canvas."""
        center = self.canvas_size / 2.0
        xi = int(round(wx / self.meters_per_pixel + center))
        yi = int(round(-wy / self.meters_per_pixel + center))
        return xi, yi

    def update(self, detections: list, world_points: np.ndarray):
        """
        Met à jour les trajectoires à partir des détections trackées.

        Args:
            detections:   Liste de PersonDetection (avec track_id).
            world_points: (N, 2) coordonnées monde correspondantes (même ordre).
        """
        for det, (wx, wy) in zip(detections, world_points):
            tid = det.track_id if det.track_id is not None else -1
            xi, yi = self.world_to_canvas(wx, wy)
            self._trails[tid].append((xi, yi))

    def render(self, detections: list, world_points: np.ndarray,
               grid_spacing_m: float = 1.0) -> np.ndarray:
        """Construit l'image BEV : grille + trails + points courants."""
        canvas = np.full((self.canvas_size, self.canvas_size, 3), 30, dtype=np.uint8)
        self._draw_grid(canvas, grid_spacing_m)

        # Dessine les trails (dégradé d'opacité vers le passé)
        for tid, trail in self._trails.items():
            color = _track_color(tid)
            pts = list(trail)
            n = len(pts)
            for i in range(1, n):
                alpha = i / n  # plus récent = plus opaque
                c = tuple(int(v * alpha) for v in color)
                xi0, yi0 = pts[i - 1]
                xi1, yi1 = pts[i]
                if (0 <= xi0 < self.canvas_size and 0 <= yi0 < self.canvas_size and
                        0 <= xi1 < self.canvas_size and 0 <= yi1 < self.canvas_size):
                    cv2.line(canvas, (xi0, yi0), (xi1, yi1), c, 2)

        # Dessine les positions courantes
        for det, (wx, wy) in zip(detections, world_points):
            tid = det.track_id if det.track_id is not None else -1
            color = _track_color(tid)
            xi, yi = self.world_to_canvas(wx, wy)
            if 0 <= xi < self.canvas_size and 0 <= yi < self.canvas_size:
                cv2.circle(canvas, (xi, yi), 7, color, -1)
                cv2.circle(canvas, (xi, yi), 7, (255, 255, 255), 1)
                if tid != -1:
                    cv2.putText(canvas, str(tid), (xi + 8, yi + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.putText(canvas, f"Personnes: {len(world_points)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return canvas

    def _draw_grid(self, canvas: np.ndarray, spacing_m: float):
        step_px = int(spacing_m / self.meters_per_pixel)
        if step_px <= 0:
            return
        for x in range(0, self.canvas_size, step_px):
            cv2.line(canvas, (x, 0), (x, self.canvas_size), (60, 60, 60), 1)
        for y in range(0, self.canvas_size, step_px):
            cv2.line(canvas, (0, y), (self.canvas_size, y), (60, 60, 60), 1)
        center = self.canvas_size // 2
        cv2.line(canvas, (center, 0), (center, self.canvas_size), (100, 100, 100), 1)
        cv2.line(canvas, (0, center), (self.canvas_size, center), (100, 100, 100), 1)
