"""
main.py
========
Pipeline complet : IMAGE RGB -> profondeur + segmentation -> plan du sol
-> BEV -> détection personnes -> tracking simple + heatmap.

Usage :
    python main.py --image path/vers/image.jpg
    python main.py --video path/vers/video.mp4
    python main.py --webcam 0

Design :
    L'auto-calibration (profondeur -> segmentation -> plan du sol -> K)
    est coûteuse (modèles lourds) et n'a besoin d'être refaite que
    ponctuellement : en mode vidéo, on la recalcule seulement toutes les
    `RECALIBRATE_EVERY_N_FRAMES` frames (la caméra étant fixe, le plan sol
    ne change pas d'une frame à l'autre).
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import cv2

from utils.config import Config
from models.depth_estimator import DepthEstimator
from models.segmentation import SceneSegmenter
from models.camera_pose import estimate_virtual_camera, VirtualCamera
from models.detector import PersonDetector
from geometry.bev_transform import image_to_bev
from visualization.display import draw_detections_and_floor, BevRenderer

RECALIBRATE_EVERY_N_FRAMES = 300  # ~10s à 30fps : le sol ne bouge pas, pas besoin de refaire tourner depth+seg à chaque frame


class BevPipeline:
    """Encapsule les 3 modèles + l'état de calibration caméra courant."""

    def __init__(self, config: Config):
        self.config = config
        print(f"[Pipeline] Device: {config.device}")

        print("[Pipeline] Chargement du modèle de profondeur (Depth Anything V2)...")
        self.depth_estimator = DepthEstimator(
            config.depth_model_id, config.depth_model_fallback_id, config.device
        )

        print("[Pipeline] Chargement du modèle de segmentation sémantique...")
        self.segmenter = SceneSegmenter(config.segmentation_model_id, config.device)

        print("[Pipeline] Chargement du détecteur de personnes (YOLO)...")
        self.detector = PersonDetector(
            config.yolo_model_id, config.person_class_id,
            config.detection_conf_threshold, config.device,
        )

        self.virtual_camera: VirtualCamera | None = None

    def calibrate(self, image: np.ndarray) -> VirtualCamera:
        """Étapes 1 à 3 du pipeline : profondeur -> segmentation -> plan du sol."""
        depth_map = self.depth_estimator.estimate(image)
        label_map = self.segmenter.segment(image)
        floor_mask = SceneSegmenter.floor_mask(label_map)

        floor_pixel_ratio = floor_mask.mean()
        print(f"[Pipeline] Sol détecté sur {floor_pixel_ratio:.1%} des pixels.")
        if floor_pixel_ratio < 0.02:
            raise RuntimeError(
                "Moins de 2% de l'image est classée 'sol' : la scène est peut-être "
                "trop encombrée, ou la caméra ne voit quasiment pas le sol. "
                "Vérifier l'angle de la caméra."
            )

        vcam = estimate_virtual_camera(
            image_shape=image.shape,
            depth_map=depth_map,
            floor_mask=floor_mask,
            hfov_deg=self.config.assumed_horizontal_fov_deg,
            ransac_iterations=self.config.ransac_iterations,
            ransac_distance_threshold=self.config.ransac_distance_threshold_m,
            ransac_min_inlier_ratio=self.config.ransac_min_inlier_ratio,
            pointcloud_stride=self.config.depth_pointcloud_stride,
        )
        print(f"[Pipeline] Calibration OK -> hauteur caméra estimée: {vcam.height_m:.2f}m "
              f"(inliers RANSAC: {vcam.inlier_ratio:.1%})")

        self.virtual_camera = vcam
        self._last_floor_mask = floor_mask
        return vcam

    def process_frame(self, image: np.ndarray) -> tuple[list, np.ndarray]:
        """Étapes 5-6 : détection + projection BEV (réutilise la calibration en cours)."""
        detections = self.detector.detect(image)

        world_points = []
        for det in detections:
            u, v = det.foot_point
            try:
                x, y = image_to_bev(u, v, self.virtual_camera.K, self.virtual_camera.plane_cam)
                world_points.append((x, y))
            except ValueError:
                continue  # point au-dessus de l'horizon / projection invalide

        world_points = np.array(world_points) if world_points else np.empty((0, 2))
        return detections, world_points


def run_on_image(pipeline: BevPipeline, image_path: str, config: Config):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Erreur : impossible de lire l'image '{image_path}'")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pipeline.calibrate(image_rgb)
    detections, world_points = pipeline.process_frame(image_rgb)

    annotated = draw_detections_and_floor(image_bgr, detections, pipeline._last_floor_mask)

    bev_renderer = BevRenderer(
        config.bev_canvas_size_px, config.bev_meters_per_pixel,
        config.heatmap_decay, config.heatmap_gaussian_sigma_px,
    )
    bev_renderer.update_heatmap(world_points)
    bev_image = bev_renderer.render(world_points)

    cv2.imshow("Camera + detections + sol segmente", annotated)
    cv2.imshow("Vue BEV + heatmap occupation", bev_image)
    print("Appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(pipeline: BevPipeline, source, config: Config):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir la source vidéo '{source}'")
        sys.exit(1)

    bev_renderer = BevRenderer(
        config.bev_canvas_size_px, config.bev_meters_per_pixel,
        config.heatmap_decay, config.heatmap_gaussian_sigma_px,
    )

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if pipeline.virtual_camera is None or frame_idx % RECALIBRATE_EVERY_N_FRAMES == 0:
            try:
                pipeline.calibrate(frame_rgb)
            except RuntimeError as e:
                print(f"[Pipeline] Calibration échouée sur cette frame ({e}), on continue avec la précédente si disponible.")
                if pipeline.virtual_camera is None:
                    frame_idx += 1
                    continue

        detections, world_points = pipeline.process_frame(frame_rgb)
        bev_renderer.update_heatmap(world_points)

        annotated = draw_detections_and_floor(frame_bgr, detections, pipeline._last_floor_mask)
        bev_image = bev_renderer.render(world_points)

        density = len(world_points) / max((config.bev_canvas_size_px * config.bev_meters_per_pixel) ** 2, 1e-6)
        cv2.putText(annotated, f"Personnes: {len(world_points)} | Densite: {density:.2f} pers/m2",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Camera + detections + sol segmente", annotated)
        cv2.imshow("Vue BEV + heatmap occupation", bev_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Pipeline BEV auto-calibré (sans homographie manuelle).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Chemin vers une image fixe.")
    group.add_argument("--video", type=str, help="Chemin vers un fichier vidéo.")
    group.add_argument("--webcam", type=int, help="Index de la webcam (ex: 0).")
    args = parser.parse_args()

    config = Config()
    pipeline = BevPipeline(config)

    if args.image:
        run_on_image(pipeline, args.image, config)
    elif args.video:
        run_on_video(pipeline, args.video, config)
    elif args.webcam is not None:
        run_on_video(pipeline, args.webcam, config)


if __name__ == "__main__":
    main()
