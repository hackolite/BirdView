"""
main.py
========
Pipeline complet : IMAGE RGB -> profondeur + segmentation -> plan du sol
-> BEV -> détection personnes -> tracking simple + heatmap.

Usage :
    # Mode heatmap (défaut) - convertit une vidéo en vue BEV avec heatmap
    python main.py -m heatmap -i video_input.mp4 -o video_output.mp4

    # Mode tracking - affiche les trajectoires individuelles en BEV
    python main.py -m tracking -i video_input.mp4 -o video_output.mp4

    # Sans sortie fichier (affichage fenêtre seulement)
    python main.py -m heatmap -i video_input.mp4

    # Image fixe
    python main.py -m heatmap -i image.jpg

    # Webcam (index 0)
    python main.py -m heatmap -i 0

    # Mode sans affichage (Google Colab / serveur headless)
    python main.py -m heatmap -i video.mp4 -o output.mp4 --no-display

    # Compatibilité avec les anciens arguments
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
import os
import sys
import numpy as np
import cv2

from utils.config import Config
from models.depth_estimator import DepthEstimator
from models.segmentation import SceneSegmenter
from models.camera_pose import estimate_virtual_camera, VirtualCamera
from models.detector import PersonDetector
from geometry.bev_transform import image_to_bev
from visualization.display import (
    draw_detections_and_floor, BevRenderer, TrackingRenderer,
)

RECALIBRATE_EVERY_N_FRAMES = 300  # ~10s à 30fps : le sol ne bouge pas, pas besoin de refaire tourner depth+seg à chaque frame

# Détecte automatiquement un environnement headless (Google Colab, serveur sans écran)
_IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


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
        self._last_floor_mask: np.ndarray | None = None

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

    def process_frame(self, image: np.ndarray,
                      mode: str = "heatmap") -> tuple[list, np.ndarray]:
        """
        Étapes 5-6 : détection (ou tracking) + projection BEV.

        Args:
            image: (H, W, 3) uint8 RGB.
            mode:  "heatmap" -> model.predict() sans ID persistant.
                   "tracking" -> model.track() avec ID ByteTrack persistant.

        Returns:
            (detections, world_points) où world_points est (N, 2) en mètres.
        """
        if mode == "tracking":
            detections = self.detector.track(image)
        else:
            detections = self.detector.detect(image)

        world_points = []
        valid_detections = []
        for det in detections:
            u, v = det.foot_point
            try:
                x, y = image_to_bev(u, v, self.virtual_camera.K, self.virtual_camera.plane_cam)
                world_points.append((x, y))
                valid_detections.append(det)
            except ValueError:
                continue  # point au-dessus de l'horizon / projection invalide

        world_points = np.array(world_points) if world_points else np.empty((0, 2))
        return valid_detections, world_points


def _make_combined_frame(annotated: np.ndarray, bev: np.ndarray,
                         target_h: int) -> np.ndarray:
    """Assemble deux images côte à côte à la même hauteur."""
    def _resize_h(img, h):
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        return cv2.resize(img, (w, h))

    left = _resize_h(annotated, target_h)
    right = _resize_h(bev, target_h)
    return np.hstack([left, right])


def _open_video_writer(output_path: str, fps: float,
                       width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[Avertissement] Impossible d'ouvrir le fichier de sortie '{output_path}'.")
    return writer


def run_on_image(pipeline: BevPipeline, image_path: str, config: Config,
                 mode: str = "heatmap", output_path: str | None = None,
                 no_display: bool = False):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Erreur : impossible de lire l'image '{image_path}'")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pipeline.calibrate(image_rgb)
    detections, world_points = pipeline.process_frame(image_rgb, mode=mode)

    annotated = draw_detections_and_floor(image_bgr, detections, pipeline._last_floor_mask)

    if mode == "tracking":
        renderer = TrackingRenderer(
            config.bev_canvas_size_px, config.bev_meters_per_pixel,
        )
        if len(world_points) > 0:
            renderer.update(detections, world_points)
        bev_image = renderer.render(detections, world_points)
    else:
        bev_renderer = BevRenderer(
            config.bev_canvas_size_px, config.bev_meters_per_pixel,
            config.heatmap_decay, config.heatmap_gaussian_sigma_px,
        )
        bev_renderer.update_heatmap(world_points)
        bev_image = bev_renderer.render(world_points)

    if output_path:
        combined = _make_combined_frame(annotated, bev_image, config.bev_canvas_size_px)
        cv2.imwrite(output_path, combined)
        print(f"[Pipeline] Image sauvegardée -> {output_path}")

    if not no_display and not _IN_COLAB:
        cv2.imshow("Camera + detections + sol segmente", annotated)
        cv2.imshow("Vue BEV + heatmap occupation", bev_image)
        print("Appuyez sur une touche pour fermer.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(pipeline: BevPipeline, source, config: Config,
                 mode: str = "heatmap", output_path: str | None = None,
                 no_display: bool = False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir la source vidéo '{source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    canvas_size = config.bev_canvas_size_px

    writer: cv2.VideoWriter | None = None
    if output_path:
        # Sortie : image caméra annotée (redimensionnée) + BEV côte à côte
        writer = _open_video_writer(output_path, fps, canvas_size * 2, canvas_size)

    if mode == "tracking":
        renderer = TrackingRenderer(canvas_size, config.bev_meters_per_pixel)
    else:
        renderer = BevRenderer(
            canvas_size, config.bev_meters_per_pixel,
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
                print(f"[Pipeline] Calibration échouée sur cette frame ({e}), "
                      "on continue avec la précédente si disponible.")
                if pipeline.virtual_camera is None:
                    frame_idx += 1
                    continue

        detections, world_points = pipeline.process_frame(frame_rgb, mode=mode)

        if mode == "tracking":
            if len(world_points) > 0:
                renderer.update(detections, world_points)
            bev_image = renderer.render(detections, world_points)
        else:
            renderer.update_heatmap(world_points)
            bev_image = renderer.render(world_points)

        annotated = draw_detections_and_floor(frame_bgr, detections, pipeline._last_floor_mask)

        density = len(world_points) / max((config.bev_canvas_size_px * config.bev_meters_per_pixel) ** 2, 1e-6)
        cv2.putText(annotated, f"Personnes: {len(world_points)} | Densite: {density:.2f} pers/m2",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer is not None:
            combined = _make_combined_frame(annotated, bev_image, canvas_size)
            writer.write(combined)

        if not no_display and not _IN_COLAB:
            cv2.imshow("Camera + detections + sol segmente", annotated)
            cv2.imshow("Vue BEV + heatmap occupation", bev_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Pipeline] Vidéo sauvegardée -> {output_path}")
    if not no_display and not _IN_COLAB:
        cv2.destroyAllWindows()


def _parse_input(value: str):
    """
    Interprète la valeur de -i/--input :
      - entier -> index webcam
      - fichier image (extension jpg/png/bmp/tiff) -> mode image
      - autre -> chemin vidéo
    """
    try:
        return int(value), "webcam"
    except ValueError:
        ext = os.path.splitext(value)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
            return value, "image"
        return value, "video"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BirdView - Pipeline BEV auto-calibré.\n"
            "Convertit une vue caméra en vue de haut (Bird's Eye View) "
            "avec heatmap d'occupation ou tracking de personnes.\n\n"
            "Exemples :\n"
            "  python main.py -m heatmap  -i video.mp4 -o sortie.mp4\n"
            "  python main.py -m tracking -i video.mp4 -o sortie.mp4\n"
            "  python main.py -m heatmap  -i 0                          # webcam\n"
            "  python main.py -m heatmap  -i image.jpg -o sortie.jpg\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Entrée ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input", type=str, metavar="SOURCE",
        help="Source d'entrée : chemin vidéo/image, ou index webcam (ex: 0).",
    )
    # Anciens arguments conservés pour la compatibilité
    input_group.add_argument("--image", type=str, help=argparse.SUPPRESS)
    input_group.add_argument("--video", type=str, help=argparse.SUPPRESS)
    input_group.add_argument("--webcam", type=int, help=argparse.SUPPRESS)

    # --- Mode ---
    parser.add_argument(
        "-m", "--mode", choices=["heatmap", "tracking"], default="heatmap",
        help="Mode de visualisation BEV : 'heatmap' (défaut) ou 'tracking'.",
    )

    # --- Sortie ---
    parser.add_argument(
        "-o", "--output", type=str, default=None, metavar="DEST",
        help="Chemin de sortie pour sauvegarder la vidéo/image résultat (optionnel).",
    )

    # --- Affichage ---
    parser.add_argument(
        "--no-display", action="store_true",
        help="Désactive l'affichage des fenêtres OpenCV (utile sur serveur ou Google Colab).",
    )

    args = parser.parse_args()

    config = Config()
    pipeline = BevPipeline(config)
    mode = args.mode
    no_display = args.no_display

    # Routing selon la source d'entrée
    if args.input is not None:
        source, kind = _parse_input(args.input)
        if kind == "image":
            run_on_image(pipeline, source, config, mode=mode,
                         output_path=args.output, no_display=no_display)
        else:  # webcam ou vidéo
            run_on_video(pipeline, source, config, mode=mode,
                         output_path=args.output, no_display=no_display)
    elif args.image:
        run_on_image(pipeline, args.image, config, mode=mode,
                     output_path=args.output, no_display=no_display)
    elif args.video:
        run_on_video(pipeline, args.video, config, mode=mode,
                     output_path=args.output, no_display=no_display)
    elif args.webcam is not None:
        run_on_video(pipeline, args.webcam, config, mode=mode,
                     output_path=args.output, no_display=no_display)


if __name__ == "__main__":
    main()
