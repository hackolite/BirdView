"""
config.py
=========
Configuration centrale du pipeline BEV auto-calibré.

Toutes les constantes "magiques" du projet vivent ici pour éviter
qu'elles ne soient éparpillées dans le code (bonne pratique open source).
"""

from dataclasses import dataclass, field
import torch


# ---------------------------------------------------------------------------
# Classes ADE20K liées au sol (utilisées par la segmentation sémantique).
# ADE20K est un dataset généraliste (intérieur ET extérieur), donc on couvre
# plusieurs labels compatibles avec "sol traversable" : floor, road,
# sidewalk, earth, rug, path, runway (utile en aéroport / tarmac).
# Index = classe telle que retournée par les checkpoints HuggingFace
# "segformer-*-ade-*" (mapping ADE20K standard, 150 classes, 0-indexed).
# ---------------------------------------------------------------------------
ADE20K_FLOOR_CLASSES = {
    3: "floor",
    6: "road",
    11: "sidewalk",
    13: "earth/ground",
    28: "rug",
    52: "path",
    54: "runway",
}

ADE20K_WALL_CLASSES = {0: "wall"}
ADE20K_CEILING_CLASSES = {5: "ceiling"}


@dataclass
class Config:
    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- Modèle de profondeur monoculaire ---
    # Variante "métrique" (indoor) : donne une profondeur en mètres directement,
    # ce qui est essentiel puisqu'on n'a AUCUNE calibration externe pour lever
    # l'ambiguïté d'échelle d'un modèle de profondeur relative.
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    # Fallback si le modèle métrique n'est pas disponible : profondeur relative
    # (l'échelle absolue sera alors approximative, recalée heuristiquement).
    depth_model_fallback_id: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # --- Modèle de segmentation sémantique (pour extraire floor_mask) ---
    segmentation_model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512"

    # --- Détecteur de personnes ---
    yolo_model_id: str = "yolo11n.pt"  # nano : rapide, suffisant pour la classe "person"
    person_class_id: int = 0  # classe COCO "person"
    detection_conf_threshold: float = 0.35

    # --- Caméra virtuelle (aucune calibration réelle disponible) ---
    # Hypothèse de champ de vision horizontal. C'est le seul a priori du
    # pipeline : un FOV "raisonnable" de caméra de vidéosurveillance.
    # Peut être affiné plus tard par une estimation de points de fuite.
    assumed_horizontal_fov_deg: float = 70.0

    # --- RANSAC plan du sol ---
    ransac_iterations: int = 500
    ransac_distance_threshold_m: float = 0.05  # tolérance au plan (mètres)
    ransac_min_inlier_ratio: float = 0.3
    depth_pointcloud_stride: int = 4  # sous-échantillonnage de la depth map pour la vitesse

    # --- Vue BEV ---
    bev_canvas_size_px: int = 800
    bev_meters_per_pixel: float = 0.02  # résolution de la vue du dessus (2cm/pixel)
    bev_margin_m: float = 1.0

    # --- Heatmap d'occupation ---
    heatmap_decay: float = 0.98  # facteur d'oubli exponentiel par frame
    heatmap_gaussian_sigma_px: int = 15
