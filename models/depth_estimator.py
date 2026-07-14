"""
models/depth_estimator.py
==========================
Estimation de profondeur monoculaire via Depth Anything V2.

On utilise la variante "métrique" (entraînée sur des scènes intérieures,
NYU-Depth-V2 / Hypersim) car c'est LA pièce qui nous permet de nous passer
de toute calibration : elle donne directement une profondeur en mètres,
ce qui lève l'ambiguïté d'échelle inhérente à un modèle de profondeur
relative (indispensable pour que le plan RANSAC et la hauteur caméra
estimée ci-après aient un sens physique).

Le modèle est téléchargé automatiquement depuis le Hub HuggingFace lors du
premier appel (nécessite un accès réseau à huggingface.co).
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import torch


class DepthEstimator:
    def __init__(self, model_id: str, fallback_model_id: str | None = None, device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self.is_metric = "Metric" in model_id
        self._pipe = None

        try:
            self._load(model_id)
        except Exception as e:
            if fallback_model_id is None:
                raise
            print(f"[DepthEstimator] Échec du chargement de '{model_id}' ({e}). "
                  f"Repli sur '{fallback_model_id}' (profondeur relative, non métrique).")
            self.model_id = fallback_model_id
            self.is_metric = False
            self._load(fallback_model_id)

    def _load(self, model_id: str):
        # Import différé : transformers/torch peuvent être lourds à charger,
        # on ne paie ce coût qu'à l'instanciation réelle de la classe.
        from transformers import pipeline
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device == "cuda" else -1,
        )

    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H, W, 3) uint8 RGB.
        Returns:
            depth_map: (H, W) float32.
                Si `self.is_metric` est True -> profondeur en mètres.
                Sinon -> profondeur relative (à recaler heuristiquement,
                cf. `calibrate_scale_from_person_height`).
        """
        pil_image = Image.fromarray(image)
        result = self._pipe(pil_image)
        depth = result["predicted_depth"]

        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().cpu().numpy()

        # Le pipeline HF peut renvoyer une résolution différente de l'image
        # d'entrée -> on resize pour rester aligné pixel à pixel avec le RGB.
        if depth.shape != image.shape[:2]:
            depth_img = Image.fromarray(depth.astype(np.float32))
            depth_img = depth_img.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
            depth = np.array(depth_img)

        return depth.astype(np.float32)

    def calibrate_scale_from_person_height(
        self, depth_map: np.ndarray, person_bbox_heights_px: list[float],
        assumed_real_height_m: float = 1.70,
    ) -> np.ndarray:
        """
        Filet de sécurité si le modèle métrique n'a pas pu être chargé
        (`self.is_metric == False`) : recale grossièrement l'échelle de la
        depth map relative en utilisant la taille humaine moyenne comme
        référence statistique (principe d'auto-calibration par pose humaine,
        cf. littérature "Single View Physical Distance Estimation").

        C'est une approximation : à n'utiliser qu'en dernier recours.
        """
        if self.is_metric or not person_bbox_heights_px:
            return depth_map
        # Heuristique simple : le facteur d'échelle est approximé comme
        # constant sur l'image (raisonnable pour un sol globalement plan
        # et une caméra pas trop grand-angle).
        median_px_height = np.median(person_bbox_heights_px)
        # Cette relation exacte dépend de la focale ; ici on applique un
        # facteur correctif global déduit empiriquement à défaut de mieux.
        scale_factor = assumed_real_height_m / (median_px_height / depth_map.shape[0])
        return depth_map * scale_factor * 0.1  # facteur de repli conservateur
