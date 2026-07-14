"""
models/segmentation.py
========================
Segmentation sémantique de la scène pour extraire floor_mask
(sol / route / trottoir / tarmac...), wall_mask, ceiling_mask.

Modèle : SegFormer fine-tuné sur ADE20K (150 classes génériques, adaptées
aussi bien à un magasin, un aéroport qu'un restaurant). ADE20K est un bon
compromis pour rester "agnostique au domaine" : ce n'est pas un modèle
spécialisé sol-de-supermarché ou sol-d'aéroport, il généralise.

SAM2 est mentionné dans le cahier des charges comme alternative : SAM2 fait
de la segmentation d'INSTANCE (il faut un prompt : point, boîte...), pas de
la segmentation SÉMANTIQUE par classe. Pour extraire "le sol" sans aucune
interaction utilisateur, un modèle de segmentation sémantique pré-entraîné
(SegFormer/ADE20K) est donc le choix adapté ici. SAM2 redevient pertinent
en aval (étape 2 de la roadmap) pour séparer proprement chaque personne en
foule dense, cf. discussion précédente.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.config import ADE20K_FLOOR_CLASSES, ADE20K_WALL_CLASSES, ADE20K_CEILING_CLASSES


class SceneSegmenter:
    def __init__(self, model_id: str, device: str = "cpu"):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H, W, 3) uint8 RGB.
        Returns:
            label_map: (H, W) int, classe ADE20K par pixel, à la résolution
                       ORIGINALE de l'image (upsamplé depuis la sortie modèle).
        """
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, num_classes, h', w') basse résolution

        upsampled = F.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        )
        label_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
        return label_map

    @staticmethod
    def floor_mask(label_map: np.ndarray) -> np.ndarray:
        """(H, W) booléen : pixels appartenant à une classe 'sol traversable'."""
        mask = np.zeros_like(label_map, dtype=bool)
        for class_id in ADE20K_FLOOR_CLASSES:
            mask |= (label_map == class_id)
        return mask

    @staticmethod
    def wall_mask(label_map: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(label_map, dtype=bool)
        for class_id in ADE20K_WALL_CLASSES:
            mask |= (label_map == class_id)
        return mask

    @staticmethod
    def ceiling_mask(label_map: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(label_map, dtype=bool)
        for class_id in ADE20K_CEILING_CLASSES:
            mask |= (label_map == class_id)
        return mask

    @staticmethod
    def obstacle_mask(label_map: np.ndarray) -> np.ndarray:
        """Tout ce qui n'est ni sol, ni mur, ni plafond -> objets/obstacles."""
        floor = SceneSegmenter.floor_mask(label_map)
        wall = SceneSegmenter.wall_mask(label_map)
        ceiling = SceneSegmenter.ceiling_mask(label_map)
        return ~(floor | wall | ceiling)
