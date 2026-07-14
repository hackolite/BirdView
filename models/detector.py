"""
models/detector.py
====================
Détection de personnes (YOLOv8 / YOLO11 via Ultralytics).

Pour chaque personne détectée, on extrait le "point pied" (bottom-center
de la bounding box) : c'est ce point, supposé au contact du sol, qui sera
projeté en BEV via geometry.bev_transform.image_to_bev().
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PersonDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        """Point sol : milieu bas de la bounding box, cf. cahier des charges."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1


class PersonDetector:
    def __init__(self, model_id: str, person_class_id: int = 0,
                 conf_threshold: float = 0.35, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(model_id)
        self.person_class_id = person_class_id
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> list[PersonDetection]:
        """
        Args:
            image: (H, W, 3) uint8 RGB.
        Returns:
            Liste de PersonDetection (uniquement la classe "person").
        """
        results = self.model.predict(
            image,
            classes=[self.person_class_id],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(PersonDetection(x1, y1, x2, y2, conf))

        return detections
