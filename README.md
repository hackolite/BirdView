# BirdView 🦅

Pipeline de conversion automatique d'une vue caméra en vue de haut (*Bird's Eye View* / BEV), **sans calibration manuelle** ni marquage de points au sol.

Deux modes de sortie :
- **heatmap** – carte de densité cumulée des personnes (rouge = zones les plus fréquentées)
- **tracking** – trajectoires individuelles colorées par ID de personne (ByteTrack)

---

## Comment ça marche ?

```
Caméra RGB
    │
    ├─► Depth Anything V2 (profondeur métrique)  ─┐
    │                                              ├─► Plan du sol (RANSAC + SVD)
    └─► SegFormer / ADE20K (segmentation floor)  ─┘
                                                       │
                                                       ▼
                                              Matrice K virtuelle
                                                       │
                                                       ▼
                             YOLO11 (détection personnes) ──► point pied → BEV
                                                       │
                                           ┌───────────┴──────────┐
                                           ▼                       ▼
                                      heatmap BEV          tracking BEV
```

1. **Auto-calibration** : la depth map + la segmentation du sol → nuage de points 3D → RANSAC → équation du plan sol → matrice K intrinsèque estimée depuis un FOV horizontal supposé (70° par défaut).
2. **Projection BEV** : pour chaque personne détectée, le point pied (bas de la bounding box) est projeté sur le plan sol par intersection rayon/plan → coordonnées X, Y en mètres.
3. **Recalibration périodique** : toutes les 300 frames (~10 s à 30 fps) pour s'adapter à un changement de scène tout en économisant les ressources.

---

## Installation

### En local (pip)

```bash
# Cloner le dépôt
git clone https://github.com/hackolite/BirdView.git
cd BirdView

# Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install -e .
```

Ou directement sans cloner :

```bash
pip install git+https://github.com/hackolite/BirdView.git
```

### Google Colab

Copiez et exécutez ce bloc au début de votre notebook :

```python
# Installation de BirdView et de ses dépendances
!pip install git+https://github.com/hackolite/BirdView.git

# (Optionnel) Installer une version spécifique de PyTorch avec CUDA
# !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Puis dans une cellule de code :

```python
import subprocess, sys, os

# Lancer le pipeline en mode headless (pas d'écran sur Colab)
# Remplacez "video_input.mp4" par votre fichier uploadé dans /content/

# Si installé via pip (entry point 'birdview') :
subprocess.run([
    "birdview",
    "-m", "heatmap",
    "-i", "/content/video_input.mp4",
    "-o", "/content/video_output.mp4",
    "--no-display",
], check=True)

# Ou directement avec Python (depuis le répertoire cloné) :
# subprocess.run([
#     sys.executable, "main.py",
#     "-m", "heatmap",
#     "-i", "/content/video_input.mp4",
#     "-o", "/content/video_output.mp4",
#     "--no-display",
# ], check=True)

# Afficher le résultat dans le notebook
from IPython.display import Video
Video("/content/video_output.mp4", embed=True)
```

> **Note GPU** : BirdView détecte automatiquement CUDA. Sur Colab, activez le runtime GPU (*Modifier → Paramètres du notebook → Accélérateur matériel → GPU*).

---

## Ligne de commande

### Syntaxe générale

```
python main.py -m <mode> -i <source> [-o <sortie>] [--no-display]
```

| Argument | Description |
|---|---|
| `-m`, `--mode` | `heatmap` (défaut) ou `tracking` |
| `-i`, `--input` | Chemin vidéo, image, ou index webcam (`0`, `1`…) |
| `-o`, `--output` | Fichier de sortie (vidéo `.mp4` ou image `.jpg`/`.png`) |
| `--no-display` | Désactive les fenêtres OpenCV (requis sur Colab / serveur) |

### Exemples

```bash
# Convertir une vidéo en heatmap BEV
python main.py -m heatmap -i video_input.mp4 -o video_output.mp4

# Convertir une vidéo avec tracking des individus
python main.py -m tracking -i video_input.mp4 -o video_output.mp4

# Traiter une image fixe
python main.py -m heatmap -i scene.jpg -o scene_bev.jpg

# Webcam en temps réel (mode heatmap, sans sauvegarde)
python main.py -m heatmap -i 0

# Webcam en temps réel avec tracking
python main.py -m tracking -i 0

# Mode headless (Google Colab / serveur sans écran)
python main.py -m heatmap -i video.mp4 -o output.mp4 --no-display

# Anciens arguments (compatibilité)
python main.py --video video.mp4
python main.py --image scene.jpg
python main.py --webcam 0
```

### Format de sortie

La vidéo de sortie est une vue **côte à côte** : image caméra annotée (gauche) + vue BEV (droite).

```
┌─────────────────────┬─────────────────────┐
│  Vue caméra         │  Vue BEV            │
│  (bounding boxes,   │  (heatmap ou        │
│   masque sol vert)  │   trajectoires)     │
└─────────────────────┴─────────────────────┘
```

---

## Configuration

Tous les hyperparamètres sont centralisés dans `utils/config.py` :

| Paramètre | Défaut | Description |
|---|---|---|
| `assumed_horizontal_fov_deg` | `70.0` | FOV horizontal supposé (degrés). À ajuster selon votre caméra. |
| `depth_model_id` | `Depth-Anything-V2-Metric-Indoor-Small-hf` | Modèle de profondeur (HuggingFace). |
| `segmentation_model_id` | `segformer-b2-finetuned-ade-512-512` | Modèle de segmentation (HuggingFace). |
| `yolo_model_id` | `yolo11n.pt` | Modèle YOLO (nano = rapide). |
| `ransac_iterations` | `500` | Nombre d'itérations RANSAC. |
| `bev_canvas_size_px` | `800` | Taille du canvas BEV (pixels). |
| `bev_meters_per_pixel` | `0.02` | Résolution de la vue du dessus (2 cm/pixel). |
| `heatmap_decay` | `0.98` | Facteur d'oubli exponentiel de la heatmap. |

---

## Structure du projet

```
BirdView/
├── main.py                      # Point d'entrée, CLI
├── setup.py                     # Package pip
├── requirements.txt             # Dépendances
│
├── geometry/
│   ├── bev_transform.py         # Homographie auto-calibrée image→monde
│   ├── ground_plane.py          # RANSAC + SVD pour estimer le plan sol
│   └── projection.py            # Primitives sténopé (K, rétroprojection, nuage)
│
├── models/
│   ├── camera_pose.py           # VirtualCamera : K + plan sol
│   ├── depth_estimator.py       # Depth Anything V2 (profondeur métrique)
│   ├── detector.py              # YOLO11 (détection + tracking ByteTrack)
│   └── segmentation.py          # SegFormer / ADE20K (floor_mask)
│
├── utils/
│   └── config.py                # Tous les hyperparamètres
│
└── visualization/
    └── display.py               # BevRenderer (heatmap) + TrackingRenderer (trails)
```

---

## Prérequis système

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CPU ou CUDA)
- Accès internet au premier lancement (téléchargement automatique des modèles HuggingFace et YOLO)

---

## Dépannage

**`RuntimeError: Moins de 2% de l'image est classée 'sol'`**
→ La caméra ne voit pas suffisamment le sol. Inclinez-la davantage vers le bas, ou réduisez `ransac_min_inlier_ratio` dans `config.py`.

**`RuntimeError: RANSAC n'a pas trouvé de plan sol cohérent`**
→ La scène est trop encombrée ou la depth map est bruitée. Augmentez `ransac_iterations` ou réduisez `ransac_distance_threshold_m`.

**Profondeur non métrique (fallback)**
→ Si le modèle `Metric-Indoor` n'est pas accessible, le pipeline bascule sur le modèle relatif avec une heuristique de recalage d'échelle. Les distances en mètres seront approximatives.

**Affichage noir sur Google Colab**
→ Ajoutez `--no-display` à la commande et sauvegardez le résultat avec `-o`.
