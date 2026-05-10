"""
Recadrage automatique pour Lightroom.

Deux strategies :
1. Face-centered : si des visages sont detectes, recadre pour les placer
   a la regle des tiers (tiers superieur).
2. Horizon straightening : detecte l'inclinaison via les lignes dominantes
   et corrige l'angle.

Les valeurs sont en coordonnees normalisees (0.0-1.0) compatibles XMP Lightroom :
  crs:CropTop, crs:CropLeft, crs:CropBottom, crs:CropRight, crs:CropAngle
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CropResult:
    """Resultat du recadrage automatique."""
    crop_top: float = 0.0       # 0.0-1.0
    crop_left: float = 0.0      # 0.0-1.0
    crop_bottom: float = 1.0    # 0.0-1.0
    crop_right: float = 1.0     # 0.0-1.0
    crop_angle: float = 0.0     # degres (rotation)
    has_crop: bool = False

    def to_xmp_attrs(self) -> dict:
        if not self.has_crop:
            return {}
        attrs = {
            "crs:CropTop": f"{self.crop_top:.6f}",
            "crs:CropLeft": f"{self.crop_left:.6f}",
            "crs:CropBottom": f"{self.crop_bottom:.6f}",
            "crs:CropRight": f"{self.crop_right:.6f}",
            "crs:CropAngle": f"{self.crop_angle:.4f}",
            "crs:HasCrop": "True",
            "crs:CropConstrainToWarp": "0",
        }
        return attrs


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def detect_horizon_angle(gray: np.ndarray, max_angle: float = 3.0) -> float:
    """
    Detecte l'inclinaison de l'horizon via les lignes dominantes (Hough).
    Retourne l'angle de correction en degres.
    Limite a +-max_angle pour eviter les faux positifs.
    """
    try:
        h, w = gray.shape[:2]

        # Resize pour performance
        scale = 600 / max(h, w) if max(h, w) > 600 else 1.0
        if scale < 1.0:
            small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            small = gray

        # Detection des bords
        edges = cv2.Canny(small, 50, 150, apertureSize=3)

        # Hough lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=int(min(small.shape) * 0.15),
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return 0.0

        # Calculer l'angle de chaque ligne quasi-horizontale
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Garder seulement les lignes quasi-horizontales (< 15 degres)
            if abs(angle) < 15:
                angles.append(angle)

        if not angles:
            return 0.0

        # Angle median (robuste aux outliers)
        median_angle = float(np.median(angles))

        # Limiter la correction
        return _clamp(median_angle, -max_angle, max_angle)

    except Exception:
        return 0.0


def compute_face_crop(
    img_bgr: np.ndarray,
    faces: list,
    target_ratio: Optional[float] = None,
    margin: float = 0.15,
) -> Optional[CropResult]:
    """
    Calcule un recadrage centre sur les visages detectes.

    - Place le centre des visages au tiers superieur (regle des tiers)
    - Garde tous les visages dans le cadre avec une marge
    - Conserve le ratio original (ou target_ratio si specifie)
    - Ne recadre pas plus de 25% de l'image

    faces: liste de tuples (x, y, w, h) en pixels.
    target_ratio: ratio largeur/hauteur souhaite (None = garder l'original).
    margin: marge autour des visages (fraction de la taille du visage).
    """
    if not faces or len(faces) == 0:
        return None

    h, w = img_bgr.shape[:2]

    if target_ratio is None:
        target_ratio = w / h

    # Bounding box englobant tous les visages
    face_list = list(faces)
    min_x = min(fx for fx, fy, fw, fh in face_list)
    min_y = min(fy for fx, fy, fw, fh in face_list)
    max_x = max(fx + fw for fx, fy, fw, fh in face_list)
    max_y = max(fy + fh for fx, fy, fw, fh in face_list)

    # Centre des visages
    face_cx = (min_x + max_x) / 2
    face_cy = (min_y + max_y) / 2

    # Taille de la zone des visages avec marge
    face_w = (max_x - min_x) * (1 + margin * 2)
    face_h = (max_y - min_y) * (1 + margin * 2)

    # La zone de crop doit etre au moins 75% de l'image originale
    min_crop_w = w * 0.75
    min_crop_h = h * 0.75

    # Taille de crop basee sur le ratio cible
    crop_w = max(face_w * 2.5, min_crop_w)  # au moins 2.5x la zone visages
    crop_h = crop_w / target_ratio

    if crop_h < max(face_h * 2.5, min_crop_h):
        crop_h = max(face_h * 2.5, min_crop_h)
        crop_w = crop_h * target_ratio

    # Ne pas depasser l'image originale
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    # Recalculer pour maintenir le ratio
    if crop_w / crop_h > target_ratio:
        crop_w = crop_h * target_ratio
    else:
        crop_h = crop_w / target_ratio

    # Positionner le crop : visages au tiers superieur
    # Le centre vertical des visages doit etre a 1/3 du haut du crop
    crop_cy = face_cy + crop_h * (0.5 - 0.36)  # decaler pour mettre les visages au tiers
    crop_cx = face_cx  # centre horizontal sur les visages

    # Calculer les bords
    crop_left = crop_cx - crop_w / 2
    crop_top = crop_cy - crop_h / 2
    crop_right = crop_cx + crop_w / 2
    crop_bottom = crop_cy + crop_h / 2

    # Ajuster si le crop depasse les bords de l'image
    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > w:
        crop_left -= (crop_right - w)
        crop_right = w
    if crop_bottom > h:
        crop_top -= (crop_bottom - h)
        crop_bottom = h

    # Clamp final
    crop_left = _clamp(crop_left, 0, w)
    crop_top = _clamp(crop_top, 0, h)
    crop_right = _clamp(crop_right, 0, w)
    crop_bottom = _clamp(crop_bottom, 0, h)

    # Verifier que le crop est significatif (retire au moins 3% quelque part)
    area_ratio = ((crop_right - crop_left) * (crop_bottom - crop_top)) / (w * h)
    if area_ratio > 0.97:
        # Crop negligeable — ne pas recadrer
        return None

    return CropResult(
        crop_top=crop_top / h,
        crop_left=crop_left / w,
        crop_bottom=crop_bottom / h,
        crop_right=crop_right / w,
        has_crop=True,
    )


def auto_crop(
    img_bgr: np.ndarray,
    faces: list,
    enable_face_crop: bool = True,
    enable_straighten: bool = True,
) -> CropResult:
    """
    Point d'entree principal. Combine le recadrage visages + redressement.
    """
    result = CropResult()

    # 1. Redressement horizon
    if enable_straighten:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        angle = detect_horizon_angle(gray)
        if abs(angle) > 0.3:  # Seulement si l'inclinaison est perceptible
            result.crop_angle = -angle  # Sens inverse pour corriger
            result.has_crop = True

    # 2. Recadrage sur visages
    if enable_face_crop and faces is not None and len(faces) > 0:
        face_crop = compute_face_crop(img_bgr, faces)
        if face_crop is not None:
            result.crop_top = face_crop.crop_top
            result.crop_left = face_crop.crop_left
            result.crop_bottom = face_crop.crop_bottom
            result.crop_right = face_crop.crop_right
            result.has_crop = True

    return result
