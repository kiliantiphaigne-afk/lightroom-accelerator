"""
Recadrage automatique pour Lightroom.

Trois strategies combinees :
1. Redressement par lignes verticales (murs, colonnes, portes — ideal pour les events indoor)
2. Redressement par alignement des yeux (si visages detectes)
3. Face-centered crop : recadre pour placer les visages a la regle des tiers

Les valeurs sont en coordonnees normalisees (0.0-1.0) compatibles XMP Lightroom :
  crs:CropTop, crs:CropLeft, crs:CropBottom, crs:CropRight, crs:CropAngle
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class CropResult:
    """Resultat du recadrage automatique."""
    crop_top: float = 0.0       # 0.0-1.0
    crop_left: float = 0.0      # 0.0-1.0
    crop_bottom: float = 1.0    # 0.0-1.0
    crop_right: float = 1.0     # 0.0-1.0
    crop_angle: float = 0.0     # degres (rotation pour Lightroom)
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


# ---------------------------------------------------------------------------
# Detection d'inclinaison
# ---------------------------------------------------------------------------

def _detect_angle_from_lines(gray: np.ndarray, max_angle: float = 5.0) -> Optional[float]:
    """
    Detecte l'inclinaison via les lignes quasi-verticales ET quasi-horizontales.
    Les lignes verticales (murs, colonnes, portes) sont plus fiables en interieur.
    Retourne l'angle de tilt en degres, ou None si pas assez de lignes.
    """
    try:
        h, w = gray.shape[:2]

        # Resize pour performance
        target = 600
        scale = target / max(h, w) if max(h, w) > target else 1.0
        if scale < 1.0:
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = gray.copy()

        edges = cv2.Canny(small, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=int(min(small.shape) * 0.12),
            maxLineGap=12
        )

        if lines is None or len(lines) == 0:
            return None

        h_angles = []   # angles des lignes quasi-horizontales
        v_angles = []   # angles des lignes quasi-verticales

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue

            angle = np.degrees(np.arctan2(dy, dx))
            length = np.sqrt(dx * dx + dy * dy)

            # Lignes quasi-horizontales (±10°)
            if abs(angle) < 10:
                h_angles.append((angle, length))

            # Lignes quasi-verticales (80°-100° ou -80° a -100°)
            # L'angle par rapport a la verticale = angle - 90 (ou angle + 90)
            elif 80 < abs(angle) < 100:
                # Convertir en ecart par rapport a 90°
                v_tilt = angle - 90 if angle > 0 else angle + 90
                v_angles.append((v_tilt, length))

        # Privilegier les lignes verticales (plus fiables en interieur)
        # Ponderer par la longueur de chaque ligne
        if v_angles and sum(l for _, l in v_angles) > sum(l for _, l in h_angles) * 0.5:
            # Utiliser les verticales
            total_len = sum(l for _, l in v_angles)
            if total_len > 0:
                weighted = sum(a * l for a, l in v_angles) / total_len
                return _clamp(weighted, -max_angle, max_angle)

        if h_angles:
            total_len = sum(l for _, l in h_angles)
            if total_len > 0:
                weighted = sum(a * l for a, l in h_angles) / total_len
                return _clamp(weighted, -max_angle, max_angle)

        return None

    except Exception:
        return None


def _detect_angle_from_eyes(
    img_bgr: np.ndarray,
    face_rects: list,
) -> Optional[float]:
    """
    Detecte l'inclinaison a partir de l'alignement des yeux.
    Si les yeux ne sont pas horizontaux, la photo est penchee.
    Plus fiable que les lignes pour les portraits.
    """
    try:
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        angles = []

        for (fx, fy, fw, fh) in face_rects:
            # Adapter les coords au resize
            sfx = int(fx * scale)
            sfy = int(fy * scale)
            sfw = int(fw * scale)
            sfh = int(fh * scale)

            # Region du visage
            if sfy < 0 or sfx < 0 or sfy + sfh > gray.shape[0] or sfx + sfw > gray.shape[1]:
                continue

            roi = gray[sfy:sfy + sfh, sfx:sfx + sfw]
            if roi.size == 0:
                continue

            roi_eq = cv2.equalizeHist(roi)

            eyes = eye_cascade.detectMultiScale(
                roi_eq, scaleFactor=1.1, minNeighbors=5,
                minSize=(int(sfw * 0.1), int(sfh * 0.05)),
                maxSize=(int(sfw * 0.5), int(sfh * 0.4)),
            )

            if not isinstance(eyes, np.ndarray) or len(eyes) < 2:
                continue

            # Trier les yeux par position X pour avoir gauche/droite
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            # Prendre les 2 premiers (gauche et droite)
            e1 = eyes_sorted[0]
            e2 = eyes_sorted[1]

            # Centre de chaque oeil
            c1x = e1[0] + e1[2] / 2
            c1y = e1[1] + e1[3] / 2
            c2x = e2[0] + e2[2] / 2
            c2y = e2[1] + e2[3] / 2

            dx = c2x - c1x
            dy = c2y - c1y

            if abs(dx) < 5:
                continue

            eye_angle = np.degrees(np.arctan2(dy, dx))

            # Si l'angle est petit (< 8°), c'est un vrai tilt
            if abs(eye_angle) < 8:
                angles.append(eye_angle)

        if not angles:
            return None

        return float(np.median(angles))

    except Exception:
        return None


def detect_tilt(
    img_bgr: np.ndarray,
    face_rects: list,
    max_angle: float = 5.0,
) -> float:
    """
    Detection d'inclinaison combinee : yeux + lignes.
    Les yeux sont prioritaires (plus fiables pour les portraits).
    Retourne l'angle de tilt en degres.
    """
    # 1. Essayer d'abord avec les yeux (plus fiable pour events)
    eye_angle = _detect_angle_from_eyes(img_bgr, face_rects) if face_rects else None

    # 2. Detecter via les lignes structurelles
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    line_angle = _detect_angle_from_lines(gray, max_angle)

    # Decision
    if eye_angle is not None and line_angle is not None:
        # Les deux concordent ? Moyenne ponderee (yeux = 2x plus de poids)
        if abs(eye_angle - line_angle) < 3:
            angle = (eye_angle * 2 + line_angle) / 3
        else:
            # Divergence — faire confiance aux yeux
            angle = eye_angle
    elif eye_angle is not None:
        angle = eye_angle
    elif line_angle is not None:
        angle = line_angle
    else:
        return 0.0

    # Seuil minimum pour corriger (eviter les micro-corrections inutiles)
    if abs(angle) < 0.4:
        return 0.0

    return _clamp(angle, -max_angle, max_angle)


# ---------------------------------------------------------------------------
# Recadrage sur les visages
# ---------------------------------------------------------------------------

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
    crop_w = max(face_w * 2.5, min_crop_w)
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
    crop_cy = face_cy + crop_h * (0.5 - 0.36)
    crop_cx = face_cx

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

    # Verifier que le crop est significatif (retire au moins 3%)
    area_ratio = ((crop_right - crop_left) * (crop_bottom - crop_top)) / (w * h)
    if area_ratio > 0.97:
        return None

    return CropResult(
        crop_top=crop_top / h,
        crop_left=crop_left / w,
        crop_bottom=crop_bottom / h,
        crop_right=crop_right / w,
        has_crop=True,
    )


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def auto_crop(
    img_bgr: np.ndarray,
    faces: list,
    enable_face_crop: bool = True,
    enable_straighten: bool = True,
) -> CropResult:
    """
    Point d'entree principal. Combine redressement + recadrage visages.
    """
    result = CropResult()

    # 1. Redressement (lignes verticales + alignement yeux)
    if enable_straighten:
        tilt = detect_tilt(img_bgr, faces)
        if abs(tilt) > 0:
            # Lightroom CropAngle : positif = rotation anti-horaire
            # Si la photo penche a droite (tilt positif), on corrige avec un angle negatif
            result.crop_angle = -tilt
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
