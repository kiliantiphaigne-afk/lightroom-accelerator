"""
Analyse individuelle de chaque photo.
Calcule : blur score, exposition, visages, contexte, score global.
"""

import cv2
import numpy as np
import rawpy
import exifread
import io
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw"}
JPEG_EXTENSIONS = {".jpg", ".jpeg", ".tiff", ".tif", ".png"}

# Chargement du detecteur de visages OpenCV (Haar cascade, rapide)
_face_cascade = None
_eye_cascade = None

def _get_cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return _face_cascade, _eye_cascade


@dataclass
class PhotoAnalysis:
    path: Path
    blur_score: float = 0.0          # Variance Laplacien — plus haut = plus net
    mean_brightness: float = 128.0   # 0-255
    overexposed_ratio: float = 0.0   # Fraction de pixels brules
    underexposed_ratio: float = 0.0  # Fraction de pixels noirs
    face_count: int = 0
    face_rects: list = field(default_factory=list)  # [(x,y,w,h), ...] pour le crop
    open_eyes: bool = False
    context: str = "unknown"         # outdoor_day | indoor_flash | low_light | backlit | mixed
    score: float = 0.0               # 0-100, score final
    rating: int = 0                  # -1 (rejet) a 5 etoiles
    datetime_taken: Optional[datetime] = None
    iso: int = 0
    flash_fired: bool = False
    color_temp_exif: Optional[int] = None
    # Rempli apres burst detection
    burst_group: Optional[str] = None
    is_burst_best: bool = False


def load_preview(path: Path, max_size: int = 1200) -> Optional[np.ndarray]:
    """
    Charge une image pour analyse. Utilise le JPEG embarque dans les RAW
    pour la rapidite. Retourne un tableau BGR ou None si echec.
    """
    ext = path.suffix.lower()

    if ext in RAW_EXTENSIONS:
        try:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    buf = np.frombuffer(thumb.data, dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is not None:
                        return _resize(img, max_size)
                # Fallback : demosaic rapide (half_size)
                rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return _resize(img, max_size)
        except Exception:
            return None

    elif ext in JPEG_EXTENSIONS:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                return _resize(img, max_size)
        except Exception:
            pass

    return None


def _resize(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def read_exif(path: Path) -> dict:
    """Lit les tags EXIF. Fonctionne sur RAW et JPEG."""
    try:
        with open(path, "rb") as f:
            return exifread.process_file(f, stop_tag="UNDEF", details=False)
    except Exception:
        return {}


def _safe_ratio(val):
    """Convertit un objet Ratio/IFDRational exifread en float."""
    try:
        return float(val.values[0].num) / float(val.values[0].den)
    except Exception:
        try:
            return float(str(val))
        except Exception:
            return 0.0


def parse_exif(tags: dict) -> dict:
    """Extrait les champs utiles des tags EXIF bruts."""
    result = {
        "datetime": None,
        "iso": 0,
        "flash_fired": False,
        "color_temp": None,
        "exposure_time": 0.0,
    }

    # Date/heure
    for key in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        if key in tags:
            try:
                result["datetime"] = datetime.strptime(str(tags[key]), "%Y:%m:%d %H:%M:%S")
                break
            except Exception:
                pass

    # ISO
    if "EXIF ISOSpeedRatings" in tags:
        try:
            result["iso"] = int(str(tags["EXIF ISOSpeedRatings"]))
        except Exception:
            pass

    # Flash
    if "EXIF Flash" in tags:
        flash_val = str(tags["EXIF Flash"]).lower()
        result["flash_fired"] = "fired" in flash_val or (
            not ("did not fire" in flash_val) and "1" in flash_val
        )

    # Temperature couleur (si disponible)
    if "EXIF ColorTemperature" in tags:
        try:
            result["color_temp"] = int(str(tags["EXIF ColorTemperature"]))
        except Exception:
            pass

    # Temps d'exposition
    for key in ("EXIF ExposureTime", "EXIF ShutterSpeedValue"):
        if key in tags:
            result["exposure_time"] = _safe_ratio(tags[key])
            break

    return result


# ---------------------------------------------------------------------------
# Analyses image
# ---------------------------------------------------------------------------

def compute_blur_score(gray: np.ndarray) -> float:
    """
    Variance du Laplacien sur une image en niveaux de gris.
    Score > 100 : image nette. < 50 : floue.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure(gray: np.ndarray) -> dict:
    """
    Analyse l'histogramme pour evaluer l'exposition.
    Retourne mean_brightness, overexposed_ratio, underexposed_ratio.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return {"mean": 128.0, "overexposed": 0.0, "underexposed": 0.0}

    hist_norm = hist / total
    mean = float(np.dot(np.arange(256), hist_norm))
    overexposed = float(hist_norm[245:].sum())     # pixels brules
    underexposed = float(hist_norm[:10].sum())     # pixels noirs

    return {"mean": mean, "overexposed": overexposed, "underexposed": underexposed}


def detect_backlit(img_bgr: np.ndarray) -> bool:
    """
    Detection backlit : la peripherie est beaucoup plus lumineuse que le centre.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Zone centrale (40% de la surface)
    cy, cx = h // 2, w // 2
    cy_off, cx_off = int(h * 0.3), int(w * 0.3)
    center = gray[cy - cy_off:cy + cy_off, cx - cx_off:cx + cx_off]

    # Bordure = tout sauf le centre
    mask_border = np.ones((h, w), dtype=np.uint8)
    mask_border[cy - cy_off:cy + cy_off, cx - cx_off:cx + cx_off] = 0
    border_pixels = gray[mask_border == 1]

    if center.size == 0 or border_pixels.size == 0:
        return False

    center_mean = float(center.mean())
    border_mean = float(border_pixels.mean())

    # Backlit si la bordure est 50+ points plus lumineuse que le centre
    return (border_mean - center_mean) > 50


def detect_faces(img_bgr: np.ndarray) -> tuple[int, bool, list]:
    """
    Detection de visages et d'yeux ouverts.
    Retourne (nb_visages, yeux_ouverts, face_rects).
    face_rects en coordonnees de l'image ORIGINALE (pas resized).
    Protege contre les crashes OpenCV sur certains RAW.
    """
    try:
        face_cascade, eye_cascade = _get_cascades()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Resize pour eviter les crashes OpenCV sur les grandes images
        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.15, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return 0, False, []

        # Remettre les coordonnees a l'echelle originale
        inv_scale = 1.0 / scale
        face_rects = [
            (int(x * inv_scale), int(y * inv_scale),
             int(fw * inv_scale), int(fh * inv_scale))
            for (x, y, fw, fh) in faces
        ]

        # Cherche des yeux dans au moins un visage
        open_eyes = False
        for (x, y, fw, fh) in faces:
            roi = gray[y:y + fh, x:x + fw]
            eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.15, minNeighbors=3, minSize=(10, 10))
            if isinstance(eyes, np.ndarray) and len(eyes) >= 1:
                open_eyes = True
                break

        return int(len(faces)), open_eyes, face_rects

    except Exception:
        return 0, False, []


def detect_context(exif_parsed: dict, exposure: dict) -> str:
    """
    Determine le contexte de prise de vue depuis l'EXIF et l'exposition.
    """
    iso = exif_parsed.get("iso", 0)
    flash = exif_parsed.get("flash_fired", False)
    mean = exposure.get("mean", 128)

    if iso > 2500:
        return "low_light"
    if flash:
        return "indoor_flash"
    if mean > 190 and exposure.get("overexposed", 0) > 0.05:
        return "outdoor_bright"
    if iso < 800 and not flash:
        return "outdoor_day"
    return "indoor_ambient"


# ---------------------------------------------------------------------------
# Score global
# ---------------------------------------------------------------------------

def compute_score(
    blur: float,
    exposure: dict,
    face_count: int,
    open_eyes: bool,
    is_backlit: bool,
    context: str,
) -> float:
    """
    Calcule un score de qualite de 0 a 100.
    """
    score = 0.0

    # --- Netteté (0-45 pts) ---
    # Blur score : 0=flou total, 1000+=tres net
    # On normalise sur une courbe log
    blur_clamped = max(0.0, min(blur, 2000.0))
    blur_pts = (np.log1p(blur_clamped) / np.log1p(2000.0)) * 45
    score += blur_pts

    # --- Exposition (0-25 pts) ---
    mean = exposure.get("mean", 128)
    over = exposure.get("overexposed", 0)
    under = exposure.get("underexposed", 0)

    # Ideal : mean entre 90 et 185
    if 90 <= mean <= 185:
        expo_pts = 25.0
    else:
        deviation = min(abs(mean - 128), 128)
        expo_pts = max(0.0, 25.0 - deviation * 0.25)

    # Penalites pour pixels brules ou noirs
    expo_pts -= over * 60     # -6 pts si 10% de pixels brules
    expo_pts -= under * 40
    expo_pts = max(0.0, expo_pts)
    score += expo_pts

    # --- Visages (0-20 pts) ---
    if face_count >= 1:
        score += 15
        if open_eyes:
            score += 5

    # --- Penalites ---
    if is_backlit:
        score -= 5   # penalite legere (backlit peut etre voulu)

    return float(max(0.0, min(100.0, score)))


def score_to_rating(score: float) -> int:
    """Convertit un score 0-100 en rating Lightroom (-1 a 5)."""
    if score < 28:
        return -1   # Rejet
    if score < 42:
        return 1
    if score < 56:
        return 2
    if score < 68:
        return 3
    if score < 82:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------------------

def analyze_photo(path: Path, enable_faces: bool = True) -> Optional[PhotoAnalysis]:
    """
    Analyse complete d'une photo. Retourne None si le fichier ne peut pas etre lu.
    Protege contre toutes les erreurs OpenCV / rawpy.
    """
    result = PhotoAnalysis(path=path)

    try:
        # EXIF
        tags = read_exif(path)
        exif = parse_exif(tags)
        result.datetime_taken = exif["datetime"]
        result.iso = exif["iso"]
        result.flash_fired = exif["flash_fired"]
        result.color_temp_exif = exif["color_temp"]

        # Chargement image
        img = load_preview(path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Analyses
        result.blur_score = compute_blur_score(gray)
        exposure = compute_exposure(gray)
        result.mean_brightness = exposure["mean"]
        result.overexposed_ratio = exposure["overexposed"]
        result.underexposed_ratio = exposure["underexposed"]

        is_backlit = detect_backlit(img)

        if enable_faces:
            result.face_count, result.open_eyes, result.face_rects = detect_faces(img)

        result.context = "backlit" if is_backlit else detect_context(exif, exposure)

        # Score et rating
        result.score = compute_score(
            result.blur_score, exposure,
            result.face_count, result.open_eyes,
            is_backlit, result.context
        )
        result.rating = score_to_rating(result.score)

    except Exception:
        # En cas de crash OpenCV/rawpy, on retourne quand meme la photo
        # avec un score par defaut (pas de rejet, pas de correction)
        result.score = 50.0
        result.rating = 2
        result.context = "unknown"

    return result
