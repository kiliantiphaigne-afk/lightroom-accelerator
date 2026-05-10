"""
Calcul des corrections Lightroom (valeurs XMP crs:) basees sur
l'analyse de chaque photo. Tout reste non-destructif.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .photo_analyzer import PhotoAnalysis


@dataclass
class LightroomCorrections:
    """Valeurs de correction compatibles Lightroom Classic (namespace crs:)."""
    # Tonalite
    Exposure2012:   float = 0.0   # -5.00 a +5.00
    Contrast2012:   int   = 0     # -100 a +100
    Highlights2012: int   = 0     # -100 a +100
    Shadows2012:    int   = 0     # -100 a +100
    Whites2012:     int   = 0     # -100 a +100
    Blacks2012:     int   = 0     # -100 a +100

    # Presence
    Clarity2012:    int   = 0     # -100 a +100
    Dehaze:         int   = 0     # -100 a +100
    Vibrance:       int   = 0     # -100 a +100
    Saturation:     int   = 0     # -100 a +100

    # Balance des blancs
    ColorTemperature: Optional[int] = None   # Kelvin, None = garder EXIF
    ColorTint:       int = 0                 # -150 a +150

    # Version process Lightroom
    ProcessVersion: str = "11.0"

    def to_xmp_attrs(self) -> dict:
        """Convertit en dictionnaire d'attributs XMP."""
        attrs = {
            "crs:ProcessVersion": self.ProcessVersion,
            "crs:Exposure2012": f"{self.Exposure2012:.2f}",
            "crs:Contrast2012": str(self.Contrast2012),
            "crs:Highlights2012": str(self.Highlights2012),
            "crs:Shadows2012": str(self.Shadows2012),
            "crs:Whites2012": str(self.Whites2012),
            "crs:Blacks2012": str(self.Blacks2012),
            "crs:Clarity2012": str(self.Clarity2012),
            "crs:Dehaze": str(self.Dehaze),
            "crs:Vibrance": str(self.Vibrance),
            "crs:Saturation": str(self.Saturation),
            "crs:ColorTint": str(self.ColorTint),
            "crs:AutoTone": "False",
            "crs:HasCrop": "False",
            "crs:AlreadyApplied": "True",
        }
        if self.ColorTemperature is not None:
            attrs["crs:ColorTemperature"] = str(self.ColorTemperature)
        return attrs


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def calculate_exposure_correction(mean_brightness: float, over: float, under: float) -> tuple:
    """
    Retourne (Exposure2012, Highlights2012, Shadows2012, Whites2012, Blacks2012).
    """
    exposure = 0.0
    highlights = 0
    shadows = 0
    whites = 0
    blacks = 0

    # Photo sous-exposee — corrections conservatrices (point de depart, pas final)
    if mean_brightness < 70:
        # Seulement les vrais cas sombres
        exposure = _clamp((100 - mean_brightness) / 120.0, 0.0, 1.2)
        shadows = _clamp(int((80 - mean_brightness) * 0.3), 0, 30)
        blacks = _clamp(int((70 - mean_brightness) * 0.15), 0, 10)

    # Photo sur-exposee
    elif mean_brightness > 200:
        exposure = _clamp(-(mean_brightness - 195) / 80.0, -1.5, 0.0)
        highlights = _clamp(-int((mean_brightness - 190) * 0.7), -80, 0)
        whites = _clamp(-int((mean_brightness - 200) * 0.4), -40, 0)

    # Pixels brules : reduire les hautes lumieres meme si la moyenne est bonne
    if over > 0.05:
        highlights = _clamp(highlights - int(over * 150), -80, 0)

    # Pixels sous-noirs : pousser les noirs legerement
    if under > 0.08:
        blacks = _clamp(blacks + int(under * 80), 0, 15)

    return exposure, highlights, shadows, whites, blacks


def calculate_color_temp(photo: PhotoAnalysis) -> Optional[int]:
    """
    Suggere une temperature couleur basee sur le contexte.
    Retourne None pour laisser Lightroom utiliser l'EXIF.
    """
    if photo.color_temp_exif:
        base = photo.color_temp_exif
    else:
        base = None

    # Corrections par contexte
    corrections = {
        "indoor_flash": -200,      # Flash trop chaud → leger refroidissement
        "indoor_ambient": +150,    # Ambiance chaude → pousser un peu vers le chaud
        "low_light": +200,         # Tungstene souvent → compenser
        "outdoor_day": 0,          # Laisser tel quel
        "outdoor_bright": -100,    # Lumiere vive → leger refroidissement
        "backlit": +100,           # Backlit souvent bleuté → rechauffer
    }

    if base is None:
        # Valeurs par defaut si pas d'EXIF temp
        defaults = {
            "indoor_flash": 5500,
            "indoor_ambient": 3800,
            "low_light": 3400,
            "outdoor_day": 5600,
            "outdoor_bright": 5200,
            "backlit": 6000,
        }
        return defaults.get(photo.context, None)

    offset = corrections.get(photo.context, 0)
    return _clamp(base + offset, 2000, 12000)


def build_corrections(photo: PhotoAnalysis, auto_color: bool = True) -> LightroomCorrections:
    """
    Construit les corrections Lightroom pour une photo analysee.
    Ne cree pas de corrections pour les photos rejetees.
    """
    c = LightroomCorrections()

    if photo.rating == -1:
        return c  # Pas de corrections pour les rejets

    # --- Exposition ---
    exposure, highlights, shadows, whites, blacks = calculate_exposure_correction(
        photo.mean_brightness,
        photo.overexposed_ratio,
        photo.underexposed_ratio,
    )
    c.Exposure2012 = exposure
    c.Highlights2012 = highlights
    c.Shadows2012 = shadows
    c.Whites2012 = whites
    c.Blacks2012 = blacks

    # --- Corrections par contexte ---
    context = photo.context

    if context == "backlit":
        # Recuperation douce
        c.Highlights2012 = _clamp(c.Highlights2012 - 30, -70, 0)
        c.Shadows2012    = _clamp(c.Shadows2012 + 20, 0, 50)
        c.Whites2012     = _clamp(c.Whites2012 - 10, -40, 0)

    elif context == "low_light":
        c.Shadows2012 = _clamp(c.Shadows2012 + 15, 0, 40)
        c.Clarity2012 = 5

    elif context == "indoor_flash":
        c.Highlights2012 = _clamp(c.Highlights2012 - 15, -50, 0)

    elif context == "outdoor_bright":
        c.Highlights2012 = _clamp(c.Highlights2012 - 15, -50, 0)
        c.Vibrance       = 8

    elif context == "outdoor_day":
        c.Vibrance = 10

    elif context == "indoor_ambient":
        c.Vibrance    = 8
        c.Shadows2012 = _clamp(c.Shadows2012 + 10, 0, 40)

    # --- Temperature couleur ---
    if auto_color:
        c.ColorTemperature = calculate_color_temp(photo)

    return c
