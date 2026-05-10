"""
Generation des fichiers XMP sidecar compatibles Lightroom Classic.

Le fichier .xmp est place a cote de la photo (meme dossier, meme nom).
Lightroom le lit automatiquement via "Lire les metadonnees depuis le fichier".
"""

from pathlib import Path
from typing import Optional
from .photo_analyzer import PhotoAnalysis
from .corrections import LightroomCorrections


XMP_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
{attrs}/>
 </rdf:RDF>
</x:xmpmeta>
"""


def _rating_to_xmp(rating: int) -> int:
    """
    Lightroom XMP :
    -1 = photo rejetee (drapeau X)
     0 = sans etoile
     1-5 = etoiles
    """
    return max(-1, min(5, rating))


def _color_label(photo: PhotoAnalysis) -> Optional[str]:
    """
    Assigne un label couleur en fonction du contexte pour faciliter
    le tri dans Lightroom. Optionnel — peut etre desactive.
    """
    labels = {
        "outdoor_day":    "Green",
        "outdoor_bright": "Green",
        "indoor_flash":   "Blue",
        "indoor_ambient": "Blue",
        "low_light":      "Purple",
        "backlit":        "Yellow",
    }
    return labels.get(photo.context, "")


def write_xmp(
    photo: PhotoAnalysis,
    corrections: LightroomCorrections,
    use_color_labels: bool = True,
) -> Path:
    """
    Ecrit le fichier XMP sidecar pour une photo.
    Retourne le chemin du fichier XMP cree.
    """
    xmp_path = photo.path.with_suffix(".xmp")

    # Attributs de base
    attrs: dict = {}

    # Rating
    attrs["xmp:Rating"] = str(_rating_to_xmp(photo.rating))

    # Label couleur (contexte)
    if use_color_labels:
        label = _color_label(photo)
        if label:
            attrs["xmp:Label"] = label

    # Corrections camera raw
    attrs.update(corrections.to_xmp_attrs())

    # Formattage des attributs XMP (indentation 4 espaces)
    attr_lines = "\n".join(f'    {k}="{v}"' for k, v in attrs.items())

    xmp_content = XMP_TEMPLATE.format(attrs=attr_lines)

    xmp_path.write_text(xmp_content, encoding="utf-8")
    return xmp_path


def write_report(
    photos: list[PhotoAnalysis],
    output_path: Path,
) -> None:
    """
    Ecrit un rapport CSV lisible dans Excel.
    Utile pour avoir une vue globale des decisions prises.
    """
    import csv

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Fichier", "Rating", "Score", "Nettete", "Luminosite moy.",
            "Sur-expose %", "Sous-expose %", "Visages", "Yeux ouverts",
            "Contexte", "Rafale", "Meilleure du groupe", "Date/heure"
        ])

        for p in sorted(photos, key=lambda x: x.path.name):
            writer.writerow([
                p.path.name,
                p.rating,
                f"{p.score:.1f}",
                f"{p.blur_score:.0f}",
                f"{p.mean_brightness:.0f}",
                f"{p.overexposed_ratio * 100:.1f}",
                f"{p.underexposed_ratio * 100:.1f}",
                p.face_count,
                "Oui" if p.open_eyes else "Non",
                p.context,
                p.burst_group or "",
                "Oui" if p.is_burst_best else "Non",
                str(p.datetime_taken) if p.datetime_taken else "",
            ])
