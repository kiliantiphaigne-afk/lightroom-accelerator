"""
Generation des fichiers XMP sidecar compatibles Lightroom Classic.

Le fichier .xmp est place a cote de la photo (meme dossier, meme nom).
Lightroom le lit automatiquement via "Lire les metadonnees depuis le fichier".
"""

import shutil
from pathlib import Path
from typing import Optional, List
from .photo_analyzer import PhotoAnalysis
from .corrections import LightroomCorrections
from .auto_crop import CropResult


# Template XMP avec support des keywords (bloc dc:subject)
XMP_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
{attrs}>
{keywords_block}  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

# Template sans keywords (self-closing tag)
XMP_TEMPLATE_NO_KW = """\
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
    return max(-1, min(5, rating))


def _color_label(photo: PhotoAnalysis) -> Optional[str]:
    labels = {
        "outdoor_day":    "Green",
        "outdoor_bright": "Green",
        "indoor_flash":   "Blue",
        "indoor_ambient": "Blue",
        "low_light":      "Purple",
        "backlit":        "Yellow",
    }
    return labels.get(photo.context, "")


def generate_keywords(photo: PhotoAnalysis) -> List[str]:
    """
    Genere des mots-cles automatiques basés sur l'analyse de la photo.
    Lightroom les affiche dans le panneau Mots-cles et permet le filtrage.
    """
    keywords = []

    # Type de sujet selon le nombre de visages
    if photo.face_count == 0:
        keywords.append("ambiance")
    elif photo.face_count == 1:
        keywords.append("portrait")
    elif photo.face_count <= 4:
        keywords.append("groupe")
    else:
        keywords.append("foule")

    # Contexte de prise de vue
    context_kw = {
        "outdoor_day": "exterieur",
        "outdoor_bright": "exterieur",
        "indoor_flash": "interieur",
        "indoor_ambient": "interieur",
        "low_light": "ambiance-sombre",
        "backlit": "contre-jour",
    }
    kw = context_kw.get(photo.context)
    if kw:
        keywords.append(kw)

    # Flash
    if photo.flash_fired:
        keywords.append("flash")

    # Qualite
    if photo.score >= 82:
        keywords.append("top")
    if photo.is_burst_best and photo.burst_group:
        keywords.append("meilleure-rafale")

    return keywords


def _build_keywords_block(keywords: List[str]) -> str:
    """Genere le bloc XML dc:subject pour les mots-cles Lightroom."""
    if not keywords:
        return ""
    items = "\n".join(f"       <rdf:li>{kw}</rdf:li>" for kw in keywords)
    return (
        "   <dc:subject>\n"
        "    <rdf:Bag>\n"
        f"{items}\n"
        "    </rdf:Bag>\n"
        "   </dc:subject>\n"
    )


def backup_xmp(photo_path: Path) -> Optional[Path]:
    """
    Si un fichier .xmp existe deja, le copie en .xmp.bak.
    Retourne le chemin du backup ou None si pas de XMP existant.
    """
    xmp_path = photo_path.with_suffix(".xmp")
    if xmp_path.exists():
        bak_path = photo_path.with_suffix(".xmp.bak")
        shutil.copy2(str(xmp_path), str(bak_path))
        return bak_path
    return None


def write_xmp(
    photo: PhotoAnalysis,
    corrections: LightroomCorrections,
    crop: Optional[CropResult] = None,
    keywords: Optional[List[str]] = None,
    use_color_labels: bool = True,
    do_backup: bool = True,
) -> Path:
    """
    Ecrit le fichier XMP sidecar pour une photo.
    Sauvegarde l'ancien XMP en .xmp.bak si do_backup=True.
    Retourne le chemin du fichier XMP cree.
    """
    xmp_path = photo.path.with_suffix(".xmp")

    # Backup de l'existant
    if do_backup:
        backup_xmp(photo.path)

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

    # Recadrage automatique
    if crop is not None and crop.has_crop:
        attrs.update(crop.to_xmp_attrs())

    # Formattage des attributs XMP (indentation 4 espaces)
    attr_lines = "\n".join(f'    {k}="{v}"' for k, v in attrs.items())

    # Choix du template selon la presence de keywords
    if keywords:
        kw_block = _build_keywords_block(keywords)
        xmp_content = XMP_TEMPLATE.format(attrs=attr_lines, keywords_block=kw_block)
    else:
        xmp_content = XMP_TEMPLATE_NO_KW.format(attrs=attr_lines)

    xmp_path.write_text(xmp_content, encoding="utf-8")
    return xmp_path


def write_report(
    photos: list[PhotoAnalysis],
    output_path: Path,
) -> None:
    """
    Ecrit un rapport CSV lisible dans Excel.
    """
    import csv

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Fichier", "Rating", "Score", "Nettete", "Luminosite moy.",
            "Sur-expose %", "Sous-expose %", "Visages", "Yeux ouverts",
            "Contexte", "Keywords", "Rafale", "Meilleure du groupe", "Date/heure"
        ])

        for p in sorted(photos, key=lambda x: x.path.name):
            kw = generate_keywords(p)
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
                ", ".join(kw),
                p.burst_group or "",
                "Oui" if p.is_burst_best else "Non",
                str(p.datetime_taken) if p.datetime_taken else "",
            ])
