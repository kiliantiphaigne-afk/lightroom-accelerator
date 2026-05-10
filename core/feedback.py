"""
Systeme de feedback / apprentissage.

Apres que l'utilisateur a corrige les notes dans Lightroom,
ce module relit les XMP pour detecter les changements et
stocker les preferences de l'utilisateur.

Au fil des sessions, l'outil apprend :
- Quel niveau de flou l'utilisateur accepte
- Quels contextes il note plus haut/bas
- Son style de culling (strict vs permissif)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from collections import defaultdict

from .photo_analyzer import PhotoAnalysis, RAW_EXTENSIONS, JPEG_EXTENSIONS


FEEDBACK_FILE = "feedback_data.json"

SUPPORTED_EXTS = RAW_EXTENSIONS | JPEG_EXTENSIONS


def read_xmp_rating(xmp_path: Path) -> Optional[int]:
    """
    Lit la note depuis un fichier XMP (apres edition dans Lightroom).
    Retourne le rating (-1 a 5) ou None si illisible.
    """
    try:
        tree = ET.parse(str(xmp_path))
        root = tree.getroot()

        # Chercher xmp:Rating dans les attributs
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xmp": "http://ns.adobe.com/xap/1.0/",
        }

        # Methode 1 : attribut sur rdf:Description
        for desc in root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
            rating = desc.get("{http://ns.adobe.com/xap/1.0/}Rating")
            if rating is not None:
                return int(rating)

        # Methode 2 : element enfant
        for elem in root.iter("{http://ns.adobe.com/xap/1.0/}Rating"):
            if elem.text:
                return int(elem.text)

        return None

    except Exception:
        return None


def collect_feedback(
    photos: List[PhotoAnalysis],
    folder: Path,
) -> dict:
    """
    Compare les notes de l'outil avec les notes actuelles dans les XMP
    (apres edition Lightroom). Retourne les stats de feedback.

    Retourne :
    {
        "total": int,
        "changed": int,
        "corrections": [
            {
                "file": str,
                "tool_rating": int,
                "user_rating": int,
                "delta": int,
                "features": {...}
            }
        ],
        "stats": {
            "promoted": int,   # user a monte la note
            "demoted": int,    # user a baisse la note
            "unchanged": int,
        }
    }
    """
    corrections = []
    promoted = 0
    demoted = 0
    unchanged = 0

    for photo in photos:
        xmp_path = photo.path.with_suffix(".xmp")
        if not xmp_path.exists():
            continue

        user_rating = read_xmp_rating(xmp_path)
        if user_rating is None:
            continue

        tool_rating = photo.rating
        delta = user_rating - tool_rating

        if delta > 0:
            promoted += 1
        elif delta < 0:
            demoted += 1
        else:
            unchanged += 1

        if delta != 0:
            corrections.append({
                "file": photo.path.name,
                "tool_rating": tool_rating,
                "user_rating": user_rating,
                "delta": delta,
                "features": {
                    "blur_score": round(photo.blur_score, 1),
                    "mean_brightness": round(photo.mean_brightness, 1),
                    "overexposed": round(photo.overexposed_ratio, 3),
                    "underexposed": round(photo.underexposed_ratio, 3),
                    "face_count": photo.face_count,
                    "open_eyes": photo.open_eyes,
                    "context": photo.context,
                    "iso": photo.iso,
                    "flash": photo.flash_fired,
                },
            })

    return {
        "total": len(photos),
        "changed": len(corrections),
        "corrections": corrections,
        "stats": {
            "promoted": promoted,
            "demoted": demoted,
            "unchanged": unchanged,
        },
    }


def save_feedback(
    feedback: dict,
    folder: Path,
    feedback_path: Optional[Path] = None,
) -> Path:
    """
    Sauvegarde le feedback dans le fichier JSON persistant.
    Accumule les sessions — ne remplace pas les anciennes.
    """
    if feedback_path is None:
        feedback_path = Path(FEEDBACK_FILE)

    # Charger les donnees existantes
    existing = {"version": 2, "sessions": [], "learned": {}}
    if feedback_path.exists():
        try:
            existing = json.loads(feedback_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Ajouter la session
    session = {
        "date": datetime.now().isoformat(),
        "folder": str(folder),
        "total_photos": feedback["total"],
        "total_corrections": feedback["changed"],
        "stats": feedback["stats"],
        "corrections": feedback["corrections"],
    }
    existing["sessions"].append(session)

    # Recalculer les apprentissages
    existing["learned"] = _compute_learned(existing["sessions"])

    feedback_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return feedback_path


def _compute_learned(sessions: list) -> dict:
    """
    Calcule les ajustements appris a partir de toutes les sessions.
    Retourne des recommandations pour les futures analyses.
    """
    all_corrections = []
    for s in sessions:
        all_corrections.extend(s.get("corrections", []))

    if not all_corrections:
        return {}

    learned = {}

    # 1. Seuil de flou optimal
    # Trouver les photos que l'utilisateur a gardees malgre un blur_score bas
    kept_blur_scores = [
        c["features"]["blur_score"]
        for c in all_corrections
        if c["user_rating"] >= 1 and c["tool_rating"] == -1
        and c["features"]["blur_score"] > 0
    ]
    if kept_blur_scores:
        # Le seuil devrait etre en dessous du minimum garde par l'utilisateur
        learned["blur_threshold_suggested"] = max(5, int(min(kept_blur_scores) * 0.8))

    # 2. Ajustements par contexte
    context_deltas = defaultdict(list)
    for c in all_corrections:
        ctx = c["features"]["context"]
        context_deltas[ctx].append(c["delta"])

    context_adj = {}
    for ctx, deltas in context_deltas.items():
        avg = sum(deltas) / len(deltas)
        if abs(avg) >= 0.3:  # Seulement si le biais est significatif
            context_adj[ctx] = round(avg, 2)

    if context_adj:
        learned["context_adjustments"] = context_adj

    # 3. Tendance generale (strict vs permissif)
    all_deltas = [c["delta"] for c in all_corrections]
    avg_delta = sum(all_deltas) / len(all_deltas)
    learned["avg_delta"] = round(avg_delta, 2)
    learned["total_corrections_analyzed"] = len(all_corrections)

    if avg_delta > 0.5:
        learned["tendency"] = "L'utilisateur est plus permissif que l'outil"
    elif avg_delta < -0.5:
        learned["tendency"] = "L'utilisateur est plus strict que l'outil"
    else:
        learned["tendency"] = "L'outil est bien calibre"

    return learned


def load_learned(feedback_path: Optional[Path] = None) -> dict:
    """
    Charge les preferences apprises depuis le fichier de feedback.
    Retourne un dict vide si pas de donnees.
    """
    if feedback_path is None:
        feedback_path = Path(FEEDBACK_FILE)

    if not feedback_path.exists():
        return {}

    try:
        data = json.loads(feedback_path.read_text(encoding="utf-8"))
        return data.get("learned", {})
    except Exception:
        return {}


def apply_learned_adjustments(
    photos: List[PhotoAnalysis],
    learned: dict,
) -> int:
    """
    Applique les ajustements appris aux photos.
    Modifie les ratings in-place.
    Retourne le nombre de photos ajustees.
    """
    if not learned:
        return 0

    context_adj = learned.get("context_adjustments", {})
    avg_delta = learned.get("avg_delta", 0)
    n_adjusted = 0

    for photo in photos:
        adjustment = 0.0

        # Ajustement par contexte
        if photo.context in context_adj:
            adjustment += context_adj[photo.context]

        # Ajustement global (tendance)
        if abs(avg_delta) >= 0.3:
            adjustment += avg_delta * 0.5  # Appliquer 50% du delta moyen

        if abs(adjustment) < 0.5:
            continue

        # Appliquer
        adj_int = round(adjustment)
        new_rating = photo.rating + adj_int

        # Clamp
        if new_rating < -1:
            new_rating = -1
        if new_rating > 5:
            new_rating = 5

        if new_rating != photo.rating:
            photo.rating = new_rating
            photo.score = max(0, min(100, photo.score + adj_int * 15))
            n_adjusted += 1

    return n_adjusted
