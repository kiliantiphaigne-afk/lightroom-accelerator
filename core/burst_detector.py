"""
Detection de rafales et de doublons.

Une rafale = plusieurs photos prises en moins de N secondes (defaut 2s).
Pour chaque rafale, on garde les meilleures photos et on rejette les autres.

Les doublons = photos quasi-identiques detectees par hash perceptuel.
"""

import imagehash
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from .photo_analyzer import PhotoAnalysis, load_preview


def group_bursts(photos: List[PhotoAnalysis], gap_seconds: float = 2.0) -> List[PhotoAnalysis]:
    """
    Groupe les photos en rafales basees sur l'heure de prise de vue EXIF.
    Assigne un burst_group a chaque photo (None = photo isolee).
    Marque is_burst_best pour les meilleures photos de chaque rafale.

    Les photos sans datetime sont ignorees du groupement (traitees comme isolees).
    """
    # Separons les photos avec et sans datetime
    with_dt = [p for p in photos if p.datetime_taken is not None]
    without_dt = [p for p in photos if p.datetime_taken is None]

    # Trier par date
    with_dt.sort(key=lambda p: p.datetime_taken)

    bursts: List[List[PhotoAnalysis]] = []
    current_burst: List[PhotoAnalysis] = []

    for photo in with_dt:
        if not current_burst:
            current_burst.append(photo)
        else:
            delta = (photo.datetime_taken - current_burst[-1].datetime_taken).total_seconds()
            if delta <= gap_seconds:
                current_burst.append(photo)
            else:
                bursts.append(current_burst)
                current_burst = [photo]

    if current_burst:
        bursts.append(current_burst)

    # Traitement des rafales (>= 2 photos)
    burst_id = 0
    for burst in bursts:
        if len(burst) == 1:
            # Photo isolee — pas de groupe
            burst[0].burst_group = None
            burst[0].is_burst_best = True
            continue

        burst_id += 1
        group_name = f"burst_{burst_id:04d}"

        # Trouver les meilleures photos
        best_count = _best_count_for_burst(len(burst))
        sorted_burst = sorted(burst, key=lambda p: p.score, reverse=True)

        for i, photo in enumerate(sorted_burst):
            photo.burst_group = group_name
            photo.is_burst_best = (i < best_count)

        # Les photos non best dans une rafale : score reduit + pire rating
        for photo in sorted_burst[best_count:]:
            photo.score = max(0.0, photo.score - 20)
            # Si c'est deja un rejet, on laisse. Sinon on degrade.
            if photo.rating > 1:
                photo.rating = max(-1, photo.rating - 2)

    return photos


def _best_count_for_burst(total: int) -> int:
    """Combien de photos garder dans une rafale selon sa taille."""
    if total <= 3:
        return 1
    if total <= 6:
        return 2
    if total <= 10:
        return 3
    return max(3, total // 4)


def detect_duplicates(
    photos: List[PhotoAnalysis],
    hash_threshold: int = 8,
    max_per_folder: Optional[int] = None,
) -> List[PhotoAnalysis]:
    """
    Detecte les photos quasi-identiques via hash perceptuel (pHash).
    Dans chaque groupe de doublons, garde la meilleure, rejette les autres.

    hash_threshold : distance de Hamming max pour considerer deux images identiques.
    Plus petite = plus strict.
    """
    # On ne compare que les photos qui ne sont PAS deja rejetees
    candidates = [p for p in photos if p.rating != -1]
    rejected   = [p for p in photos if p.rating == -1]

    if not candidates:
        return photos

    # Calcul des hashes
    hashes: Dict[Path, imagehash.ImageHash] = {}
    for photo in candidates:
        img = load_preview(photo.path, max_size=200)  # petite taille suffisante pour le hash
        if img is None:
            continue
        try:
            from PIL import Image as PILImage
            import cv2
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            hashes[photo.path] = imagehash.phash(pil_img)
        except Exception:
            pass

    # Regroupement par similarite (union-find simple)
    path_to_photo = {p.path: p for p in candidates}
    visited = set()
    duplicate_groups: List[List[PhotoAnalysis]] = []

    candidate_paths = [p.path for p in candidates if p.path in hashes]

    for i, path_a in enumerate(candidate_paths):
        if path_a in visited:
            continue
        group = [path_to_photo[path_a]]
        visited.add(path_a)

        for path_b in candidate_paths[i + 1:]:
            if path_b in visited:
                continue
            dist = hashes[path_a] - hashes[path_b]
            if dist <= hash_threshold:
                group.append(path_to_photo[path_b])
                visited.add(path_b)

        if len(group) > 1:
            duplicate_groups.append(group)

    # Dans chaque groupe de doublons : garder le meilleur, rejeter les autres
    for group in duplicate_groups:
        sorted_group = sorted(group, key=lambda p: p.score, reverse=True)
        for photo in sorted_group[1:]:
            photo.rating = -1
            photo.score = max(0.0, photo.score - 30)

    return photos
