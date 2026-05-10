"""
Notation des photos par Claude Vision (Anthropic API).

Envoie des thumbnails a Claude qui juge la qualite de chaque photo
comme un photographe professionnel. Infiniment plus precis que les
heuristiques pixel (blur score, histogramme).

Usage :
  rater = AIRater(api_key="sk-ant-...")
  ratings = rater.rate_batch(photos, callback=on_progress)
"""

import base64
import json
import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass

from .photo_analyzer import PhotoAnalysis, load_preview


BATCH_SIZE = 3  # Moins de photos par batch = plus d'attention par photo

SYSTEM_PROMPT = """Tu es un photographe événementiel EXIGEANT qui fait le culling pour un client.

Tu DOIS être sélectif. Sur un event typique, un bon photographe garde 30-40% des photos et rejette le reste. Sois dur.

Note chaque photo de 0 à 5 :

0 = REJET IMMÉDIAT :
  - Sujet principal flou ou pas net (même légèrement)
  - Visage dans l'ombre, pas visible, ou pas mis en valeur
  - Yeux fermés, mi-clos, ou regard absent
  - Expression figée, gênée, bouche ouverte en pleine parole
  - Dos, nuque, 3/4 arrière sans visage identifiable
  - Sujet coupé de façon gênante (tête, menton, main)
  - Exposition ratée (même partiellement brûlée ou bouchée)
  - Photo "entre deux moments" — rien ne se passe
  - Photo de remplissage (décor vide, plafond, sol, mobilier seul)
  - Doublon moins bon d'une autre photo (même scène, moins bonne expression)

1 = Faible — Techniquement passable mais expression ou moment raté. On garde SEULEMENT si c'est la seule photo de cette personne.

2 = Moyenne — Correcte, rien de spécial. Publiable si besoin de volume.

3 = Bonne — Visage net, expression naturelle, bon cadrage. Publiable sans hésitation.

4 = Très bonne — Émotion visible, composition soignée, lumière réussie. À montrer au client.

5 = Excellente — THE photo. Moment fort, expression parfaite, lumière magnifique. Portfolio.

RÈGLES STRICTES :
- Si le visage du sujet principal n'est pas net ET bien éclairé → 0 ou 1, jamais plus
- Si personne ne sourit et que ce n'est pas un moment d'action → 0 ou 1
- Une photo "correcte mais sans intérêt" = 2 maximum, pas 3
- Le 3 est la note PAR DÉFAUT d'une bonne photo. Le 4 et 5 sont RARES.
- Tu dois rejeter (0) au moins 20% des photos d'un lot typique"""

USER_PROMPT_TEMPLATE = """Voici {count} photo(s) d'un événement. Sois EXIGEANT. Note chaque photo de 0 à 5.

Rappel : rejette (0) toute photo où le visage principal est dans l'ombre, flou, coupé, ou avec une mauvaise expression. Garde 30-40% max en 3+.

Réponds UNIQUEMENT avec un JSON array de {count} objets, un par photo dans l'ordre :
[{{"rating": 0, "reason": "visage dans l'ombre, pas exploitable"}}, ...]

Pas de texte avant ou après le JSON."""


@dataclass
class AIRating:
    """Resultat de notation IA pour une photo."""
    rating: int          # 0-5
    reason: str          # Justification courte
    confidence: float    # 0-1 (toujours 1 pour l'instant)


def _photo_to_thumbnail_b64(photo: PhotoAnalysis, max_size: int = 400) -> Optional[str]:
    """
    Charge une photo, redimensionne en thumbnail JPEG, retourne en base64.
    Petit (400px) pour minimiser les couts API.
    """
    img = load_preview(photo.path, max_size=max_size)
    if img is None:
        return None

    # Encoder en JPEG (qualite 70 = bon equilibre taille/qualite)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    success, buffer = cv2.imencode(".jpg", img, encode_params)
    if not success:
        return None

    return base64.standard_b64encode(buffer.tobytes()).decode("utf-8")


class AIRater:
    """Interface avec Claude Vision pour noter les photos."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def rate_batch(
        self,
        photos: List[PhotoAnalysis],
        batch_size: int = BATCH_SIZE,
        callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Note toutes les photos par lots.

        callback(done, total, current_file) est appele apres chaque lot.
        Retourne un dict[photo_path_str] -> AIRating.
        """
        results = {}
        total = len(photos)
        done = 0

        for i in range(0, total, batch_size):
            batch = photos[i:i + batch_size]
            batch_results = self._rate_single_batch(batch)

            for photo, rating in zip(batch, batch_results):
                results[str(photo.path)] = rating

            done += len(batch)
            if callback:
                last_name = batch[-1].path.name if batch else ""
                callback(done, total, last_name)

        return results

    def _rate_single_batch(self, photos: List[PhotoAnalysis]) -> List[AIRating]:
        """Envoie un lot de photos a Claude et parse la reponse."""

        # Preparer les images en base64
        content = []
        valid_indices = []

        for idx, photo in enumerate(photos):
            b64 = _photo_to_thumbnail_b64(photo)
            if b64 is None:
                continue

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
            content.append({
                "type": "text",
                "text": f"Photo {idx + 1}",
            })
            valid_indices.append(idx)

        if not content:
            return [AIRating(rating=2, reason="impossible a charger", confidence=0) for _ in photos]

        # Ajouter le prompt
        content.append({
            "type": "text",
            "text": USER_PROMPT_TEMPLATE.format(count=len(valid_indices)),
        })

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )

            # Parser la reponse JSON
            text = response.content[0].text.strip()

            # Extraire le JSON (parfois Claude ajoute du texte autour)
            json_start = text.find("[")
            json_end = text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                text = text[json_start:json_end]

            parsed = json.loads(text)

            # Construire les resultats
            results = [AIRating(rating=2, reason="non analysee", confidence=0)] * len(photos)

            for i, (valid_idx, item) in enumerate(zip(valid_indices, parsed)):
                rating = int(item.get("rating", 2))
                reason = str(item.get("reason", ""))
                results[valid_idx] = AIRating(
                    rating=max(0, min(5, rating)),
                    reason=reason,
                    confidence=1.0,
                )

            return results

        except Exception as e:
            # En cas d'erreur API, retourner des ratings par defaut
            return [
                AIRating(rating=2, reason=f"erreur API: {str(e)[:50]}", confidence=0)
                for _ in photos
            ]


def apply_ai_ratings(
    photos: List[PhotoAnalysis],
    ai_ratings: dict,
) -> None:
    """
    Applique les notes IA aux photos (modifie in-place).
    Convertit le rating IA (0-5) en rating Lightroom (-1 a 5).
    """
    for photo in photos:
        key = str(photo.path)
        if key not in ai_ratings:
            continue

        ai = ai_ratings[key]
        if ai.confidence == 0:
            continue  # Pas de rating fiable

        # Convertir : 0 IA = -1 Lightroom (rejet), 1-5 = 1-5
        if ai.rating == 0:
            photo.rating = -1
        else:
            photo.rating = ai.rating

        # Mettre a jour le score pour la coherence
        photo.score = ai.rating * 20  # 0=0, 1=20, 2=40, 3=60, 4=80, 5=100
