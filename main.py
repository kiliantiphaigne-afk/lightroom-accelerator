"""
Lightroom Accelerator — outil de culling et d'edition automatique.

Lance ce fichier directement :  python main.py
Ou utilise le .exe genere par build_exe.bat.

Workflow :
  1. Choisir le dossier contenant les photos
  2. Ajuster les parametres
  3. Cliquer Analyser
  4. Dans Lightroom : selectionner tout → Lire les metadonnees depuis le fichier
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import queue
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

from core.photo_analyzer import (
    analyze_photo, PhotoAnalysis, RAW_EXTENSIONS, JPEG_EXTENSIONS
)
from core.burst_detector import group_bursts, detect_duplicates
from core.corrections import build_corrections, harmonize_white_balance
from core.auto_crop import auto_crop
from core.xmp_writer import write_xmp, write_report, generate_keywords
from core.feedback import (
    collect_feedback, save_feedback, load_learned, apply_learned_adjustments
)

CONFIG_FILE = Path("config.json")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

APP_TITLE   = "Lightroom Accelerator"
APP_VERSION = "1.0.0"
BG_DARK     = "#1e1e2e"
BG_PANEL    = "#2a2a3e"
BG_INPUT    = "#313145"
FG_MAIN     = "#cdd6f4"
FG_DIM      = "#6c7086"
ACCENT      = "#89b4fa"
GREEN       = "#a6e3a1"
RED         = "#f38ba8"
YELLOW      = "#f9e2af"
PURPLE      = "#cba6f7"

SUPPORTED_EXTS = RAW_EXTENSIONS | JPEG_EXTENSIONS


# ---------------------------------------------------------------------------
# Config (cle API persistante)
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def get_api_key() -> Optional[str]:
    return load_config().get("anthropic_api_key")

def set_api_key(key: str):
    cfg = load_config()
    cfg["anthropic_api_key"] = key
    save_config(cfg)


# ---------------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------------

def styled_frame(parent, bg=BG_PANEL, **kwargs) -> ttk.Frame:
    f = tk.Frame(parent, bg=bg, **kwargs)
    return f


def label(parent, text, fg=FG_MAIN, bg=BG_PANEL, font_size=10, bold=False, **kwargs) -> tk.Label:
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, fg=fg, bg=bg,
                    font=("Segoe UI", font_size, weight), **kwargs)


def heading(parent, text, bg=BG_PANEL) -> tk.Label:
    return label(parent, text, fg=ACCENT, bg=bg, font_size=11, bold=True)


# ---------------------------------------------------------------------------
# Fenetres et widgets
# ---------------------------------------------------------------------------

class SettingsPanel(tk.LabelFrame):
    """Panneau de configuration des parametres d'analyse."""

    def __init__(self, parent):
        super().__init__(
            parent, text=" Paramètres ", bg=BG_PANEL, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove", padx=12, pady=8
        )

        # -- Variables --
        self.blur_threshold     = tk.IntVar(value=60)
        self.burst_gap          = tk.DoubleVar(value=2.0)
        self.enable_faces       = tk.BooleanVar(value=True)
        self.enable_duplicates  = tk.BooleanVar(value=True)
        self.auto_corrections   = tk.BooleanVar(value=True)
        self.auto_crop          = tk.BooleanVar(value=True)
        self.ai_rating          = tk.BooleanVar(value=False)
        self.color_labels       = tk.BooleanVar(value=True)
        self.generate_report    = tk.BooleanVar(value=True)
        self.max_workers        = tk.IntVar(value=4)

        row = 0

        def row_lbl(text):
            nonlocal row
            label(self, text, bg=BG_PANEL, fg=FG_DIM).grid(
                row=row, column=0, sticky="w", pady=3
            )

        def row_widget(widget):
            nonlocal row
            widget.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
            row += 1

        # Seuil de flou
        row_lbl("Seuil flou (netteté min.)")
        frame_blur = tk.Frame(self, bg=BG_PANEL)
        self._blur_scale = tk.Scale(
            frame_blur, from_=10, to=200, orient="horizontal",
            variable=self.blur_threshold, bg=BG_PANEL, fg=FG_MAIN,
            highlightthickness=0, troughcolor=BG_INPUT, activebackground=ACCENT,
            length=160, showvalue=False
        )
        self._blur_scale.pack(side="left")
        self._blur_val_lbl = tk.Label(
            frame_blur, textvariable=self.blur_threshold,
            bg=BG_PANEL, fg=ACCENT, font=("Segoe UI", 10), width=4
        )
        self._blur_val_lbl.pack(side="left")
        row_widget(frame_blur)

        # Gap rafale
        row_lbl("Intervalle rafale (sec.)")
        frame_gap = tk.Frame(self, bg=BG_PANEL)
        self._gap_scale = tk.Scale(
            frame_gap, from_=0.5, to=10.0, resolution=0.5, orient="horizontal",
            variable=self.burst_gap, bg=BG_PANEL, fg=FG_MAIN,
            highlightthickness=0, troughcolor=BG_INPUT, activebackground=ACCENT,
            length=160, showvalue=False
        )
        self._gap_scale.pack(side="left")
        self._gap_val_lbl = tk.Label(
            frame_gap, textvariable=self.burst_gap,
            bg=BG_PANEL, fg=ACCENT, font=("Segoe UI", 10), width=4
        )
        self._gap_val_lbl.pack(side="left")
        row_widget(frame_gap)

        # Threads
        row_lbl("Threads parallèles")
        frame_thr = tk.Frame(self, bg=BG_PANEL)
        self._thr_scale = tk.Scale(
            frame_thr, from_=1, to=16, orient="horizontal",
            variable=self.max_workers, bg=BG_PANEL, fg=FG_MAIN,
            highlightthickness=0, troughcolor=BG_INPUT, activebackground=ACCENT,
            length=160, showvalue=False
        )
        self._thr_scale.pack(side="left")
        self._thr_val_lbl = tk.Label(
            frame_thr, textvariable=self.max_workers,
            bg=BG_PANEL, fg=ACCENT, font=("Segoe UI", 10), width=4
        )
        self._thr_val_lbl.pack(side="left")
        row_widget(frame_thr)

        # Checkboxes
        def chk(text, var, fg=FG_MAIN):
            nonlocal row
            c = tk.Checkbutton(
                self, text=text, variable=var,
                bg=BG_PANEL, fg=fg, selectcolor=BG_INPUT,
                activebackground=BG_PANEL, activeforeground=fg,
                font=("Segoe UI", 10)
            )
            c.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            row += 1

        chk("Détection des visages / yeux fermés", self.enable_faces, FG_MAIN)
        chk("Détecter les doublons (hash perceptuel)", self.enable_duplicates, FG_MAIN)
        chk("Appliquer corrections auto (exposition, couleur)", self.auto_corrections, GREEN)
        chk("Recadrage automatique (visages + horizon)", self.auto_crop, GREEN)
        chk("Tri IA — Claude Vision (plus précis, ~10€/3000 photos)", self.ai_rating, ACCENT)
        chk("Labels couleur par contexte (vert/bleu/violet...)", self.color_labels, FG_MAIN)
        chk("Générer rapport CSV", self.generate_report, FG_MAIN)

        self.columnconfigure(1, weight=1)


class ResultsPanel(tk.LabelFrame):
    """Panneau affichant les resultats de l'analyse."""

    def __init__(self, parent):
        super().__init__(
            parent, text=" Résultats ", bg=BG_PANEL, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove", padx=12, pady=8
        )
        self._vars = {}
        rows = [
            ("total",     "📁 Total photos",       FG_MAIN),
            ("analyzed",  "🔍 Analysées",           FG_MAIN),
            ("picks",     "✅ Sélectionnées",        GREEN),
            ("rejects",   "❌ Rejetées",             RED),
            ("bursts",    "📸 Rafales détectées",    YELLOW),
            ("dupes",     "🔁 Doublons supprimés",   YELLOW),
            ("time",      "⏱ Temps d'analyse",       FG_DIM),
        ]
        for i, (key, text, color) in enumerate(rows):
            var = tk.StringVar(value="—")
            self._vars[key] = var
            label(self, text + " :", fg=FG_DIM, bg=BG_PANEL).grid(
                row=i, column=0, sticky="w", pady=3
            )
            label(self, "", fg=color, bg=BG_PANEL, bold=True,
                  textvariable=var).grid(
                row=i, column=1, sticky="e", padx=(16, 0), pady=3
            )
        self.columnconfigure(0, weight=1)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._vars:
                self._vars[key].set(str(value))

    def reset(self):
        for var in self._vars.values():
            var.set("—")


class LogPanel(tk.LabelFrame):
    """Zone de logs avec defilement automatique."""

    def __init__(self, parent):
        super().__init__(
            parent, text=" Journal ", bg=BG_PANEL, fg=ACCENT,
            font=("Segoe UI", 10, "bold"), bd=1, relief="groove", padx=6, pady=6
        )
        self._text = tk.Text(
            self, bg=BG_DARK, fg=FG_MAIN, font=("Consolas", 9),
            wrap="word", state="disabled", relief="flat",
            insertbackground=FG_MAIN, selectbackground=BG_INPUT
        )
        scrollbar = ttk.Scrollbar(self, command=self._text.yview)
        self._text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._text.pack(fill="both", expand=True)

        # Tags pour colorer les messages
        self._text.tag_config("info",  foreground=FG_MAIN)
        self._text.tag_config("ok",    foreground=GREEN)
        self._text.tag_config("warn",  foreground=YELLOW)
        self._text.tag_config("error", foreground=RED)
        self._text.tag_config("dim",   foreground=FG_DIM)

    def log(self, message: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._text.config(state="normal")
        self._text.insert("end", f"[{ts}] {message}\n", level)
        self._text.see("end")
        self._text.config(state="disabled")

    def clear(self):
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(820, 640)

        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Titre
        header = tk.Frame(self, bg=BG_DARK, pady=14)
        header.pack(fill="x", padx=20)
        tk.Label(
            header, text=APP_TITLE, fg=ACCENT, bg=BG_DARK,
            font=("Segoe UI", 18, "bold")
        ).pack(side="left")
        tk.Label(
            header, text=f"v{APP_VERSION}", fg=FG_DIM, bg=BG_DARK,
            font=("Segoe UI", 10)
        ).pack(side="left", padx=(8, 0), anchor="s", pady=(0, 4))

        # Separateur
        tk.Frame(self, bg=BG_PANEL, height=1).pack(fill="x")

        # Selecteur de dossier
        folder_frame = tk.Frame(self, bg=BG_DARK, pady=10, padx=20)
        folder_frame.pack(fill="x")

        tk.Label(
            folder_frame, text="Dossier photos :", fg=FG_DIM, bg=BG_DARK,
            font=("Segoe UI", 10)
        ).pack(side="left")

        self._folder_var = tk.StringVar()
        self._folder_entry = tk.Entry(
            folder_frame, textvariable=self._folder_var,
            bg=BG_INPUT, fg=FG_MAIN, insertbackground=FG_MAIN,
            font=("Segoe UI", 10), relief="flat", bd=6
        )
        self._folder_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

        self._browse_btn = tk.Button(
            folder_frame, text="📁 Parcourir", command=self._browse,
            bg=BG_PANEL, fg=ACCENT, activebackground=BG_INPUT, activeforeground=ACCENT,
            font=("Segoe UI", 10), relief="flat", padx=10, cursor="hand2"
        )
        self._browse_btn.pack(side="left")

        self._recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            folder_frame, text="Sous-dossiers", variable=self._recursive_var,
            bg=BG_DARK, fg=FG_DIM, selectcolor=BG_INPUT,
            activebackground=BG_DARK, font=("Segoe UI", 10)
        ).pack(side="left", padx=(12, 0))

        # Corps principal : settings | resultats
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # Colonne gauche : settings
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both", padx=(0, 8))

        self._settings = SettingsPanel(left)
        self._settings.pack(fill="both", expand=False)

        # Boutons
        btn_frame = tk.Frame(left, bg=BG_DARK, pady=10)
        btn_frame.pack(fill="x")

        self._run_btn = tk.Button(
            btn_frame, text="▶  ANALYSER", command=self._start,
            bg=ACCENT, fg=BG_DARK, activebackground="#74a8f0", activeforeground=BG_DARK,
            font=("Segoe UI", 12, "bold"), relief="flat", padx=20, pady=8, cursor="hand2"
        )
        self._run_btn.pack(side="left", fill="x", expand=True)

        self._stop_btn = tk.Button(
            btn_frame, text="⏹  Arrêter", command=self._stop,
            bg=BG_PANEL, fg=RED, activebackground=BG_INPUT, activeforeground=RED,
            font=("Segoe UI", 11), relief="flat", padx=16, pady=8, cursor="hand2",
            state="disabled"
        )
        self._stop_btn.pack(side="left", padx=(8, 0))

        # Bouton feedback
        btn_frame2 = tk.Frame(left, bg=BG_DARK)
        btn_frame2.pack(fill="x")

        self._feedback_btn = tk.Button(
            btn_frame2, text="📊  Collecter feedback Lightroom",
            command=self._collect_feedback,
            bg=BG_PANEL, fg=PURPLE, activebackground=BG_INPUT, activeforeground=PURPLE,
            font=("Segoe UI", 10), relief="flat", padx=10, pady=6, cursor="hand2"
        )
        self._feedback_btn.pack(fill="x")

        tk.Label(
            left, text="↑ Après avoir corrigé les notes dans Lightroom",
            fg=FG_DIM, bg=BG_DARK, font=("Segoe UI", 8)
        ).pack(anchor="w", pady=(2, 0))

        # Colonne droite : resultats + log
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        self._results = ResultsPanel(right)
        self._results.pack(fill="x", pady=(0, 8))

        self._log = LogPanel(right)
        self._log.pack(fill="both", expand=True)

        # Barre de progression
        prog_frame = tk.Frame(self, bg=BG_DARK)
        prog_frame.pack(fill="x", padx=16, pady=(0, 12))

        self._prog_var = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(
            prog_frame, variable=self._prog_var, maximum=100, mode="determinate"
        )
        self._progress.pack(fill="x")

        self._prog_lbl_var = tk.StringVar(value="Prêt")
        tk.Label(
            prog_frame, textvariable=self._prog_lbl_var,
            fg=FG_DIM, bg=BG_DARK, font=("Segoe UI", 9)
        ).pack(anchor="e", pady=(2, 0))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self):
        folder = filedialog.askdirectory(title="Choisir le dossier de photos")
        if folder:
            self._folder_var.set(folder)

    def _collect_feedback(self):
        """Lit les XMP apres edition Lightroom et apprend des corrections."""
        folder_str = self._folder_var.get().strip()
        if not folder_str:
            messagebox.showwarning("Dossier manquant", "Choisissez le dossier que vous avez analysé.")
            return

        folder = Path(folder_str)
        if not folder.exists():
            messagebox.showerror("Erreur", f"Dossier introuvable : {folder_str}")
            return

        self._log.log("Collecte du feedback Lightroom…", "ok")

        # Relancer une analyse rapide pour avoir les features
        def do_feedback():
            try:
                # Scanner les photos
                all_files = [
                    f for f in (folder.rglob("*") if self._recursive_var.get() else folder.iterdir())
                    if f.suffix.lower() in SUPPORTED_EXTS
                ]

                photos = []
                for f in all_files:
                    p = analyze_photo(f, enable_faces=False)
                    if p is not None:
                        photos.append(p)

                feedback = collect_feedback(photos, folder)
                fb_path = save_feedback(feedback, folder)

                stats = feedback["stats"]
                self._queue.put(("log", (
                    f"Feedback collecté : {feedback['changed']} correction(s) détectée(s)", "ok"
                )))
                self._queue.put(("log", (
                    f"  ↑ {stats['promoted']} promue(s)  ↓ {stats['demoted']} rétrogradée(s)  = {stats['unchanged']} inchangée(s)", "dim"
                )))

                learned = load_learned()
                if "tendency" in learned:
                    self._queue.put(("log", (f"  → {learned['tendency']}", "ok")))
                if "blur_threshold_suggested" in learned:
                    self._queue.put(("log", (
                        f"  → Seuil de flou suggéré : {learned['blur_threshold_suggested']}", "ok"
                    )))

                self._queue.put(("log", (f"Données sauvegardées dans {fb_path.name}", "dim")))

            except Exception as e:
                self._queue.put(("log", (f"Erreur feedback : {e}", "error")))

        threading.Thread(target=do_feedback, daemon=True).start()

    def _start(self):
        folder_str = self._folder_var.get().strip()
        if not folder_str:
            messagebox.showwarning("Dossier manquant", "Veuillez choisir un dossier de photos.")
            return

        folder = Path(folder_str)
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Erreur", f"Dossier introuvable : {folder_str}")
            return

        # Verifier cle API si mode IA
        if self._settings.ai_rating.get():
            key = get_api_key()
            if not key:
                key = simpledialog.askstring(
                    "Clé API Anthropic",
                    "Entrez votre clé API Anthropic (sk-ant-...) :\n\n"
                    "Créez-en une sur console.anthropic.com → API Keys",
                    parent=self,
                )
                if key and key.startswith("sk-ant-"):
                    set_api_key(key)
                else:
                    messagebox.showwarning(
                        "Clé invalide",
                        "La clé doit commencer par sk-ant-. Tri IA désactivé."
                    )
                    self._settings.ai_rating.set(False)

        self._log.clear()
        self._results.reset()
        self._prog_var.set(0)
        self._running = True
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._browse_btn.config(state="disabled")

        # Copier les parametres (thread-safe)
        params = {
            "folder": folder,
            "recursive": self._recursive_var.get(),
            "blur_threshold": self._settings.blur_threshold.get(),
            "burst_gap": self._settings.burst_gap.get(),
            "enable_faces": self._settings.enable_faces.get(),
            "enable_duplicates": self._settings.enable_duplicates.get(),
            "auto_corrections": self._settings.auto_corrections.get(),
            "auto_crop": self._settings.auto_crop.get(),
            "ai_rating": self._settings.ai_rating.get(),
            "color_labels": self._settings.color_labels.get(),
            "generate_report": self._settings.generate_report.get(),
            "max_workers": self._settings.max_workers.get(),
        }

        self._worker_thread = threading.Thread(
            target=self._run_analysis, args=(params,), daemon=True
        )
        self._worker_thread.start()

    def _stop(self):
        self._running = False
        self._queue.put(("log", ("Arrêt demandé…", "warn")))

    def _on_done(self):
        self._running = False
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._browse_btn.config(state="normal")

    # ------------------------------------------------------------------
    # Worker (thread secondaire)
    # ------------------------------------------------------------------

    def _run_analysis(self, params: dict):
        """Tourne dans un thread secondaire. Communique via self._queue."""
        start_time = time.time()
        folder: Path = params["folder"]

        def log(msg, level="info"):
            self._queue.put(("log", (msg, level)))

        def progress(pct, msg=""):
            self._queue.put(("progress", (pct, msg)))

        def results(**kwargs):
            self._queue.put(("results", kwargs))

        try:
            # 1. Lister les fichiers
            log(f"Scan de {folder} …", "dim")
            if params["recursive"]:
                all_files = [
                    f for f in folder.rglob("*")
                    if f.suffix.lower() in SUPPORTED_EXTS
                ]
            else:
                all_files = [
                    f for f in folder.iterdir()
                    if f.suffix.lower() in SUPPORTED_EXTS
                ]

            all_files.sort(key=lambda f: f.name)
            total = len(all_files)

            if total == 0:
                log("Aucune photo trouvée dans ce dossier.", "warn")
                self._queue.put(("done", None))
                return

            log(f"{total} photo(s) trouvée(s). Analyse en cours…", "ok")
            results(total=total)

            # 2. Analyse parallele
            analyzed: List[PhotoAnalysis] = []
            n_done = 0
            n_errors = 0

            def analyze_one(path: Path) -> Optional[PhotoAnalysis]:
                if not self._running:
                    return None
                return analyze_photo(path, enable_faces=params["enable_faces"])

            with ThreadPoolExecutor(max_workers=params["max_workers"]) as executor:
                futures = {executor.submit(analyze_one, f): f for f in all_files}

                for future in as_completed(futures):
                    if not self._running:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    n_done += 1
                    pct = (n_done / total) * 60  # 0-60% pour l'analyse

                    try:
                        result = future.result()
                        if result is not None:
                            analyzed.append(result)
                        else:
                            n_errors += 1
                    except Exception as e:
                        n_errors += 1
                        log(f"Erreur sur {futures[future].name}: {e}", "error")

                    progress(pct, f"Analyse {n_done}/{total} — {futures[future].name}")

            if not self._running:
                log("Analyse interrompue.", "warn")
                self._queue.put(("done", None))
                return

            log(f"Analyse terminée : {len(analyzed)} OK, {n_errors} erreur(s).", "ok")
            results(analyzed=len(analyzed))

            # 3. Detection de rafales
            log("Groupement des rafales…", "dim")
            progress(62, "Groupement des rafales…")
            group_bursts(analyzed, gap_seconds=params["burst_gap"])

            burst_groups = {p.burst_group for p in analyzed if p.burst_group}
            n_bursts = len(burst_groups)
            log(f"{n_bursts} rafale(s) détectée(s).", "ok")
            results(bursts=n_bursts)

            # 4. Detection de doublons
            n_dupes = 0
            if params["enable_duplicates"]:
                log("Détection des doublons (hash perceptuel)…", "dim")
                progress(68, "Détection des doublons…")
                before_rejects = sum(1 for p in analyzed if p.rating == -1)
                detect_duplicates(analyzed)
                after_rejects = sum(1 for p in analyzed if p.rating == -1)
                n_dupes = after_rejects - before_rejects
                log(f"{n_dupes} doublon(s) supprimé(s).", "ok")
                results(dupes=n_dupes)

            # 5. Application du seuil de flou (sauf si mode IA)
            if not params["ai_rating"]:
                blur_thresh = params["blur_threshold"]
                n_blur_rejected = 0
                for p in analyzed:
                    if p.rating != -1 and p.blur_score < blur_thresh:
                        p.rating = -1
                        n_blur_rejected += 1
                if n_blur_rejected:
                    log(f"{n_blur_rejected} photo(s) rejetée(s) pour flou (seuil={blur_thresh}).", "warn")

            # 5b. Appliquer les preferences apprises (feedback precedent)
            learned = load_learned()
            if learned:
                n_adj = apply_learned_adjustments(analyzed, learned)
                if n_adj:
                    log(f"Preferences apprises appliquées : {n_adj} photo(s) ajustée(s).", "ok")
                    tendency = learned.get("tendency", "")
                    if tendency:
                        log(f"  → {tendency}", "dim")

            # 5c. Tri IA (Claude Vision) — remplace les notes heuristiques
            if params["ai_rating"]:
                api_key = get_api_key()
                if not api_key:
                    log("Clé API Anthropic manquante. Tri IA désactivé.", "error")
                else:
                    try:
                        from core.ai_rater import AIRater, apply_ai_ratings
                        log("Tri IA en cours (Claude Vision)… Cela peut prendre quelques minutes.", "ok")
                        progress(65, "Tri IA en cours…")

                        rater = AIRater(api_key=api_key)

                        def ai_callback(done, total_ai, name):
                            pct = 65 + (done / total_ai) * 15
                            progress(pct, f"IA {done}/{total_ai} — {name}")
                            log(f"  IA : {done}/{total_ai} traitées", "dim")

                        ai_ratings = rater.rate_batch(analyzed, callback=ai_callback)
                        apply_ai_ratings(analyzed, ai_ratings)

                        n_ai_rejects = sum(1 for p in analyzed if p.rating == -1)
                        n_ai_picks = sum(1 for p in analyzed if p.rating >= 1)
                        log(f"Tri IA terminé : {n_ai_picks} sélectionnées / {n_ai_rejects} rejetées.", "ok")
                    except Exception as e:
                        log(f"Erreur tri IA : {e}. Fallback sur les notes heuristiques.", "error")

            # 6. Recadrage auto (chargement image necessaire)
            n_cropped = 0
            crop_data = {}  # path -> CropResult

            if params["auto_crop"]:
                log("Recadrage automatique…", "dim")
                progress(72, "Recadrage automatique…")
                from core.photo_analyzer import load_preview
                for i, photo in enumerate(analyzed):
                    if not self._running:
                        break
                    if photo.rating == -1:
                        continue
                    try:
                        img = load_preview(photo.path)
                        if img is not None:
                            crop_result = auto_crop(
                                img, photo.face_rects,
                                enable_face_crop=True,
                                enable_straighten=True,
                            )
                            if crop_result.has_crop:
                                crop_data[photo.path] = crop_result
                                n_cropped += 1
                    except Exception:
                        pass
                log(f"{n_cropped} photo(s) recadrée(s).", "ok")

            # 7. Calcul des corrections par photo
            log("Calcul des corrections…", "dim")
            progress(78, "Calcul des corrections…")

            all_corrections = {}
            for photo in analyzed:
                c = build_corrections(photo, auto_color=params["auto_corrections"])
                all_corrections[str(photo.path)] = c

            # 8. Harmonisation balance des blancs par sequence
            if params["auto_corrections"]:
                log("Harmonisation balance des blancs par séquence…", "dim")
                progress(82, "Harmonisation WB…")
                harmonize_white_balance(analyzed, all_corrections, gap_seconds=30.0)
                log("Balance des blancs harmonisée.", "ok")

            # 9. Ecriture XMP (backup + keywords + corrections + crop)
            log("Écriture des fichiers XMP…", "dim")
            progress(85, "Écriture des fichiers XMP…")

            n_xmp = 0
            n_picks = 0
            n_rejects = 0
            n_backups = 0

            for i, photo in enumerate(analyzed):
                if not self._running:
                    break

                corrections = all_corrections.get(str(photo.path), build_corrections(photo))
                crop = crop_data.get(photo.path)
                keywords = generate_keywords(photo)

                # Backup de l'ancien XMP si existant
                bak = photo.path.with_suffix(".xmp")
                if bak.exists():
                    n_backups += 1

                write_xmp(
                    photo, corrections,
                    crop=crop,
                    keywords=keywords,
                    use_color_labels=params["color_labels"],
                    do_backup=True,
                )
                n_xmp += 1

                if photo.rating == -1:
                    n_rejects += 1
                else:
                    n_picks += 1

                pct = 85 + (i / len(analyzed)) * 12
                progress(pct, f"XMP {i+1}/{len(analyzed)} — {photo.path.name}")

            log(f"{n_xmp} fichier(s) XMP écrit(s).", "ok")
            if n_backups:
                log(f"{n_backups} ancien(s) XMP sauvegardé(s) en .xmp.bak", "dim")
            results(picks=n_picks, rejects=n_rejects)

            # 7. Rapport CSV
            if params["generate_report"] and analyzed:
                report_path = folder / "lightroom_accelerator_report.csv"
                try:
                    write_report(analyzed, report_path)
                    log(f"Rapport CSV : {report_path.name}", "ok")
                except Exception as e:
                    log(f"Impossible d'écrire le rapport : {e}", "error")

            # Temps total
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            results(time=elapsed_str)

            progress(100, f"Terminé en {elapsed_str}")
            log("=" * 55, "dim")
            log(f"  Résultat final : {n_picks} sélectionnées / {n_rejects} rejetées", "ok")
            log(f"  XMP prêts. Dans Lightroom : Ctrl+A → Lire les métadonnées", "ok")
            log("=" * 55, "dim")

        except Exception as e:
            log(f"Erreur inattendue : {e}", "error")
            import traceback
            log(traceback.format_exc(), "error")

        finally:
            self._queue.put(("done", None))

    # ------------------------------------------------------------------
    # Polling queue (thread principal)
    # ------------------------------------------------------------------

    def _poll_queue(self):
        """Traite les messages envoyes par le worker thread."""
        try:
            while True:
                msg_type, payload = self._queue.get_nowait()

                if msg_type == "log":
                    message, level = payload
                    self._log.log(message, level)

                elif msg_type == "progress":
                    pct, msg = payload
                    self._prog_var.set(pct)
                    if msg:
                        self._prog_lbl_var.set(msg)

                elif msg_type == "results":
                    self._results.update(**payload)

                elif msg_type == "done":
                    self._on_done()

        except queue.Empty:
            pass

        # Re-planifier le polling dans 100ms
        self.after(100, self._poll_queue)


# ---------------------------------------------------------------------------
# Entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()

    # Style ttk
    style = ttk.Style(app)
    try:
        style.theme_use("vista")   # Windows natif
    except Exception:
        pass

    style.configure("TProgressbar",
                    troughcolor=BG_INPUT,
                    background=ACCENT,
                    bordercolor=BG_PANEL,
                    lightcolor=ACCENT,
                    darkcolor=ACCENT)

    app.mainloop()
