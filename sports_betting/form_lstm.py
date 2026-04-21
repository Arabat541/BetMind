# ============================================================
# form_lstm.py — AJ : Extracteur de forme par séquences temporelles
#
# Architecture : LSTM(32) → Dense(1, sigmoid) si Keras disponible.
# Fallback numpy : moyenne exponentielle + pente linéaire (aucune
# dépendance lourde requise).
#
# Chaque équipe est représentée par sa séquence des SEQ_LEN derniers
# matchs avant la date cible. Chaque pas de temps encode :
#   [win, draw, loss, goals_for/5, goals_against/5, was_home, xg/3, xga/3]
#
# Sortie : float [0, 1] — "form quality score" (proche de 1 = grande forme)
# ============================================================

import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

SEQ_LEN    = 10   # matchs dans la fenêtre
N_FEATURES = 8    # features par pas de temps
LSTM_PATH  = os.path.join(os.path.dirname(__file__), "models", "football_lstm_form.pkl")

try:
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
except Exception:
    pass

# Tentative d'import Keras
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow import keras
    KERAS_OK = True
except ImportError:
    KERAS_OK = False


# ════════════════════════════════════════════════════════════
# ENCODAGE D'UNE SÉQUENCE
# ════════════════════════════════════════════════════════════

def _encode_match(result: float, goals_for: int, goals_against: int,
                  was_home: int, xg: float = 0.0, xga: float = 0.0) -> np.ndarray:
    """
    Encode un match en vecteur de 8 features.
    result : 1=win, 0.5=draw, 0=loss (du point de vue de l'équipe)
    """
    return np.array([
        float(result == 1.0),           # win
        float(result == 0.5),           # draw
        float(result == 0.0),           # loss
        min(goals_for / 5.0, 1.0),
        min(goals_against / 5.0, 1.0),
        float(was_home),
        min(xg / 3.0, 1.0),
        min(xga / 3.0, 1.0),
    ], dtype=np.float32)


def build_team_sequence(matches: list[tuple]) -> np.ndarray:
    """
    Construit un tenseur (SEQ_LEN, N_FEATURES) depuis une liste
    de tuples (result, goals_for, goals_against, was_home, xg, xga).
    Si moins de SEQ_LEN matchs, le début est complété par des zéros (pad).
    """
    seq = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    recent = matches[-SEQ_LEN:]          # garder les plus récents
    offset = SEQ_LEN - len(recent)
    for i, m in enumerate(recent):
        r, gf, ga, home, xg, xga = m[0], m[1], m[2], m[3], m[4] if len(m) > 4 else 0, m[5] if len(m) > 5 else 0
        seq[offset + i] = _encode_match(r, gf, ga, home, xg, xga)
    return seq


# ════════════════════════════════════════════════════════════
# MODÈLE KERAS LSTM
# ════════════════════════════════════════════════════════════

def _build_keras_model() -> "keras.Model":
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mae"])
    return model


# ════════════════════════════════════════════════════════════
# FALLBACK NUMPY — FORME EXPONENTIELLE + TENDANCE
# ════════════════════════════════════════════════════════════

def _numpy_form_score(sequences: np.ndarray) -> np.ndarray:
    """
    Pour chaque séquence (SEQ_LEN, N_FEATURES), calcule un score de forme
    basé sur :
      1. Moyenne pondérée exponentiellement des résultats (dim 0, 1, 2)
      2. Pente linéaire des résultats récents (tendance positive = montée en forme)
    Retourne un vecteur de scores [0, 1].
    """
    n = sequences.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    weights = np.exp(np.linspace(-2, 0, SEQ_LEN))   # plus récent = poids élevé
    weights /= weights.sum()

    for i in range(n):
        seq = sequences[i]                           # (SEQ_LEN, N_FEATURES)
        # Résultat encodé : win=1, draw=0.5, loss=0
        results = seq[:, 0] * 1.0 + seq[:, 1] * 0.5   # [0, 1]
        # Moyenne pondérée
        ewm = float(np.dot(weights, results))
        # Tendance linéaire (slope normalisée)
        x = np.arange(SEQ_LEN, dtype=np.float32)
        if results.std() > 0:
            slope = float(np.polyfit(x, results, 1)[0]) / SEQ_LEN
        else:
            slope = 0.0
        # Score final : 70% EWM + 30% tendance (clampé 0-1)
        score = np.clip(ewm * 0.70 + (slope + 0.5) * 0.30, 0.0, 1.0)
        scores[i] = score
    return scores


# ════════════════════════════════════════════════════════════
# WRAPPER UNIFIÉ
# ════════════════════════════════════════════════════════════

class FormLSTM:
    """
    Extracteur de forme par séquences.
    Utilise Keras LSTM si disponible, sinon numpy fallback.
    """

    def __init__(self):
        self._keras_model = None
        self._is_trained  = False
        self._use_keras   = KERAS_OK
        self._load_if_exists()

    def _load_if_exists(self):
        if not os.path.exists(LSTM_PATH):
            return
        try:
            if KERAS_OK:
                keras_path = LSTM_PATH.replace(".pkl", ".keras")
                if os.path.exists(keras_path):
                    self._keras_model = keras.models.load_model(keras_path)
                    self._is_trained  = True
                    logger.info(f"FormLSTM Keras chargé : {keras_path}")
                    return
            # Fallback : pas de keras model → le numpy fallback n'a rien à charger
            self._is_trained = False
        except Exception as e:
            logger.warning(f"FormLSTM load failed : {e}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        sequences : (N, SEQ_LEN, N_FEATURES)
        Retourne (N,) de scores [0, 1].
        """
        if self._use_keras and self._is_trained and self._keras_model is not None:
            return self._keras_model.predict(sequences, verbose=0).flatten()
        return _numpy_form_score(sequences)

    def predict_one(self, matches: list[tuple]) -> float:
        """
        matches : liste de tuples (result, gf, ga, home, xg, xga)
        Retourne un score [0, 1].
        """
        seq = build_team_sequence(matches)[np.newaxis, :]   # (1, SEQ_LEN, N_FEATURES)
        return float(self.predict(seq)[0])

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20):
        """
        X : (N, SEQ_LEN, N_FEATURES)
        y : (N,) — targets binaires ou {0, 0.5, 1}
        """
        if not self._use_keras:
            logger.info("Keras non disponible — FormLSTM utilise le fallback numpy (pas d'entraînement requis).")
            self._is_trained = True
            return

        logger.info(f"FormLSTM — entraînement LSTM(32) sur {len(X)} séquences...")
        model = _build_keras_model()
        y_bin = (y >= 0.75).astype(np.float32)   # win = 1, reste = 0
        model.fit(
            X, y_bin,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            verbose=0,
        )
        self._keras_model = model
        self._is_trained  = True

        keras_path = LSTM_PATH.replace(".pkl", ".keras")
        model.save(keras_path)
        logger.info(f"FormLSTM sauvegardé : {keras_path}")


# ════════════════════════════════════════════════════════════
# CONSTRUCTION DES SÉQUENCES DEPUIS CSV FOOTBALL-DATA
# ════════════════════════════════════════════════════════════

def build_sequences_from_df(df) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit les paires (séquence, résultat_suivant) depuis un DataFrame CSV.
    df colonnes requises : Date, HomeTeam, AwayTeam, FTHG, FTAG
    Retourne (X, y) pour l'entraînement.
    """
    import pandas as pd

    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if not all(c in df.columns for c in needed):
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.array([])

    df = df.dropna(subset=needed).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Dictionnaire : {team_name: [(date, result, gf, ga, home)]}
    team_history: dict[str, list] = {}

    def _add(team, date, result, gf, ga, home):
        team_history.setdefault(team, []).append((date, result, int(gf), int(ga), int(home), 0.0, 0.0))

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        date   = row["Date"]
        if hg > ag:
            _add(ht, date, 1.0, hg, ag, 1)
            _add(at, date, 0.0, ag, hg, 0)
        elif hg < ag:
            _add(ht, date, 0.0, hg, ag, 1)
            _add(at, date, 1.0, ag, hg, 0)
        else:
            _add(ht, date, 0.5, hg, ag, 1)
            _add(at, date, 0.5, ag, hg, 0)

    X_list, y_list = [], []
    for team, history in team_history.items():
        history.sort(key=lambda x: x[0])
        for i in range(SEQ_LEN, len(history)):
            past   = history[:i]
            target = history[i]
            seq    = build_team_sequence([(m[1], m[2], m[3], m[4], m[5], m[6]) for m in past])
            X_list.append(seq)
            y_list.append(target[1])   # résultat suivant

    if not X_list:
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ════════════════════════════════════════════════════════════
# INFÉRENCE — SCORE DE FORME POUR UNE ÉQUIPE
# ════════════════════════════════════════════════════════════

_lstm_singleton: FormLSTM | None = None


def _get_lstm() -> FormLSTM:
    global _lstm_singleton
    if _lstm_singleton is None:
        _lstm_singleton = FormLSTM()
    return _lstm_singleton


def get_lstm_form(team_name: str, xg_history: dict | None = None,
                  match_history: list | None = None) -> float:
    """
    Retourne le score de forme LSTM [0, 1] pour une équipe.

    match_history : liste de tuples (result, gf, ga, was_home, xg, xga)
                    ordonnée du plus ancien au plus récent.
    xg_history    : optionnel — dict Understat pour enrichir.
    """
    if not match_history:
        return 0.5   # neutre si aucune donnée

    lstm = _get_lstm()
    return lstm.predict_one(match_history)


# ════════════════════════════════════════════════════════════
# ENTRAÎNEMENT STANDALONE (appelé par train_from_csv.py)
# ════════════════════════════════════════════════════════════

def train_lstm_from_df(df, epochs: int = 20) -> FormLSTM:
    """
    Entraîne le LSTM depuis un DataFrame CSV et le sauvegarde.
    Retourne l'instance entraînée.
    """
    X, y = build_sequences_from_df(df)
    if len(X) == 0:
        logger.warning("FormLSTM : aucune séquence construite — skip entraînement.")
        return _get_lstm()

    lstm = FormLSTM()
    lstm.train(X, y, epochs=epochs)
    global _lstm_singleton
    _lstm_singleton = lstm
    logger.info(f"FormLSTM entraîné sur {len(X)} séquences.")
    return lstm


if __name__ == "__main__":
    # Test rapide
    matches = [
        (1.0, 2, 0, 1, 1.8, 0.7),
        (0.5, 1, 1, 0, 1.1, 1.2),
        (1.0, 3, 1, 1, 2.1, 0.9),
        (1.0, 2, 0, 1, 1.6, 0.4),
        (0.0, 0, 2, 0, 0.8, 1.9),
    ]
    score = get_lstm_form("TestTeam", match_history=matches)
    print(f"Form score: {score:.4f}")
