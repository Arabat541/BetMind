# ============================================================
# ensemble_model.py — Classe EnsembleModel (XGB + LGB + meta LR)
# Importé par train_from_csv.py (entraînement) et predictor.py (inférence)
# ============================================================

import numpy as np
import pickle


class EnsembleModel:
    """
    Stacking : XGBoost + LightGBM comme base-learners,
    Logistic Regression comme méta-modèle.

    Interface sklearn : predict_proba(X) → probas [H, D, A]
    """

    def __init__(self, xgb_model=None, lgb_model=None, meta_model=None,
                 n_classes: int = 3):
        self.xgb_model  = xgb_model
        self.lgb_model  = lgb_model
        self.meta_model = meta_model
        self.n_classes  = n_classes
        self.classes_   = list(range(n_classes))

    def _stack(self, X) -> np.ndarray:
        """Concatène les probas XGB + LGB → méta-features."""
        p_xgb = self.xgb_model.predict_proba(X)   # (n, 3)
        p_lgb = self.lgb_model.predict_proba(X)   # (n, 3)
        return np.hstack([p_xgb, p_lgb])           # (n, 6)

    def predict_proba(self, X) -> np.ndarray:
        meta_X = self._stack(X)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "EnsembleModel":
        with open(path, "rb") as f:
            return pickle.load(f)
