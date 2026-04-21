# ============================================================
# ensemble_model.py — Classe EnsembleModel (XGB + LGB + meta LR)
# Importé par train_from_csv.py (entraînement) et predictor.py (inférence)
# ============================================================

import numpy as np
import pickle


class EnsembleModel:
    """
    Stacking : XGBoost + LightGBM comme base-learners,
    Logistic Regression comme méta-modèle,
    + calibration isotonic par classe (Platt scaling multiclass).

    Interface sklearn : predict_proba(X) → probas [H, D, A]
    """

    def __init__(self, xgb_model=None, lgb_model=None, meta_model=None,
                 n_classes: int = 3):
        self.xgb_model  = xgb_model
        self.lgb_model  = lgb_model
        self.meta_model = meta_model
        self.n_classes  = n_classes
        self.classes_   = list(range(n_classes))
        self.cal_models = None   # list[IsotonicRegression] par classe, None = pas de calibration

    def _stack(self, X) -> np.ndarray:
        """Concatène les probas XGB + LGB → méta-features."""
        p_xgb = self.xgb_model.predict_proba(X)   # (n, 3)
        p_lgb = self.lgb_model.predict_proba(X)   # (n, 3)
        return np.hstack([p_xgb, p_lgb])           # (n, 6)

    def predict_proba(self, X) -> np.ndarray:
        meta_X = self._stack(X)
        probas = self.meta_model.predict_proba(meta_X)   # (n, 3)

        if self.cal_models is not None:
            probas = self._apply_calibration(probas)

        return probas

    def predict_proba_with_variance(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne (probas, variance) où variance est l'écart-type max entre
        les deux base-learners (XGB vs LGB) par outcome.
        variance[i] = max std across outcomes for sample i.
        Variance élevée (> 0.08) = désaccord entre modèles → signal peu fiable.
        """
        p_xgb = self.xgb_model.predict_proba(X)   # (n, 3)
        p_lgb = self.lgb_model.predict_proba(X)    # (n, 3)

        stacked = np.stack([p_xgb, p_lgb], axis=0)  # (2, n, 3)
        std_per_outcome = stacked.std(axis=0)        # (n, 3)
        max_std = std_per_outcome.max(axis=1)        # (n,) — pire désaccord par sample

        # Probas finales via méta-modèle + calibration
        meta_X = np.hstack([p_xgb, p_lgb])
        probas = self.meta_model.predict_proba(meta_X)
        if self.cal_models is not None:
            probas = self._apply_calibration(probas)

        return probas, max_std

    def _apply_calibration(self, probas: np.ndarray) -> np.ndarray:
        """Applique la calibration isotonic par classe puis renormalise."""
        calibrated = np.column_stack([
            self.cal_models[c].predict(probas[:, c])
            for c in range(self.n_classes)
        ])
        # Clamp et renormalise pour que les probas somment à 1
        calibrated = np.clip(calibrated, 1e-6, 1.0)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated

    def calibrate(self, oof_probas: np.ndarray, y: np.ndarray) -> None:
        """
        Ajuste une IsotonicRegression par classe sur les probas OOF.
        oof_probas : (n_samples, n_classes) — sortie brute du méta-modèle sur OOF
        y          : (n_samples,) — labels vrais (entiers 0/1/2)
        """
        from sklearn.isotonic import IsotonicRegression
        self.cal_models = []
        for c in range(self.n_classes):
            y_bin = (y == c).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(oof_probas[:, c], y_bin)
            self.cal_models.append(ir)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "EnsembleModel":
        with open(path, "rb") as f:
            return pickle.load(f)
