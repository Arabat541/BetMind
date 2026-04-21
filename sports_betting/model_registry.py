# ============================================================
# model_registry.py — Suivi des retrains (MLflow ou fallback JSON)
# Enregistre accuracy / log-loss / n_samples à chaque retrain.
# Détecte une dégradation et alerte via Telegram si accuracy
# baisse de plus de DEGRADATION_THRESHOLD entre deux runs.
# ============================================================

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DEGRADATION_THRESHOLD = 0.02   # -2 pts accuracy → alerte
REGISTRY_FILE = os.path.join("data", "model_registry.json")

# ── Tentative MLflow (optionnel) ─────────────────────────────
try:
    import mlflow
    _MLFLOW_OK = True
except ImportError:
    _MLFLOW_OK = False


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_registry(reg: dict):
    os.makedirs("data", exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)


def log_run(model_name: str, metrics: dict, params: dict | None = None) -> dict:
    """
    Enregistre un run de retrain.

    model_name : "football_1x2" | "over_under" | "btts"
    metrics    : {"accuracy": float, "log_loss": float, "n_samples": int, ...}
    params     : hyperparamètres optionnels

    Retourne {"degraded": bool, "delta": float, "prev_accuracy": float}
    """
    params = params or {}
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ── MLflow tracking (si installé) ────────────────────────
    if _MLFLOW_OK:
        try:
            mlflow.set_tracking_uri(os.path.join("data", "mlruns"))
            mlflow.set_experiment(f"betmind_{model_name}")
            with mlflow.start_run():
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))})
                if params:
                    mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as e:
            logger.warning(f"MLflow log failed: {e}")

    # ── Fallback JSON registry ───────────────────────────────
    reg = _load_registry()
    history = reg.get(model_name, [])

    entry = {"ts": ts, **{k: round(float(v), 6) if isinstance(v, float) else v
                          for k, v in metrics.items()}}
    history.append(entry)
    reg[model_name] = history[-50:]   # garde les 50 derniers runs
    _save_registry(reg)

    # ── Détection dégradation ────────────────────────────────
    result = {"degraded": False, "delta": 0.0, "prev_accuracy": None}
    if len(history) >= 2:
        cur_acc  = metrics.get("accuracy", 0.0)
        prev_acc = history[-2].get("accuracy", 0.0)
        delta    = cur_acc - prev_acc
        result   = {"degraded": delta < -DEGRADATION_THRESHOLD,
                    "delta": round(delta, 4),
                    "prev_accuracy": round(prev_acc, 4)}
        if result["degraded"]:
            _alert_degradation(model_name, prev_acc, cur_acc, delta)

    logger.info(
        f"[registry] {model_name} — accuracy={metrics.get('accuracy', '?'):.3f} "
        f"log_loss={metrics.get('log_loss', '?'):.4f} "
        f"n_samples={metrics.get('n_samples', '?')}"
    )
    return result


def _alert_degradation(model_name: str, prev: float, cur: float, delta: float):
    msg = (
        f"⚠️ <b>Dégradation modèle BetMind</b>\n"
        f"Modèle : <code>{model_name}</code>\n"
        f"Accuracy : {prev:.1%} → {cur:.1%} "
        f"(<b>{delta:+.1%}</b>)\n"
        f"Seuil alerte : -{DEGRADATION_THRESHOLD:.0%} — vérifier les données sources."
    )
    try:
        from telegram_bot import send_message
        send_message(msg)
    except Exception as e:
        logger.warning(f"Telegram degradation alert failed: {e}")


def get_history(model_name: str) -> list:
    return _load_registry().get(model_name, [])


def get_all_history() -> dict:
    return _load_registry()
