#!/usr/bin/env bash
# ============================================================
# retrain.sh — Retrain hebdomadaire BetMind (cron VPS)
# Cron suggéré : 0 3 * * 1  (tous les lundis à 3h UTC)
# Usage : ./retrain.sh [--no-notify]
# ============================================================

set -euo pipefail

BETMIND_DIR="/home/evrard/apps/betmind"
PYTHON="$BETMIND_DIR/venv/bin/python3"
LOGS_DIR="$BETMIND_DIR/logs"
LOG_FILE="$LOGS_DIR/retrain_$(date +%Y%m%d_%H%M%S).log"
NOTIFY=true

for arg in "$@"; do
    [ "$arg" = "--no-notify" ] && NOTIFY=false
done

# ── Rotation logs (garde les 5 derniers retrains) ─────────────
ls -t "$LOGS_DIR"/retrain_*.log 2>/dev/null | tail -n +6 | xargs rm -f || true

exec > >(tee -a "$LOG_FILE") 2>&1

echo "════════════════════════════════════════════"
echo "  BetMind Retrain — $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════"

cd "$BETMIND_DIR"

send_telegram() {
    local msg="$1"
    # Charge TOKEN et CHAT_ID depuis .env
    if [ -f "$BETMIND_DIR/.env" ]; then
        TOKEN=$(grep TELEGRAM_TOKEN "$BETMIND_DIR/.env" | cut -d= -f2 | tr -d ' "')
        CHAT=$(grep TELEGRAM_CHAT_ID "$BETMIND_DIR/.env" | cut -d= -f2 | tr -d ' "')
        if [ -n "$TOKEN" ] && [ -n "$CHAT" ]; then
            curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
                -d "chat_id=${CHAT}" \
                -d "text=${msg}" \
                -d "parse_mode=Markdown" > /dev/null
        fi
    fi
}

$NOTIFY && send_telegram "🔄 *BetMind Retrain démarré* — $(date '+%Y-%m-%d %H:%M')"

STATUS=0

# ── 1. Football (1X2 + ensemble + SHAP) ─────────────────────
echo ""
echo "[ 1/3 ] Football 1X2 + Ensemble..."
if $PYTHON train_from_csv.py; then
    echo "  ✓ Football OK"
else
    echo "  ✗ Football ERREUR (code $?)"
    STATUS=1
fi

# ── 2. Over/Under ───────────────────────────────────────────
echo ""
echo "[ 2/3 ] Over/Under..."
if $PYTHON train_over_under.py; then
    echo "  ✓ Over/Under OK"
else
    echo "  ✗ Over/Under ERREUR"
    STATUS=1
fi

# ── 3. BTTS ─────────────────────────────────────────────────
echo ""
echo "[ 3/3 ] BTTS..."
if $PYTHON train_btts.py; then
    echo "  ✓ BTTS OK"
else
    echo "  ✗ BTTS ERREUR"
    STATUS=1
fi

echo ""
echo "════════════════════════════════════════════"
if [ $STATUS -eq 0 ]; then
    echo "  Retrain terminé avec succès — $(date '+%H:%M:%S')"
    $NOTIFY && send_telegram "✅ *BetMind Retrain terminé* — tous les modèles mis à jour\n📊 Log : \`$LOG_FILE\`"
else
    echo "  Retrain terminé avec des erreurs — voir $LOG_FILE"
    $NOTIFY && send_telegram "⚠️ *BetMind Retrain : erreurs détectées*\nVérifier \`$LOG_FILE\`"
fi
echo "════════════════════════════════════════════"

exit $STATUS
