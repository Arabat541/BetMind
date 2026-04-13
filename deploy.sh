#!/usr/bin/env bash
# ============================================================
# deploy.sh — Déploiement BetMind vers le VPS
# Usage : ./deploy.sh [--restart] [fichier1.py fichier2.py ...]
#   --restart  : redémarre le service betmind après le déploiement
#   Sans args  : déploie tous les fichiers modifiés depuis le dernier commit
# Exemples :
#   ./deploy.sh                        # fichiers git modifiés + restart auto
#   ./deploy.sh config.py bankroll.py  # fichiers spécifiques
#   ./deploy.sh --no-restart           # sans redémarrage du service
# ============================================================

set -euo pipefail

VPS_HOST="evrard@46.224.165.35"
VPS_DIR="/home/evrard/apps/betmind"
LOCAL_DIR="$(cd "$(dirname "$0")/sports_betting" && pwd)"
SERVICE="betmind"
RESTART=true

# ── Parse arguments ─────────────────────────────────────────
FILES=()
for arg in "$@"; do
    case "$arg" in
        --no-restart) RESTART=false ;;
        --restart)    RESTART=true  ;;
        *)            FILES+=("$arg") ;;
    esac
done

# ── Si aucun fichier spécifié : fichiers git modifiés ────────
if [ ${#FILES[@]} -eq 0 ]; then
    mapfile -t FILES < <(git -C "$LOCAL_DIR/.." diff --name-only HEAD -- sports_betting/ \
        | sed 's|sports_betting/||' \
        | grep '\.py$' || true)

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "Aucun fichier .py modifié depuis le dernier commit."
        echo "Utilise : ./deploy.sh fichier.py  pour déployer un fichier spécifique."
        exit 0
    fi
fi

# ── Déploiement ──────────────────────────────────────────────
echo "╔══════════════════════════════════════╗"
echo "  BetMind Deploy → $VPS_HOST"
echo "╚══════════════════════════════════════╝"

DEPLOYED=0
FAILED=0

for f in "${FILES[@]}"; do
    local_path="$LOCAL_DIR/$f"
    if [ ! -f "$local_path" ]; then
        echo "  ⚠  Fichier introuvable localement : $f"
        ((FAILED++)) || true
        continue
    fi
    echo -n "  → $f ... "
    if scp -q "$local_path" "$VPS_HOST:$VPS_DIR/$f"; then
        echo "OK"
        ((DEPLOYED++)) || true
    else
        echo "ERREUR"
        ((FAILED++)) || true
    fi
done

echo ""
echo "  Déployés : $DEPLOYED  |  Échecs : $FAILED"

# ── Vérification syntaxe Python sur le VPS ───────────────────
echo ""
echo "  Vérification syntaxe..."
SYNTAX_OK=true
for f in "${FILES[@]}"; do
    result=$(ssh "$VPS_HOST" "cd $VPS_DIR && venv/bin/python -m py_compile $f 2>&1" || true)
    if [ -n "$result" ]; then
        echo "  ✗ Erreur syntaxe dans $f :"
        echo "    $result"
        SYNTAX_OK=false
    fi
done
if $SYNTAX_OK; then
    echo "  Syntaxe OK"
fi

# ── Redémarrage service ──────────────────────────────────────
if $RESTART && $SYNTAX_OK; then
    echo ""
    echo "  Redémarrage du service $SERVICE..."
    ssh "$VPS_HOST" "sudo systemctl restart $SERVICE" && echo "  Service redémarré." || \
        echo "  ⚠  Redémarrage manuel requis : sudo systemctl restart $SERVICE"
    sleep 2
    STATUS=$(ssh "$VPS_HOST" "systemctl is-active $SERVICE 2>/dev/null" || echo "unknown")
    echo "  Statut : $STATUS"
elif ! $SYNTAX_OK; then
    echo ""
    echo "  ⚠  Redémarrage annulé (erreur syntaxe). Corrige avant de redémarrer."
fi

echo ""
echo "  Terminé."
