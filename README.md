# BetMind — Agent de Prédiction Paris Sportifs

> Même architecture que le bot de trading : données → features → ML → signal + Kelly Criterion

## Stack
- **Football** : API-Football (RapidAPI) — 100 req/jour gratuit
- **NBA** : BallDontLie API — illimité, sans clé
- **Cotes** : The Odds API — 500 req/mois gratuit
- **Modèle** : XGBoost + Distribution de Poisson (fallback)
- **Bankroll** : Kelly Criterion fractionnaire
- **Alertes** : Telegram
- **Dashboard** : Flask

---

## Installation

```bash
cd sports_betting
pip install -r requirements.txt --break-system-packages

cp .env.example .env
# → Remplis les clés API dans .env
```

---

## Clés API à obtenir (toutes gratuites)

| API | Lien | Tier gratuit |
|-----|------|-------------|
| API-Football | https://rapidapi.com/api-sports/api/api-football | 100 req/jour |
| The Odds API | https://the-odds-api.com | 500 req/mois |
| Telegram Bot | @BotFather sur Telegram | illimité |
| BallDontLie | https://app.balldontlie.io | illimité |

---

## Utilisation

### Lancement manuel (1 cycle)
```bash
python predictor.py
```

### Lancement avec scheduler automatique (08h00, 18h00, 22h00)
```bash
python predictor.py --schedule
```

### Dashboard Flask
```bash
python dashboard.py
# → http://localhost:5001
```

### Dashboard + Agent en parallèle (VPS)
```bash
# Terminal 1
python predictor.py --schedule

# Terminal 2
python dashboard.py
```

---

## Structure

```
sports_betting/
├── config.py               ← Configuration centrale
├── data_fetcher.py         ← Collecte APIs + SQLite
├── feature_engineering.py  ← Construction features ML
├── model.py                ← XGBoost + Poisson + Value Bet detector
├── bankroll.py             ← Kelly Criterion + tracking ROI
├── predictor.py            ← Pipeline principal + scheduler
├── telegram_bot.py         ← Alertes Telegram
├── dashboard.py            ← Flask dashboard
├── requirements.txt
├── .env.example
├── data/
│   └── betting.db          ← SQLite (auto-créé)
├── models/
│   ├── football_xgb_model.pkl  (généré après entraînement)
│   └── nba_xgb_model.pkl
└── logs/
    └── predictor.log
```

---

## Logique de prédiction

### Phase 1 — Sans données historiques (démarrage)
Le modèle utilise des **priors** :
- **Football** : Distribution de Poisson bivariée (λ_home, λ_away calculés depuis les stats d'équipe)
- **NBA** : Régression logistique simple sur win_rate + différentiel de points

### Phase 2 — Avec données historiques
Dès que tu as assez de matchs réglés en DB, tu peux entraîner XGBoost :

```python
from model import BettingModel
from data_fetcher import get_all_predictions
import pandas as pd

df = get_all_predictions()
df = df[df['outcome'].notna()]  # matchs réglés uniquement

from feature_engineering import get_feature_columns
cols = get_feature_columns('football')
X = df[cols]
y = df['outcome'].map({'H': 0, 'D': 1, 'A': 2})

model = BettingModel(sport='football')
metrics = model.train(X, y)
print(metrics)
```

---

## Value Bet

Un **value bet** est détecté quand :
```
P_modèle - P_implicite_bookmaker >= 5% (configurable dans config.py)
ET confiance >= 60%
```

La mise recommandée est calculée par **Kelly fractionnaire** :
```
f* = (b × p - q) / b × 0.25   (Kelly/4, conservateur)
mise = min(f*, 5%) × bankroll
```

---

## Déploiement VPS (Hetzner CX11)

```bash
# Même pattern que le bot trading
screen -S betmind
python predictor.py --schedule
Ctrl+A, D  # détacher

screen -S dashboard
python dashboard.py
Ctrl+A, D
```

Ou avec systemd (recommandé en prod) :
```bash
sudo cp betmind.service /etc/systemd/system/
sudo systemctl enable betmind
sudo systemctl start betmind
```