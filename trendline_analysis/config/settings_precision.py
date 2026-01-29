"""
Configuration OPTIMISÉE pour précision maximale des obliques RSI

Cette configuration privilégie la QUALITÉ sur la QUANTITÉ:
- Détecte moins d'obliques
- Mais celles détectées sont très précises (R² > 0.65, distance max < 4 points)
"""

# RSI Parameters
RSI_PERIOD = 14
RSI_EMA_PERIODS = [9, 14]

# ===== AMÉLIORATION #1 : Détection de pics adaptative =====
PEAK_PROMINENCE_ADAPTIVE = True  # Activer prominence adaptative

# Prominence selon volatilité RSI
PEAK_PROMINENCE_LOW_VOLATILITY = 2.5   # RSI std < 10
PEAK_PROMINENCE_MEDIUM_VOLATILITY = 3.5  # RSI std 10-15
PEAK_PROMINENCE_HIGH_VOLATILITY = 4.5    # RSI std > 15

# Fallback si non-adaptatif
PEAK_PROMINENCE = 3.5  # Augmenté de 2.0 → 3.5 (pics plus marqués)

PEAK_DISTANCE = 5      # Distance minimale entre pics (inchangé)
MIN_PEAKS_FOR_TRENDLINE = 3  # Minimum 3 pics (inchangé)

# ===== AMÉLIORATION #2 : Validation STRICTE de l'oblique =====
# R² - Coefficient de détermination
MIN_R_SQUARED = 0.65   # AUGMENTÉ de 0.25 → 0.65 (au moins 65% variance expliquée)
# Note: R² = 1.0 = oblique parfaite, R² = 0.0 = aucune corrélation

# Distance maximale des pics à l'oblique
MAX_RESIDUAL_DISTANCE = 4.0  # RÉDUIT de 8.0 → 4.0 points RSI max

# Nouvelles métriques de précision
MAX_MEAN_RESIDUAL = 2.0      # Distance MOYENNE max : 2 points RSI
MAX_STD_RESIDUAL = 1.5       # Écart-type des distances max : 1.5 points

# Pentes acceptées (inchangé)
MIN_SLOPE = -1.0       # Pente minimale (descendante)
MAX_SLOPE = 0.0        # Pente maximale (doit être négative)

# ===== AMÉLIORATION #3 : Algorithme d'ajustement =====
USE_RANSAC = True      # Utiliser RANSAC au lieu de régression linéaire simple
RANSAC_RESIDUAL_THRESHOLD = 3.0  # Un pic est "inlier" si distance < 3 points
RANSAC_MAX_TRIALS = 200           # Nombre d'essais RANSAC

# ===== AMÉLIORATION #4 : Filtrage qualité des pics =====
FILTER_PEAKS_BY_QUALITY = True   # Activer filtrage secondaire
MIN_PEAK_PROMINENCE_ABSOLUTE = 3.0  # Un pic doit être au moins 3 points au-dessus des creux
MIN_PEAK_SEPARATION = 4  # Pics séparés d'au moins 4 périodes

# Breakout Detection (inchangé)
BREAKOUT_THRESHOLD = 0.0
CONFIRMATION_PERIODS = 1
MAX_BREAKOUT_AGE = 3

# Timeframe (inchangé)
TIMEFRAME_PRIORITY = ['weekly', 'daily']
WEEKLY_LOOKBACK_PERIODS = 104  # ~2 ans
DAILY_LOOKBACK_PERIODS = 252   # ~1 an

# ===== AMÉLIORATION #5 : Scoring raffiné =====
TRENDLINE_QUALITY_WEIGHTS = {
    'r_squared': 0.50,           # AUGMENTÉ de 0.40 → 0.50 (priorité au fit)
    'mean_residual': 0.25,       # NOUVEAU : distance moyenne aux pics
    'num_points': 0.15,          # RÉDUIT de 0.30 → 0.15 (qualité > quantité)
    'slope_consistency': 0.05,   # RÉDUIT de 0.20 → 0.05
    'recency': 0.05             # RÉDUIT de 0.10 → 0.05
}

# Bonus pour obliques très précises
PRECISION_BONUS_THRESHOLD = {
    'r_squared': 0.85,     # R² > 0.85 → +10 points
    'mean_residual': 1.0,  # Distance moy < 1.0 → +10 points
    'max_residual': 2.5    # Distance max < 2.5 → +10 points
}

# Visualization (inchangé)
TRENDLINE_COLOR = '#FF9800'
BREAKOUT_COLOR = '#4CAF50'
TRENDLINE_WIDTH = 2
TRENDLINE_DASH = 'dash'

# ===== RÉSUMÉ DES CHANGEMENTS =====
"""
AVANT (settings.py):
- PEAK_PROMINENCE = 2.0       → Détectait trop de petits pics
- MIN_R_SQUARED = 0.25        → Obliques imprécises acceptées
- MAX_RESIDUAL_DISTANCE = 8.0 → Pics pouvaient être très loin de l'oblique
- Régression linéaire simple  → Sensible aux outliers

APRÈS (settings_precision.py):
- PEAK_PROMINENCE = 3.5 (adaptatif 2.5-4.5)  → Pics marqués seulement
- MIN_R_SQUARED = 0.65                        → Obliques précises uniquement
- MAX_RESIDUAL_DISTANCE = 4.0                 → Pics proches de l'oblique
- RANSAC + validations strictes               → Robuste aux outliers

IMPACT ATTENDU:
- Moins d'obliques détectées (-30% à -50%)
- Mais précision géométrique excellente (R² moyen > 0.75)
- Distance moyenne pics/oblique < 2 points RSI
- Réduction faux positifs estimée : 40-50%
"""
