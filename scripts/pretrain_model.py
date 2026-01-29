#!/usr/bin/env python3
"""
Pre-Training Script - Calibrage Initial du Modele

Ce script effectue un backtest historique complet pour:
1. Simuler les signaux qui auraient ete generes sur 6 mois
2. Calculer les outcomes reels (gains/pertes)
3. Optimiser les poids des 4 piliers
4. Sauvegarder les poids calibres

Utilisation:
    python scripts/pretrain_model.py [--months 6] [--symbols 100]

Resultat:
    data/shadow_tracking/adjusted_weights.json (poids optimises)
    data/shadow_tracking/pretrain_report.json (rapport complet)
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MARKETS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulatedSignal:
    """Signal simule historiquement"""
    symbol: str
    date: str
    entry_price: float

    # Scores simules (0-100)
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    news_score: float
    total_score: float

    # Outcome reel
    price_t5: float = 0.0
    price_t10: float = 0.0
    price_t20: float = 0.0
    pnl_t5: float = 0.0
    pnl_t10: float = 0.0
    pnl_t20: float = 0.0
    max_gain: float = 0.0
    max_drawdown: float = 0.0

    # Prix quotidiens pour simulation de sorties realiste
    # Format: [{'day': 1, 'close': 150.0, 'high': 152.0, 'low': 148.0, 'pnl_close': 1.5, ...}, ...]
    daily_prices: List[Dict] = None

    # Classification
    is_winner: bool = False  # True si pnl_t20 >= 5%

    def __post_init__(self):
        if self.daily_prices is None:
            self.daily_prices = []


class ModelPretrainer:
    """
    Pre-entraine le modele sur donnees historiques.

    Approche:
    1. Pour chaque stock dans l'univers
    2. Pour chaque semaine des N derniers mois
    3. Calculer les indicateurs techniques
    4. Generer un score synthetique
    5. Verifier l'outcome reel
    6. Optimiser les poids par regression
    """

    def __init__(
        self,
        months_back: int = 6,
        max_symbols: int = 0,  # 0 = pas de limite
        universe: str = 'us_top',  # us_top, us_all, europe, global
        output_dir: str = "data/shadow_tracking",
        on_progress: callable = None  # Callback pour mise a jour temps reel
    ):
        self.months_back = months_back
        self.max_symbols = max_symbols
        self.universe = universe
        self.output_dir = Path(output_dir)
        self.on_progress = on_progress  # Callback: (event_type, data) -> None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Chemin vers les fichiers de tickers
        self.tickers_dir = Path(__file__).parent.parent / "data" / "tickers"

        # Statistiques en temps reel
        self.live_stats = {
            'signals_total': 0,
            'signals_buy': 0,
            'wins': 0,
            'losses': 0,
            'total_gain': 0.0,
            'total_loss': 0.0,
            'best_trade': None,
            'worst_trade': None,
            'last_signals': []  # 10 derniers signaux
        }

        self.signals: List[SimulatedSignal] = []

        # Poids initiaux (egaux)
        self.weights = {
            "technical": 0.25,
            "fundamental": 0.25,
            "sentiment": 0.25,
            "news": 0.25
        }

        # Parametres de sortie (a optimiser)
        self.exit_params = {
            "take_profit_pct": 0.10,     # +10% par defaut
            "stop_loss_pct": 0.05,       # -5% par defaut
            "trailing_stop_pct": 0.05,   # 5% trailing
            "use_trailing": True,
            "max_hold_days": 20
        }

    def _emit(self, event_type: str, data: dict):
        """Emettre un evenement de progression pour affichage temps reel"""
        if self.on_progress:
            try:
                self.on_progress(event_type, data)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    def _load_tickers_from_file(self, filename: str) -> List[str]:
        """Charger les tickers depuis un fichier JSON"""
        filepath = self.tickers_dir / filename
        if not filepath.exists():
            logger.warning(f"Ticker file not found: {filepath}")
            return []

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Les tickers peuvent etre dans data['tickers'] ou directement dans data
            if isinstance(data, dict) and 'tickers' in data:
                tickers = data['tickers']
                # Chaque ticker peut etre un dict avec 'symbol' ou directement une string
                if tickers and isinstance(tickers[0], dict):
                    return [t['symbol'] for t in tickers if 'symbol' in t]
                return tickers
            elif isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return [t.get('symbol', t.get('ticker', '')) for t in data]
                return data
            return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def get_stock_universe(self) -> List[str]:
        """Obtenir l'univers de stocks pour le backtest selon le marche choisi"""
        symbols = []

        if self.universe == 'us_top':
            # SP500 + NASDAQ top
            symbols.extend(self._load_tickers_from_file('sp500.json'))
            # Ajouter les NASDAQ qui ne sont pas deja dans SP500
            nasdaq = self._load_tickers_from_file('nasdaq.json')
            symbols.extend([s for s in nasdaq if s not in symbols])

        elif self.universe == 'us_all':
            # Tout US: SP500 + NASDAQ complet + NYSE complet
            symbols.extend(self._load_tickers_from_file('sp500.json'))
            symbols.extend(self._load_tickers_from_file('nasdaq_full.json'))
            symbols.extend(self._load_tickers_from_file('nyse_full.json'))

        elif self.universe == 'europe':
            # Europe seulement
            symbols.extend(self._load_tickers_from_file('europe.json'))
            symbols.extend(self._load_tickers_from_file('cac40.json'))

        elif self.universe == 'global':
            # US + Europe
            symbols.extend(self._load_tickers_from_file('sp500.json'))
            symbols.extend(self._load_tickers_from_file('nasdaq_full.json'))
            symbols.extend(self._load_tickers_from_file('europe.json'))

        else:
            # Defaut: US top seulement
            symbols.extend(self._load_tickers_from_file('sp500.json'))

        # Si aucun ticker charge, utiliser une liste de fallback
        if not symbols:
            logger.warning("No tickers loaded, using fallback list")
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                'AMD', 'INTC', 'NFLX', 'PYPL', 'ADBE', 'CRM', 'ORCL',
                'CSCO', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX',
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
                'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY', 'BMY',
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC'
            ]

        # Deduplicate en preservant l'ordre
        symbols = list(dict.fromkeys(symbols))

        # Appliquer la limite si specifiee (0 = pas de limite)
        if self.max_symbols > 0:
            return symbols[:self.max_symbols]
        return symbols

    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculer le score technique basique"""
        if df is None or len(df) < 50:
            return 50.0  # Score neutre

        try:
            close = df['Close']

            # EMAs
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            ema_200 = close.ewm(span=200).mean() if len(df) >= 200 else ema_50

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9).mean()

            score = 50.0

            # EMA alignment (+20)
            if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
                score += 20
            elif ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
                score -= 15

            # RSI conditions (+15)
            current_rsi = rsi.iloc[-1]
            if 40 <= current_rsi <= 60:
                score += 10  # Zone neutre mais stable
            elif 30 <= current_rsi < 40:
                score += 15  # Potentiel rebond
            elif current_rsi > 70:
                score -= 10  # Surchauffe
            elif current_rsi < 30:
                score += 5   # Survente mais risque

            # MACD crossover (+15)
            if macd.iloc[-1] > signal_line.iloc[-1]:
                score += 15
                if macd.iloc[-2] <= signal_line.iloc[-2]:
                    score += 5  # Fresh crossover

            # Volume trend (+10)
            if 'Volume' in df.columns:
                vol_avg = df['Volume'].rolling(20).mean()
                if df['Volume'].iloc[-1] > vol_avg.iloc[-1] * 1.2:
                    score += 10

            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"Technical score error: {e}")
            return 50.0

    def calculate_fundamental_score(self, symbol: str) -> float:
        """Score fondamental simplifie (basé sur P/E, etc.)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            score = 50.0

            # P/E ratio
            pe = info.get('forwardPE') or info.get('trailingPE')
            if pe:
                if pe < 15:
                    score += 15
                elif pe < 25:
                    score += 10
                elif pe > 50:
                    score -= 15

            # Profit margin
            margin = info.get('profitMargins', 0)
            if margin > 0.2:
                score += 15
            elif margin > 0.1:
                score += 10
            elif margin < 0:
                score -= 10

            # Revenue growth
            growth = info.get('revenueGrowth', 0)
            if growth > 0.2:
                score += 15
            elif growth > 0.1:
                score += 10
            elif growth < 0:
                score -= 10

            return max(0, min(100, score))

        except Exception:
            return 50.0  # Neutre si erreur

    def calculate_momentum_sentiment(self, df: pd.DataFrame) -> float:
        """Score sentiment basé sur le momentum prix (proxy)"""
        if df is None or len(df) < 20:
            return 50.0

        try:
            # Momentum 5 jours
            ret_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100

            # Momentum 20 jours
            ret_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100

            score = 50.0

            # Short-term momentum
            if ret_5d > 5:
                score += 20
            elif ret_5d > 2:
                score += 10
            elif ret_5d < -5:
                score -= 15

            # Medium-term momentum
            if ret_20d > 10:
                score += 15
            elif ret_20d > 5:
                score += 10
            elif ret_20d < -10:
                score -= 15

            return max(0, min(100, score))

        except Exception:
            return 50.0

    def simulate_news_score(self, price_change: float) -> float:
        """
        Score news simulé - SANS look-ahead bias.

        IMPORTANT: On ne peut PAS utiliser price_change car c'est du look-ahead.
        Le vrai score news viendrait d'APIs externes (non disponibles historiquement).
        On simule avec du bruit centre sur 50 (neutre).
        """
        # Score aleatoire centre sur 50 avec ecart-type de 15
        # Cela simule l'incertitude du news score sans biais directionnel
        score = np.random.normal(50, 15)
        return max(0, min(100, score))

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Telecharger les donnees historiques"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date + timedelta(days=30))
            return df
        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    async def run_backtest(self) -> Dict:
        """
        Executer le backtest complet.

        Pour chaque semaine des N derniers mois:
        1. Calculer les scores
        2. Generer un signal si score > 55
        3. Verifier l'outcome a T+5, T+10, T+20
        """
        logger.info("=" * 60)
        logger.info("PRE-TRAINING: Starting historical backtest")
        logger.info("=" * 60)

        symbols = self.get_stock_universe()
        logger.info(f"Universe: {len(symbols)} symbols")

        today = datetime.now()
        end_date = today - timedelta(days=25)  # Laisser 25j pour verifier outcomes T+20
        start_date = end_date - timedelta(days=30 * self.months_back)

        logger.info(f"Date du jour: {today.strftime('%d/%m/%Y')}")
        logger.info(f"Période d'analyse: du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")
        logger.info(f"  → {self.months_back} mois de données, fin -25j pour vérifier les outcomes")

        # Points d'evaluation (chaque lundi)
        eval_dates = []
        current = start_date
        while current < end_date:
            if current.weekday() == 0:  # Lundi
                eval_dates.append(current)
            current += timedelta(days=1)

        logger.info(f"Evaluation points: {len(eval_dates)} dates")

        # Progress bar
        total_evaluations = len(symbols) * len(eval_dates)

        symbols_processed = 0
        with tqdm(total=total_evaluations, desc="Backtesting") as pbar:
            for symbol in symbols:
                symbols_processed += 1
                # Emettre evenement de progression symbole
                self._emit('symbol', {
                    'symbol': symbol,
                    'current': symbols_processed,
                    'total': len(symbols),
                    'progress_pct': round(symbols_processed / len(symbols) * 100, 1)
                })

                # Ceder le controle a l'event loop tous les 5 symboles
                # pour eviter de bloquer le WebSocket
                if symbols_processed % 5 == 0:
                    await asyncio.sleep(0)

                # Fetch all data once
                df_full = await self.fetch_historical_data(symbol, start_date, end_date + timedelta(days=30))

                if df_full is None or len(df_full) < 50:
                    pbar.update(len(eval_dates))
                    continue

                # Pre-calculate fundamental score (doesn't change much)
                fundamental_score = self.calculate_fundamental_score(symbol)

                date_counter = 0
                for eval_date in eval_dates:
                    pbar.update(1)
                    date_counter += 1

                    # Yield toutes les 20 dates pour eviter de bloquer l'event loop
                    if date_counter % 20 == 0:
                        await asyncio.sleep(0)

                    # Get data up to eval_date
                    df = df_full[df_full.index <= eval_date.strftime('%Y-%m-%d')]

                    if len(df) < 50:
                        continue

                    # Calculate scores
                    technical_score = self.calculate_technical_score(df)
                    sentiment_score = self.calculate_momentum_sentiment(df)

                    # Price change for news proxy
                    price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1) * 100
                    news_score = self.simulate_news_score(price_change)

                    # Total score (weighted)
                    total_score = (
                        technical_score * self.weights["technical"] +
                        fundamental_score * self.weights["fundamental"] +
                        sentiment_score * self.weights["sentiment"] +
                        news_score * self.weights["news"]
                    )

                    # Only track signals >= 50 (potential BUY)
                    if total_score < 50:
                        continue

                    entry_price = df['Close'].iloc[-1]

                    # Get future prices for outcome
                    df_future = df_full[df_full.index > eval_date.strftime('%Y-%m-%d')]

                    if len(df_future) < 20:
                        continue

                    price_t5 = df_future['Close'].iloc[min(4, len(df_future)-1)]
                    price_t10 = df_future['Close'].iloc[min(9, len(df_future)-1)]
                    price_t20 = df_future['Close'].iloc[min(19, len(df_future)-1)]
                    max_price = df_future['High'].iloc[:20].max()
                    min_price = df_future['Low'].iloc[:20].min()

                    pnl_t20 = round(((price_t20 / entry_price) - 1) * 100, 2)
                    # Winner = gain >= 5% (plus strict que 3%)
                    is_winner = pnl_t20 >= 5

                    # Stocker les prix quotidiens pour simulation de sorties realiste
                    daily_prices = []
                    for day_idx in range(min(20, len(df_future))):
                        day_close = df_future['Close'].iloc[day_idx]
                        day_high = df_future['High'].iloc[day_idx]
                        day_low = df_future['Low'].iloc[day_idx]
                        daily_prices.append({
                            'day': day_idx + 1,
                            'close': round(day_close, 2),
                            'high': round(day_high, 2),
                            'low': round(day_low, 2),
                            'pnl_close': round(((day_close / entry_price) - 1) * 100, 2),
                            'pnl_high': round(((day_high / entry_price) - 1) * 100, 2),
                            'pnl_low': round(((day_low / entry_price) - 1) * 100, 2)
                        })

                    signal = SimulatedSignal(
                        symbol=symbol,
                        date=eval_date.strftime('%Y-%m-%d'),
                        entry_price=entry_price,
                        technical_score=round(technical_score, 2),
                        fundamental_score=round(fundamental_score, 2),
                        sentiment_score=round(sentiment_score, 2),
                        news_score=round(news_score, 2),
                        total_score=round(total_score, 2),
                        price_t5=round(price_t5, 2),
                        price_t10=round(price_t10, 2),
                        price_t20=round(price_t20, 2),
                        pnl_t5=round(((price_t5 / entry_price) - 1) * 100, 2),
                        pnl_t10=round(((price_t10 / entry_price) - 1) * 100, 2),
                        pnl_t20=pnl_t20,
                        max_gain=round(((max_price / entry_price) - 1) * 100, 2),
                        max_drawdown=round(((min_price / entry_price) - 1) * 100, 2),
                        daily_prices=daily_prices,
                        is_winner=is_winner
                    )

                    self.signals.append(signal)

                    # Mise a jour des stats en temps reel
                    self.live_stats['signals_total'] += 1
                    if total_score >= 55:
                        self.live_stats['signals_buy'] += 1
                    if is_winner:
                        self.live_stats['wins'] += 1
                        self.live_stats['total_gain'] += pnl_t20
                        if self.live_stats['best_trade'] is None or pnl_t20 > self.live_stats['best_trade']['pnl']:
                            self.live_stats['best_trade'] = {'symbol': symbol, 'pnl': pnl_t20, 'date': eval_date.strftime('%Y-%m-%d')}
                    else:
                        self.live_stats['losses'] += 1
                        self.live_stats['total_loss'] += abs(pnl_t20)
                        if self.live_stats['worst_trade'] is None or pnl_t20 < self.live_stats['worst_trade']['pnl']:
                            self.live_stats['worst_trade'] = {'symbol': symbol, 'pnl': pnl_t20, 'date': eval_date.strftime('%Y-%m-%d')}

                    # Garder les 10 derniers signaux
                    self.live_stats['last_signals'].append({
                        'symbol': symbol,
                        'date': eval_date.strftime('%d/%m/%Y'),
                        'score': round(total_score, 1),
                        'tech': round(technical_score, 0),
                        'fund': round(fundamental_score, 0),
                        'sent': round(sentiment_score, 0),
                        'news': round(news_score, 0),
                        'pnl': pnl_t20,
                        'is_winner': is_winner,
                        'dominant': max([
                            ('TECH', technical_score),
                            ('FUND', fundamental_score),
                            ('SENT', sentiment_score),
                            ('NEWS', news_score)
                        ], key=lambda x: x[1])[0]
                    })
                    if len(self.live_stats['last_signals']) > 10:
                        self.live_stats['last_signals'].pop(0)

                    # Determiner le pilier dominant
                    dominant_pillar = max([
                        ('TECH', technical_score),
                        ('FUND', fundamental_score),
                        ('SENT', sentiment_score),
                        ('NEWS', news_score)
                    ], key=lambda x: x[1])[0]

                    # Emettre l'evenement avec toutes les infos
                    self._emit('signal', {
                        'symbol': symbol,
                        'date': eval_date.strftime('%d/%m/%Y'),
                        'score': round(total_score, 1),
                        'pnl': round(pnl_t20, 2),
                        'is_winner': is_winner,
                        'dominant': dominant_pillar,
                        'pillars': {
                            'tech': round(technical_score, 0),
                            'fund': round(fundamental_score, 0),
                            'sent': round(sentiment_score, 0),
                            'news': round(news_score, 0)
                        },
                        'stats': {
                            'total': self.live_stats['signals_total'],
                            'wins': self.live_stats['wins'],
                            'losses': self.live_stats['losses'],
                            'win_rate': round(self.live_stats['wins'] / max(1, self.live_stats['wins'] + self.live_stats['losses']) * 100, 1),
                            'avg_gain': round(self.live_stats['total_gain'] / max(1, self.live_stats['wins']), 2) if self.live_stats['wins'] > 0 else 0,
                            'avg_loss': round(self.live_stats['total_loss'] / max(1, self.live_stats['losses']), 2) if self.live_stats['losses'] > 0 else 0
                        }
                    })

        logger.info(f"Generated {len(self.signals)} signals")

        # Calculate initial stats
        stats = self._calculate_stats()
        logger.info(f"Initial Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"Initial Profit Factor: {stats['profit_factor']:.2f}")

        return stats

    def _calculate_stats(self, threshold: float = 55) -> Dict:
        """Calculer les statistiques pour un seuil donne"""
        signals_above = [s for s in self.signals if s.total_score >= threshold]

        if not signals_above:
            return {"win_rate": 0, "profit_factor": 0, "total": 0}

        winners = [s for s in signals_above if s.is_winner]
        losers = [s for s in signals_above if not s.is_winner]

        win_rate = len(winners) / len(signals_above) * 100

        total_gains = sum(max(0, s.pnl_t20) for s in signals_above)
        total_losses = abs(sum(min(0, s.pnl_t20) for s in signals_above))

        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

        avg_winner = sum(s.pnl_t20 for s in winners) / len(winners) if winners else 0
        avg_loser = sum(s.pnl_t20 for s in losers) / len(losers) if losers else 0

        return {
            "threshold": threshold,
            "total_signals": len(signals_above),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "expectancy": (win_rate/100 * avg_winner) + ((1 - win_rate/100) * avg_loser)
        }

    def optimize_weights(self) -> Dict[str, float]:
        """
        Optimiser les poids des piliers via regression.

        Objectif: Maximiser la correlation entre score total et outcome.
        """
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION: Finding optimal weights")
        logger.info("=" * 60)

        if len(self.signals) < 50:
            logger.warning("Not enough signals for optimization")
            return self.weights

        # Preparer les donnees
        X = np.array([
            [s.technical_score, s.fundamental_score, s.sentiment_score, s.news_score]
            for s in self.signals
        ])

        # Target: outcome binaire (winner = 1, loser = 0)
        y = np.array([1 if s.is_winner else 0 for s in self.signals])

        # Grid search pour trouver les meilleurs poids
        best_weights = self.weights.copy()
        best_score = 0

        weight_options = [0.15, 0.20, 0.25, 0.30, 0.35]

        logger.info("Grid searching optimal weights...")

        # Pre-calcul: compter les combinaisons valides
        valid_combos = 0
        for w_tech in weight_options:
            for w_fund in weight_options:
                for w_sent in weight_options:
                    w_news = 1 - w_tech - w_fund - w_sent
                    if 0.1 <= w_news <= 0.4:
                        valid_combos += 1
        logger.info(f"Testing {valid_combos} weight combinations on {len(self.signals)} signals...")

        combo_count = 0
        for w_tech in weight_options:
            for w_fund in weight_options:
                for w_sent in weight_options:
                    w_news = 1 - w_tech - w_fund - w_sent

                    if w_news < 0.1 or w_news > 0.4:
                        continue

                    combo_count += 1

                    # Calcul vectorise des scores ponderes (RAPIDE)
                    weights = np.array([w_tech, w_fund, w_sent, w_news])
                    weighted_scores = X @ weights

                    # Filtrer directement avec NumPy (PAS de creation d'objets)
                    mask = weighted_scores >= 55
                    count_above = mask.sum()

                    if count_above < 20:
                        continue

                    # Calculer le win rate directement sur le masque
                    winners = y[mask].sum()
                    win_rate = winners / count_above

                    # Score = win_rate * sqrt(sample_size) pour favoriser robustesse
                    score = win_rate * np.sqrt(count_above)

                    if score > best_score:
                        best_score = score
                        best_weights = {
                            "technical": round(w_tech, 2),
                            "fundamental": round(w_fund, 2),
                            "sentiment": round(w_sent, 2),
                            "news": round(w_news, 2)
                        }

        logger.info(f"Tested {combo_count} combinations")

        logger.info(f"Optimal weights found: {best_weights}")

        # Verify improvement
        self.weights = best_weights

        # Recalculate signals with new weights
        for s in self.signals:
            s.total_score = (
                s.technical_score * self.weights["technical"] +
                s.fundamental_score * self.weights["fundamental"] +
                s.sentiment_score * self.weights["sentiment"] +
                s.news_score * self.weights["news"]
            )

        new_stats = self._calculate_stats()
        logger.info(f"Optimized Win Rate: {new_stats['win_rate']:.1f}%")
        logger.info(f"Optimized Profit Factor: {new_stats['profit_factor']:.2f}")

        return best_weights

    def find_optimal_threshold(self) -> Tuple[float, Dict]:
        """Trouver le seuil optimal pour les signaux BUY"""
        logger.info("\n" + "=" * 60)
        logger.info("THRESHOLD: Finding optimal buy threshold")
        logger.info("=" * 60)

        thresholds = [50, 52, 55, 57, 60, 62, 65, 68, 70]
        best_threshold = 55
        best_expectancy = 0

        for threshold in thresholds:
            stats = self._calculate_stats(threshold)

            if stats['total_signals'] < 10:
                continue

            logger.info(f"Threshold {threshold}: WR={stats['win_rate']:.1f}%, "
                       f"PF={stats['profit_factor']:.2f}, N={stats['total_signals']}, "
                       f"Exp={stats['expectancy']:.2f}%")

            if stats['expectancy'] > best_expectancy and stats['total_signals'] >= 20:
                best_expectancy = stats['expectancy']
                best_threshold = threshold

        logger.info(f"\nOptimal threshold: {best_threshold} (expectancy: {best_expectancy:.2f}%)")

        return best_threshold, self._calculate_stats(best_threshold)

    def optimize_exit_params(self, threshold: float = 55) -> Dict:
        """
        Optimiser les parametres de sortie (TP, SL, Trailing).

        METHODE REALISTE: Simulation jour par jour avec les daily_prices.
        On ne peut PAS savoir si le high ou low arrive en premier dans la journee,
        donc on utilise une heuristique conservative:
        - Si SL est touche (low <= entry * (1-sl)) -> sortie a SL
        - Sinon si TP est touche (high >= entry * (1+tp)) -> sortie a TP
        - Priorite au SL car c'est plus conservateur
        """
        logger.info("\n" + "=" * 60)
        logger.info("EXIT PARAMS: Finding optimal exit strategy (REALISTIC)")
        logger.info("=" * 60)

        # Signaux au-dessus du seuil avec daily_prices
        valid_signals = [s for s in self.signals if s.total_score >= threshold and s.daily_prices]
        if len(valid_signals) < 30:
            logger.warning(f"Not enough signals for exit optimization ({len(valid_signals)} found)")
            return self.exit_params

        logger.info(f"Optimizing on {len(valid_signals)} signals with daily price data")

        # Grid search sur les parametres de sortie
        tp_options = [5, 8, 10, 12, 15, 20]           # 5% a 20%
        sl_options = [3, 4, 5, 6, 8, 10]              # 3% a 10%
        trailing_options = [None, 3, 5, 7]            # Pas de trailing ou 3-7%
        max_hold_options = [10, 15, 20]               # Duree max de detention

        best_params = self.exit_params.copy()
        best_score = 0
        results_log = []

        total_combos = len(tp_options) * len(sl_options) * len(trailing_options) * len(max_hold_options)
        logger.info(f"Testing {total_combos} combinations...")

        for tp in tp_options:
            for sl in sl_options:
                for trailing in trailing_options:
                    for max_hold in max_hold_options:
                        # Simuler les trades avec ces parametres - JOUR PAR JOUR
                        wins = 0
                        losses = 0
                        total_gain = 0
                        total_loss = 0

                        for signal in valid_signals:
                            result = self._simulate_trade_exit(
                                signal.daily_prices,
                                tp_pct=tp,
                                sl_pct=sl,
                                trailing_pct=trailing,
                                max_hold_days=max_hold
                            )

                            if result['pnl'] > 0:
                                wins += 1
                                total_gain += result['pnl']
                            else:
                                losses += 1
                                total_loss += abs(result['pnl'])

                        # Calculer les metriques
                        total_trades = wins + losses
                        if total_trades < 20:
                            continue

                        win_rate = (wins / total_trades) * 100
                        profit_factor = (total_gain / total_loss) if total_loss > 0 else 10
                        expectancy = (total_gain - total_loss) / total_trades

                        # Score combine: expectancy * sqrt(trades) pour favoriser robustesse
                        score = expectancy * np.sqrt(total_trades)

                        results_log.append({
                            'tp': tp, 'sl': sl, 'trailing': trailing, 'max_hold': max_hold,
                            'win_rate': win_rate, 'pf': profit_factor, 'exp': expectancy,
                            'score': score, 'trades': total_trades
                        })

                        if score > best_score:
                            if win_rate >= 35:  # Minimum win rate acceptable (realiste)
                                best_score = score
                                best_params = {
                                    "take_profit_pct": tp / 100,
                                    "stop_loss_pct": sl / 100,
                                    "trailing_stop_pct": (trailing / 100) if trailing else 0.05,
                                    "use_trailing": trailing is not None,
                                    "max_hold_days": max_hold
                                }

        # Logger les meilleurs resultats
        sorted_results = sorted(results_log, key=lambda x: x['score'], reverse=True)[:10]
        logger.info("\nTop 10 exit configurations:")
        for i, r in enumerate(sorted_results):
            trailing_str = f"{r['trailing']:.0f}%" if r['trailing'] else "None"
            logger.info(f"  {i+1}. TP={r['tp']:.0f}% SL={r['sl']:.0f}% "
                       f"Trail={trailing_str} Hold={r['max_hold']}d -> "
                       f"WR={r['win_rate']:.1f}% PF={r['pf']:.2f} Exp={r['exp']:.2f}%")

        logger.info(f"\nOptimal exit params: TP={best_params['take_profit_pct']*100:.0f}%, "
                   f"SL={best_params['stop_loss_pct']*100:.0f}%, "
                   f"Trailing={'Yes (' + str(int(best_params['trailing_stop_pct']*100)) + '%)' if best_params['use_trailing'] else 'No'}, "
                   f"MaxHold={best_params['max_hold_days']}d")

        self.exit_params = best_params
        return best_params

    def _simulate_trade_exit(
        self,
        daily_prices: List[Dict],
        tp_pct: float,
        sl_pct: float,
        trailing_pct: float = None,
        max_hold_days: int = 20
    ) -> Dict:
        """
        Simuler une sortie de trade jour par jour.

        IMPORTANT: On verifie SL en premier (conservative).
        Un SL touche en intraday sort immediatement, meme si TP aussi touche.

        Returns:
            {'pnl': float, 'exit_day': int, 'exit_reason': str}
        """
        if not daily_prices:
            return {'pnl': 0, 'exit_day': 0, 'exit_reason': 'no_data'}

        highest_pnl = 0  # Pour trailing stop
        trailing_stop_level = -sl_pct  # Niveau de stop (negatif)

        for day_data in daily_prices:
            day = day_data['day']
            pnl_high = day_data['pnl_high']
            pnl_low = day_data['pnl_low']
            pnl_close = day_data['pnl_close']

            # Mise a jour du trailing stop si actif
            if trailing_pct and pnl_high > highest_pnl:
                highest_pnl = pnl_high
                # Le trailing stop suit le high moins trailing_pct
                new_stop = highest_pnl - trailing_pct
                if new_stop > trailing_stop_level:
                    trailing_stop_level = new_stop

            # Verifier SL en premier (conservative - on assume le pire)
            if pnl_low <= -sl_pct:
                return {'pnl': -sl_pct, 'exit_day': day, 'exit_reason': 'stop_loss'}

            # Verifier trailing stop
            if trailing_pct and pnl_low <= trailing_stop_level:
                # Sortie au trailing stop (peut etre positif ou negatif)
                return {'pnl': trailing_stop_level, 'exit_day': day, 'exit_reason': 'trailing_stop'}

            # Verifier TP
            if pnl_high >= tp_pct:
                return {'pnl': tp_pct, 'exit_day': day, 'exit_reason': 'take_profit'}

            # Max hold days atteint
            if day >= max_hold_days:
                return {'pnl': pnl_close, 'exit_day': day, 'exit_reason': 'max_hold'}

        # Sortie au dernier jour disponible
        last_day = daily_prices[-1]
        return {'pnl': last_day['pnl_close'], 'exit_day': last_day['day'], 'exit_reason': 'end_of_data'}

    def save_results(self, stats: Dict, threshold: float):
        """Sauvegarder les resultats du pre-training"""
        # Sauvegarder les poids optimises
        weights_file = self.output_dir / "adjusted_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
        logger.info(f"Saved optimized weights to {weights_file}")

        # Sauvegarder les parametres de sortie optimises
        exit_file = self.output_dir / "exit_params.json"
        with open(exit_file, 'w') as f:
            json.dump(self.exit_params, f, indent=2)
        logger.info(f"Saved optimized exit params to {exit_file}")

        # Helper pour convertir les types NumPy en types Python natifs
        def convert_numpy(obj):
            """Convertir les types NumPy en types Python pour JSON"""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Sauvegarder le rapport complet
        report = {
            "pretrain_date": datetime.now().isoformat(),
            "period_months": self.months_back,
            "total_symbols": len(self.get_stock_universe()),
            "total_signals": len(self.signals),
            "optimized_weights": self.weights,
            "optimized_exit_params": self.exit_params,
            "optimal_threshold": int(threshold),
            "final_stats": convert_numpy(stats),
            "signals_sample": [convert_numpy(asdict(s)) for s in self.signals[:100]]  # Echantillon
        }

        report_file = self.output_dir / "pretrain_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved full report to {report_file}")

        # Sauvegarder tous les signaux pour analyse
        signals_file = self.output_dir / "pretrain_signals.json"
        with open(signals_file, 'w') as f:
            json.dump([convert_numpy(asdict(s)) for s in self.signals], f)
        logger.info(f"Saved {len(self.signals)} signals to {signals_file}")


async def main():
    """Point d'entree principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train the trading model")
    parser.add_argument("--months", type=int, default=12, help="Months of historical data (default: 12)")
    parser.add_argument("--symbols", type=int, default=0, help="Max symbols to analyze (0 = all, default: 0)")
    parser.add_argument("--universe", type=str, default='us_top',
                       choices=['us_top', 'us_all', 'europe', 'global'],
                       help="Stock universe to use")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TRADINGBOT V5 - MODEL PRE-TRAINING (REALISTIC)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Historical period: {args.months} months")
    print(f"  - Symbol universe: {args.universe} {'(all)' if args.symbols == 0 else f'(max {args.symbols})'}")
    print(f"  - Winner threshold: +5% (strict)")
    print(f"  - News score: Random (no look-ahead bias)")
    print(f"  - Exit simulation: Day-by-day with SL priority")
    print(f"  - Output: data/shadow_tracking/")
    print("\n")

    pretrainer = ModelPretrainer(
        months_back=args.months,
        max_symbols=args.symbols,
        universe=args.universe
    )

    # 1. Run backtest
    await pretrainer.run_backtest()

    # 2. Optimize weights
    optimal_weights = pretrainer.optimize_weights()

    # 3. Find optimal threshold
    threshold, final_stats = pretrainer.find_optimal_threshold()

    # 4. Optimize exit parameters (TP, SL, Trailing)
    exit_params = pretrainer.optimize_exit_params(threshold)

    # 5. Save results
    pretrainer.save_results(final_stats, threshold)

    print("\n" + "=" * 60)
    print("  PRE-TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nOptimized Weights:")
    for pillar, weight in optimal_weights.items():
        print(f"  - {pillar.capitalize()}: {weight:.0%}")
    print(f"\nOptimal Threshold: {threshold}")
    print(f"\nOptimized Exit Strategy:")
    print(f"  - Take Profit: +{exit_params['take_profit_pct']*100:.0f}%")
    print(f"  - Stop Loss: -{exit_params['stop_loss_pct']*100:.0f}%")
    print(f"  - Trailing Stop: {'Yes (' + str(int(exit_params['trailing_stop_pct']*100)) + '%)' if exit_params['use_trailing'] else 'No'}")
    print(f"\nExpected Performance:")
    print(f"  - Win Rate: {final_stats['win_rate']:.1f}%")
    print(f"  - Profit Factor: {final_stats['profit_factor']:.2f}")
    print(f"  - Avg Winner: +{final_stats['avg_winner']:.1f}%")
    print(f"  - Avg Loser: {final_stats['avg_loser']:.1f}%")
    print(f"  - Expectancy: {final_stats['expectancy']:.2f}% per trade")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
