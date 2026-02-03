"""
Feedback Loop - Analyse des résultats du marché pour apprentissage.

Récupère les top gainers/losers et analyse quels indicateurs les avaient prédits.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Chemins des fichiers de données
DATA_DIR = Path(__file__).parent.parent.parent / "data"
LEARNED_WEIGHTS_FILE = DATA_DIR / "learned_weights.json"
PATTERNS_DB_FILE = DATA_DIR / "patterns_db.json"
FEEDBACK_HISTORY_FILE = DATA_DIR / "feedback_history.json"


class FeedbackLoop:
    """
    Analyse les résultats du marché et met à jour les poids des indicateurs.
    """
    
    def __init__(self, min_move_pct: float = 5.0, lookback_days: int = 1):
        """
        Args:
            min_move_pct: Mouvement minimum pour considérer un gainer/loser (%)
            lookback_days: Jours à regarder en arrière pour les features
        """
        self.min_move_pct = min_move_pct
        self.lookback_days = lookback_days
        self._load_weights()
        self._load_patterns()
        
    def _load_weights(self):
        """Charge les poids appris."""
        if LEARNED_WEIGHTS_FILE.exists():
            with open(LEARNED_WEIGHTS_FILE) as f:
                self.weights = json.load(f)
        else:
            self.weights = self._init_default_weights()
            self._save_weights()
    
    def _init_default_weights(self) -> Dict:
        """Initialise les poids par défaut."""
        indicators = [
            # Trend
            "ema_cross_20_50", "ema_cross_50_200", "price_above_sma20",
            "price_above_sma50", "price_above_sma200", "adx_strong",
            # Momentum  
            "rsi_oversold", "rsi_overbought", "macd_bullish", "macd_bearish",
            "stoch_oversold", "stoch_overbought",
            # Volume
            "volume_spike", "obv_rising", "volume_breakout",
            # Volatility
            "bollinger_squeeze", "atr_expansion", "price_near_lower_band"
        ]
        return {ind: {
            "indicator": ind,
            "weight": 1.0,
            "accuracy": 0.5,
            "total_signals": 0,
            "correct_signals": 0,
            "last_updated": ""
        } for ind in indicators}
    
    def _save_weights(self):
        """Sauvegarde les poids."""
        with open(LEARNED_WEIGHTS_FILE, "w") as f:
            json.dump(self.weights, f, indent=2)
    
    def _load_patterns(self):
        """Charge les patterns découverts."""
        if PATTERNS_DB_FILE.exists():
            with open(PATTERNS_DB_FILE) as f:
                self.patterns = json.load(f)
        else:
            self.patterns = {"winning_combos": [], "losing_combos": []}
            self._save_patterns()
    
    def _save_patterns(self):
        """Sauvegarde les patterns."""
        with open(PATTERNS_DB_FILE, "w") as f:
            json.dump(self.patterns, f, indent=2)

    async def get_market_movers(self, date: Optional[datetime] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Récupère les top gainers et losers du jour.
        
        Returns:
            (gainers, losers) - Listes de dicts avec symbol, pct_change, volume
        """
        if date is None:
            date = datetime.now()
        
        # Liste de symboles à surveiller (S&P 500 + quelques autres)
        # En production, utiliser une API comme Polygon pour les vrais movers
        watchlist = await self._get_watchlist()
        
        gainers = []
        losers = []
        
        logger.info(f"Analysing {len(watchlist)} symbols for market movers...")
        
        # Batch download pour efficacité
        try:
            end_date = date.strftime("%Y-%m-%d")
            start_date = (date - timedelta(days=5)).strftime("%Y-%m-%d")
            
            # Télécharger par batch de 50
            for i in range(0, len(watchlist), 50):
                batch = watchlist[i:i+50]
                batch_str = " ".join(batch)
                
                try:
                    data = yf.download(batch_str, start=start_date, end=end_date, 
                                      progress=False, threads=True)
                    
                    if data.empty:
                        continue
                    
                    # Calculer les variations
                    for symbol in batch:
                        try:
                            if symbol in data["Close"].columns:
                                closes = data["Close"][symbol].dropna()
                                if len(closes) >= 2:
                                    pct_change = ((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100
                                    volume = data["Volume"][symbol].iloc[-1] if "Volume" in data else 0
                                    
                                    item = {
                                        "symbol": symbol,
                                        "pct_change": round(pct_change, 2),
                                        "volume": int(volume),
                                        "close": round(closes.iloc[-1], 2)
                                    }
                                    
                                    if pct_change >= self.min_move_pct:
                                        gainers.append(item)
                                    elif pct_change <= -self.min_move_pct:
                                        losers.append(item)
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    logger.warning(f"Batch download error: {e}")
                    continue
                
                # Rate limiting
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error getting market movers: {e}")
        
        # Trier par performance
        gainers.sort(key=lambda x: x["pct_change"], reverse=True)
        losers.sort(key=lambda x: x["pct_change"])
        
        logger.info(f"Found {len(gainers)} gainers (>+{self.min_move_pct}%), {len(losers)} losers (<-{self.min_move_pct}%)")
        
        return gainers[:20], losers[:20]
    
    async def _get_watchlist(self) -> List[str]:
        """Retourne la watchlist à analyser."""
        # Charger depuis le fichier watchlist du bot
        watchlist_file = DATA_DIR / "watchlist.txt"
        if watchlist_file.exists():
            with open(watchlist_file) as f:
                symbols = [line.strip() for line in f if line.strip()]
                return symbols[:500]  # Limiter pour la perf
        
        # Fallback: Top 100 symboles liquides
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
            "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT",
            "DHR", "LIN", "CMCSA", "VZ", "ADBE", "NKE", "TXN", "PM", "NEE",
            "CRM", "UPS", "RTX", "ORCL", "HON", "INTC", "AMD", "QCOM", "IBM",
            "GE", "CAT", "BA", "SBUX", "PLD", "SPGI", "GS", "BLK", "AMAT",
            "ISRG", "MDLZ", "ADI", "SYK", "MMC", "BKNG", "GILD", "ADP", "VRTX",
            "REGN", "AMT", "LRCX", "CB", "ZTS", "NOW", "CI", "SCHW", "MO",
            "SO", "DUK", "BSX", "FIS", "ITW", "CL", "PNC", "USB", "TGT",
            "BDX", "HUM", "SHW", "MMM", "EQIX", "MU", "PYPL", "SNPS", "CDNS",
            "CME", "AON", "ICE", "KLAC", "APD", "EMR", "MCO", "NSC", "CSX"
        ]

    async def analyze_features_at_date(self, symbol: str, date: datetime) -> Dict[str, bool]:
        """
        Calcule les features/indicateurs pour un symbole à une date donnée.
        
        Returns:
            Dict des indicateurs actifs (True/False)
        """
        features = {}
        
        try:
            # Récupérer les données historiques
            start = (date - timedelta(days=250)).strftime("%Y-%m-%d")
            end = date.strftime("%Y-%m-%d")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            
            if df.empty or len(df) < 50:
                return features
            
            # Calculer les indicateurs
            df = self._calculate_indicators(df)
            
            # Extraire les features du dernier jour
            last = df.iloc[-1]
            
            # Trend features
            features["ema_cross_20_50"] = last.get("ema20", 0) > last.get("ema50", 0)
            features["ema_cross_50_200"] = last.get("ema50", 0) > last.get("ema200", 0)
            features["price_above_sma20"] = last["Close"] > last.get("sma20", last["Close"])
            features["price_above_sma50"] = last["Close"] > last.get("sma50", last["Close"])
            features["price_above_sma200"] = last["Close"] > last.get("sma200", last["Close"])
            features["adx_strong"] = last.get("adx", 0) > 25
            
            # Momentum features
            features["rsi_oversold"] = last.get("rsi", 50) < 30
            features["rsi_overbought"] = last.get("rsi", 50) > 70
            features["macd_bullish"] = last.get("macd", 0) > last.get("macd_signal", 0)
            features["macd_bearish"] = last.get("macd", 0) < last.get("macd_signal", 0)
            features["stoch_oversold"] = last.get("stoch_k", 50) < 20
            features["stoch_overbought"] = last.get("stoch_k", 50) > 80
            
            # Volume features
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            features["volume_spike"] = last["Volume"] > avg_vol * 1.5
            features["volume_breakout"] = last["Volume"] > avg_vol * 2
            features["obv_rising"] = df.get("obv", pd.Series()).diff().iloc[-5:].mean() > 0 if "obv" in df else False
            
            # Volatility features
            if "bb_upper" in df.columns and "bb_lower" in df.columns:
                bb_width = (last["bb_upper"] - last["bb_lower"]) / last["Close"]
                features["bollinger_squeeze"] = bb_width < 0.1
                features["price_near_lower_band"] = last["Close"] < last["bb_lower"] * 1.02
            
            features["atr_expansion"] = last.get("atr", 0) > df["atr"].rolling(20).mean().iloc[-1] if "atr" in df else False
            
        except Exception as e:
            logger.warning(f"Error analyzing features for {symbol}: {e}")
        
        return features
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs techniques."""
        # EMAs
        df["ema20"] = df["Close"].ewm(span=20).mean()
        df["ema50"] = df["Close"].ewm(span=50).mean()
        df["ema200"] = df["Close"].ewm(span=200).mean()
        
        # SMAs
        df["sma20"] = df["Close"].rolling(20).mean()
        df["sma50"] = df["Close"].rolling(50).mean()
        df["sma200"] = df["Close"].rolling(200).mean()
        
        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        # Stochastic
        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["stoch_k"] = 100 * (df["Close"] - low14) / (high14 - low14)
        
        # Bollinger Bands
        df["bb_middle"] = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std
        df["bb_lower"] = df["bb_middle"] - 2 * std
        
        # ATR
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift())
        low_close = abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # ADX (simplified)
        df["adx"] = 25  # Placeholder - full ADX calculation is complex
        
        # OBV
        df["obv"] = (df["Volume"] * ((df["Close"] > df["Close"].shift()).astype(int) * 2 - 1)).cumsum()
        
        return df

    async def run_feedback_cycle(self) -> Dict:
        """
        Exécute un cycle complet de feedback.
        
        1. Récupère les gainers/losers
        2. Analyse quels indicateurs étaient actifs la veille
        3. Met à jour les poids
        
        Returns:
            Résumé du cycle
        """
        logger.info("Starting feedback cycle...")
        
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        # 1. Récupérer les movers
        gainers, losers = await self.get_market_movers(today)
        
        if not gainers and not losers:
            logger.warning("No significant movers found")
            return {"status": "no_movers", "gainers": 0, "losers": 0}
        
        # 2. Analyser les features de la veille
        reinforced = []
        penalized = []
        
        # Pour les gainers: renforcer les indicateurs bullish
        for gainer in gainers:
            features = await self.analyze_features_at_date(gainer["symbol"], yesterday)
            bullish_features = [k for k, v in features.items() if v and "bullish" in k or "oversold" in k or "cross" in k]
            
            for feat in bullish_features:
                if feat in self.weights:
                    self._reinforce_indicator(feat, True)
                    if feat not in reinforced:
                        reinforced.append(feat)
            
            await asyncio.sleep(0.2)  # Rate limiting
        
        # Pour les losers: pénaliser les indicateurs bullish qui ont raté
        for loser in losers:
            features = await self.analyze_features_at_date(loser["symbol"], yesterday)
            bullish_features = [k for k, v in features.items() if v and "bullish" in k or "oversold" in k]
            
            for feat in bullish_features:
                if feat in self.weights:
                    self._reinforce_indicator(feat, False)
                    if feat not in penalized:
                        penalized.append(feat)
            
            await asyncio.sleep(0.2)
        
        # 3. Sauvegarder
        self._save_weights()
        self._save_feedback_history(gainers, losers, reinforced, penalized)
        
        # 4. Découvrir des patterns
        await self._discover_patterns(gainers, losers, yesterday)
        
        result = {
            "status": "success",
            "date": today.strftime("%Y-%m-%d"),
            "gainers_analyzed": len(gainers),
            "losers_analyzed": len(losers),
            "indicators_reinforced": reinforced,
            "indicators_penalized": penalized,
            "top_gainer": gainers[0] if gainers else None,
            "top_loser": losers[0] if losers else None
        }
        
        logger.info(f"Feedback cycle complete: {len(reinforced)} reinforced, {len(penalized)} penalized")
        
        return result
    
    def _reinforce_indicator(self, indicator: str, correct: bool):
        """Met à jour le poids d'un indicateur."""
        if indicator not in self.weights:
            return
        
        w = self.weights[indicator]
        w["total_signals"] += 1
        
        if correct:
            w["correct_signals"] += 1
            # Augmenter le poids (max 2.0)
            w["weight"] = min(2.0, w["weight"] * 1.05)
        else:
            # Diminuer le poids (min 0.1)
            w["weight"] = max(0.1, w["weight"] * 0.95)
        
        # Recalculer l'accuracy
        if w["total_signals"] > 0:
            w["accuracy"] = w["correct_signals"] / w["total_signals"]
        
        w["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    
    def _save_feedback_history(self, gainers: List, losers: List, reinforced: List, penalized: List):
        """Sauvegarde l'historique du feedback."""
        history = []
        if FEEDBACK_HISTORY_FILE.exists():
            with open(FEEDBACK_HISTORY_FILE) as f:
                history = json.load(f)
        
        history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "gainers": len(gainers),
            "losers": len(losers),
            "reinforced": reinforced,
            "penalized": penalized
        })
        
        # Garder les 30 derniers jours
        history = history[-30:]
        
        with open(FEEDBACK_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    
    async def _discover_patterns(self, gainers: List, losers: List, date: datetime):
        """Découvre des patterns gagnants/perdants."""
        # Analyser les combos d'indicateurs fréquents chez les gainers
        gainer_combos = {}
        for gainer in gainers[:10]:
            features = await self.analyze_features_at_date(gainer["symbol"], date)
            active = tuple(sorted([k for k, v in features.items() if v]))
            if active:
                gainer_combos[active] = gainer_combos.get(active, 0) + 1
        
        # Sauvegarder les patterns fréquents
        for combo, count in gainer_combos.items():
            if count >= 3:  # Au moins 3 occurrences
                pattern = {
                    "indicators": list(combo),
                    "count": count,
                    "date": date.strftime("%Y-%m-%d")
                }
                if pattern not in self.patterns["winning_combos"]:
                    self.patterns["winning_combos"].append(pattern)
        
        # Garder les 50 derniers patterns
        self.patterns["winning_combos"] = self.patterns["winning_combos"][-50:]
        self._save_patterns()
    
    def get_top_indicators(self, n: int = 10) -> List[Dict]:
        """Retourne les N meilleurs indicateurs par accuracy."""
        sorted_weights = sorted(
            self.weights.values(),
            key=lambda x: (x["accuracy"], x["total_signals"]),
            reverse=True
        )
        return sorted_weights[:n]
    
    def get_indicator_weight(self, indicator: str) -> float:
        """Retourne le poids d'un indicateur."""
        return self.weights.get(indicator, {}).get("weight", 1.0)


# Singleton
_feedback_loop: Optional[FeedbackLoop] = None

def get_feedback_loop() -> FeedbackLoop:
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop


async def run_daily_feedback():
    """Point d'entrée pour le cron quotidien."""
    loop = get_feedback_loop()
    return await loop.run_feedback_cycle()


if __name__ == "__main__":
    # Test
    async def test():
        result = await run_daily_feedback()
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
