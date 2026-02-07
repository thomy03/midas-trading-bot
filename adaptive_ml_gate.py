#!/usr/bin/env python3
"""
ADAPTIVE ML GATE - S'active selon la volatilité
- Volatilité haute (>3%) → 5 Piliers seuls
- Volatilité normale → 5 Piliers + ML Gate
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

VOLATILITY_THRESHOLD = 0.03  # 3% daily vol = désactive ML Gate

class AdaptiveMLGate:
    def __init__(self):
        self.model = None
        self.X_train = []
        self.y_train = []
        
    def add_sample(self, features, outcome):
        self.X_train.append(features)
        self.y_train.append(outcome)
        
    def train(self):
        if len(self.X_train) < 100:
            return False
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        self.model.fit(X, y)
        return True
        
    def get_confidence(self, features):
        if self.model is None:
            return 0.5
        try:
            return self.model.predict_proba([features])[0][1]
        except:
            return 0.5
    
    def apply_adaptive_gate(self, base_score, features, volatility):
        """
        Applique le ML Gate SEULEMENT si volatilité basse
        """
        if volatility > VOLATILITY_THRESHOLD:
            # Volatilité haute → pas de ML Gate
            return base_score, "5P_ONLY"
        
        # Volatilité normale → appliquer ML Gate
        conf = self.get_confidence(features)
        if conf > 0.6:
            return base_score + 5, "ML_BOOST"
        elif conf < 0.4:
            return 0, "ML_BLOCK"
        return base_score, "ML_NEUTRAL"

def calculate_volatility(close, idx, period=20):
    """Calcule la volatilité sur les N derniers jours"""
    if idx < period:
        return 0.02  # Default
    returns = np.diff(close[idx-period:idx+1]) / close[idx-period:idx]
    return np.std(returns)

def calculate_pillars(df, idx, spy_data, spy_idx):
    close = df['Close'].values.flatten()
    
    # Technical
    tech = 50
    if idx >= 50:
        ema9 = pd.Series(close[:idx+1]).ewm(span=9).mean().iloc[-1]
        ema20 = pd.Series(close[:idx+1]).ewm(span=20).mean().iloc[-1]
        ema50 = pd.Series(close[:idx+1]).ewm(span=50).mean().iloc[-1]
        if ema9 > ema20 > ema50: tech += 20
        elif ema9 < ema20 < ema50: tech -= 15
        
        delta = pd.Series(close[:idx+1]).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        if 30 < rsi < 50: tech += 15
        elif rsi > 70: tech -= 10
    tech = max(0, min(100, tech))
    
    # Fundamental
    fund = 50
    if idx >= 60:
        mom60 = (close[idx] / close[idx-60] - 1) * 100
        if mom60 > 10: fund += 20
        elif mom60 > 5: fund += 10
        elif mom60 < -10: fund -= 15
    fund = max(0, min(100, fund))
    
    # Sentiment
    sent = 50
    if idx >= 5:
        mom5 = (close[idx] / close[idx-5] - 1) * 100
        if mom5 > 3: sent += 15
        elif mom5 < -3: sent -= 10
    sent = max(0, min(100, sent))
    
    # News
    news = 50
    open_vals = df['Open'].values.flatten()
    if idx >= 1:
        gap = (open_vals[idx] / close[idx-1] - 1) * 100
        if gap > 2: news += 15
        elif gap < -2: news -= 15
    news = max(0, min(100, news))
    
    # Regime
    regime = 50
    if spy_data is not None and spy_idx >= 50 and spy_idx < len(spy_data):
        spy_close = spy_data['Close'].values.flatten()
        trend = (spy_close[spy_idx] / spy_close[spy_idx-50] - 1) * 100
        if trend > 5: regime = 70
        elif trend < -5: regime = 30
    regime = max(0, min(100, regime))
    
    return {'technical': tech, 'fundamental': fund, 'sentiment': sent, 'news': news, 'ml_regime': regime}

def run_adaptive_comparison():
    print("=" * 70)
    print("ADAPTIVE ML GATE - Volatility-Based Switching")
    print(f"Threshold: {VOLATILITY_THRESHOLD*100:.1f}% daily volatility")
    print("=" * 70)
    
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX',
        'JPM', 'JNJ', 'XOM', 'V', 'MA', 'HD', 'MC.PA', 'SAP.DE', 'AVGO', 'CAT'
    ]
    
    print(f"\nDownloading SPY...")
    spy = yf.download('SPY', start='2010-01-01', end='2025-01-01', progress=False)
    
    WEIGHTS = {'technical': 35, 'fundamental': 20, 'sentiment': 15, 'news': 10, 'ml_regime': 20}
    
    results = {'5p': {}, 'ml_fixed': {}, 'ml_adaptive': {}}
    ml_fixed = AdaptiveMLGate()
    ml_adaptive = AdaptiveMLGate()
    
    print(f"Testing {len(symbols)} symbols...\n")
    
    for symbol in symbols:
        try:
            df = yf.download(symbol, start='2010-01-01', end='2025-01-01', progress=False)
            if len(df) < 100:
                continue
            
            close = df['Close'].values.flatten()
            
            # Stats
            stats = {'5p': {'trades': 0, 'pnl': 0}, 
                    'ml_fixed': {'trades': 0, 'pnl': 0},
                    'ml_adaptive': {'trades': 0, 'pnl': 0, 'mode_5p': 0, 'mode_ml': 0}}
            
            pos_5p = pos_ml_fixed = pos_ml_adaptive = None
            
            for i in range(60, len(df) - 10):
                date = df.index[i]
                price = close[i]
                spy_idx = spy.index.get_indexer([date], method='nearest')[0]
                
                pillars = calculate_pillars(df, i, spy, spy_idx)
                total_score = sum(pillars[k] * WEIGHTS[k] for k in WEIGHTS) / 100
                
                ml_features = [pillars[k] for k in ['technical', 'fundamental', 'sentiment', 'news', 'ml_regime']]
                volatility = calculate_volatility(close, i)
                
                # 5P Only
                if pos_5p is None:
                    if total_score >= 50:
                        pos_5p = {'entry': price, 'days': 0}
                else:
                    pos_5p['days'] += 1
                    if price <= pos_5p['entry'] * 0.92 or pos_5p['days'] >= 60:
                        ret = (price / pos_5p['entry'] - 1) * 100
                        stats['5p']['pnl'] += ret
                        stats['5p']['trades'] += 1
                        pos_5p = None
                
                # ML Fixed (always apply ML Gate)
                conf_fixed = ml_fixed.get_confidence(ml_features)
                score_ml_fixed = total_score + 5 if conf_fixed > 0.6 else (0 if conf_fixed < 0.4 else total_score)
                
                if pos_ml_fixed is None:
                    if score_ml_fixed >= 50:
                        pos_ml_fixed = {'entry': price, 'days': 0, 'pillars': pillars}
                else:
                    pos_ml_fixed['days'] += 1
                    if price <= pos_ml_fixed['entry'] * 0.92 or pos_ml_fixed['days'] >= 60:
                        ret = (price / pos_ml_fixed['entry'] - 1) * 100
                        stats['ml_fixed']['pnl'] += ret
                        stats['ml_fixed']['trades'] += 1
                        outcome = 1 if ret > 0 else 0
                        ml_fixed.add_sample([pos_ml_fixed['pillars'][k] for k in ['technical', 'fundamental', 'sentiment', 'news', 'ml_regime']], outcome)
                        if len(ml_fixed.X_train) % 200 == 0:
                            ml_fixed.train()
                        pos_ml_fixed = None
                
                # ML ADAPTIVE (apply ML Gate only if low volatility)
                score_adaptive, mode = ml_adaptive.apply_adaptive_gate(total_score, ml_features, volatility)
                
                if pos_ml_adaptive is None:
                    if score_adaptive >= 50:
                        pos_ml_adaptive = {'entry': price, 'days': 0, 'pillars': pillars}
                        if mode == "5P_ONLY":
                            stats['ml_adaptive']['mode_5p'] += 1
                        else:
                            stats['ml_adaptive']['mode_ml'] += 1
                else:
                    pos_ml_adaptive['days'] += 1
                    if price <= pos_ml_adaptive['entry'] * 0.92 or pos_ml_adaptive['days'] >= 60:
                        ret = (price / pos_ml_adaptive['entry'] - 1) * 100
                        stats['ml_adaptive']['pnl'] += ret
                        stats['ml_adaptive']['trades'] += 1
                        outcome = 1 if ret > 0 else 0
                        ml_adaptive.add_sample([pos_ml_adaptive['pillars'][k] for k in ['technical', 'fundamental', 'sentiment', 'news', 'ml_regime']], outcome)
                        if len(ml_adaptive.X_train) % 200 == 0:
                            ml_adaptive.train()
                        pos_ml_adaptive = None
            
            # Symbol avg volatility
            avg_vol = np.std(np.diff(close) / close[:-1]) * 100
            mode_ratio = stats['ml_adaptive']['mode_5p'] / max(1, stats['ml_adaptive']['mode_5p'] + stats['ml_adaptive']['mode_ml']) * 100
            
            results['5p'][symbol] = stats['5p']
            results['ml_fixed'][symbol] = stats['ml_fixed']
            results['ml_adaptive'][symbol] = stats['ml_adaptive']
            
            print(f"{symbol:6} | Vol={avg_vol:.1f}% | 5P={stats['5p']['pnl']:+.0f}% | ML Fixed={stats['ml_fixed']['pnl']:+.0f}% | Adaptive={stats['ml_adaptive']['pnl']:+.0f}% (5P mode: {mode_ratio:.0f}%)")
            
        except Exception as e:
            print(f"{symbol}: Error - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    total_5p = sum(r['pnl'] for r in results['5p'].values())
    total_ml_fixed = sum(r['pnl'] for r in results['ml_fixed'].values())
    total_ml_adaptive = sum(r['pnl'] for r in results['ml_adaptive'].values())
    
    print(f"\n{'Mode':<20} {'Total P&L':>15} {'vs 5P':>15}")
    print("-" * 55)
    print(f"{'5 Pillars':<20} {total_5p:>+14.0f}% {'-':>15}")
    print(f"{'ML Fixed':<20} {total_ml_fixed:>+14.0f}% {total_ml_fixed - total_5p:>+14.0f}%")
    print(f"{'ML ADAPTIVE':<20} {total_ml_adaptive:>+14.0f}% {total_ml_adaptive - total_5p:>+14.0f}%")
    
    print("\n" + "=" * 70)
    if total_ml_adaptive > total_ml_fixed:
        print("✅ ML ADAPTIVE WINS!")
    else:
        print("❌ ML Fixed still better")
    print("=" * 70)

if __name__ == '__main__':
    run_adaptive_comparison()
