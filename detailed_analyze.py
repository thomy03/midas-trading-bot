#!/usr/bin/env python3
"""
Detailed Stock Analysis - Generates written analysis like a financial analyst.
With SL/TP levels calculation.
"""
import asyncio
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/app')

from datetime import datetime
import httpx
import numpy as np

from src.data.market_data import MarketDataFetcher
from src.agents.pillars.technical_pillar import TechnicalPillar


def calculate_levels(df, current_price):
    """Calculate SL/TP levels based on ATR and support/resistance."""
    
    # Calculate ATR (14 periods)
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.mean(tr[-14:])
    
    # Find support/resistance
    recent_lows = df['Low'].tail(20).min()
    recent_highs = df['High'].tail(20).max()
    
    # Calculate levels
    # Stop Loss: 2x ATR below current price or recent support
    sl_atr = current_price - (2 * atr)
    sl_support = recent_lows * 0.98  # 2% below recent low
    stop_loss = max(sl_atr, sl_support)
    
    # Take Profit 1: 2x ATR above (R:R = 1:1)
    tp1 = current_price + (2 * atr)
    
    # Take Profit 2: 3x ATR above (R:R = 1:1.5) or resistance
    tp2_atr = current_price + (3 * atr)
    tp2 = min(tp2_atr, recent_highs * 1.05)
    
    # Calculate percentages
    sl_pct = ((stop_loss - current_price) / current_price) * 100
    tp1_pct = ((tp1 - current_price) / current_price) * 100
    tp2_pct = ((tp2 - current_price) / current_price) * 100
    
    # Risk/Reward ratio
    risk = current_price - stop_loss
    reward1 = tp1 - current_price
    reward2 = tp2 - current_price
    rr1 = round(reward1 / risk, 1) if risk > 0 else 0
    rr2 = round(reward2 / risk, 1) if risk > 0 else 0
    
    return {
        'stopLoss': round(stop_loss, 2),
        'stopLossPercent': round(sl_pct, 2),
        'takeProfit1': round(tp1, 2),
        'takeProfit1Percent': round(tp1_pct, 2),
        'takeProfit2': round(tp2, 2),
        'takeProfit2Percent': round(tp2_pct, 2),
        'riskRewardRatio': f"1:{rr1} / 1:{rr2}",
        'atr': round(atr, 2),
        'support': round(recent_lows, 2),
        'resistance': round(recent_highs, 2)
    }


async def detailed_analyze(symbol: str) -> dict:
    """Generate a detailed written analysis for a symbol."""
    
    symbol = symbol.upper().strip()
    
    # Get market data
    fetcher = MarketDataFetcher()
    df = fetcher.get_historical_data(symbol, period='6mo', interval='1d')
    
    if df is None or len(df) < 20:
        return {'error': f'No data for {symbol}', 'symbol': symbol}
    
    # Get current price info
    current_price = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
    change_pct = ((current_price - prev_close) / prev_close) * 100
    
    # Calculate basic metrics
    high_52w = float(df['High'].max())
    low_52w = float(df['Low'].min())
    avg_volume = int(df['Volume'].mean())
    
    # Calculate SL/TP levels
    levels = calculate_levels(df, current_price)
    
    # Technical analysis
    tech = TechnicalPillar(weight=0.30)
    data = {'df': df, 'symbol': symbol}
    tech_score = await tech.analyze(symbol, data)
    
    raw = tech_score.score
    tech_normalized = max(0, min(100, (raw + 100) / 2 if -100 <= raw <= 100 else raw))
    
    # Get other pillar scores
    fund_score = sent_score = ml_score = 50
    fund_details = sent_details = ml_details = ""
    
    try:
        from src.agents.pillars.fundamental_pillar import FundamentalPillar
        f = FundamentalPillar(weight=0.25)
        r = await f.analyze(symbol, data)
        fund_score = max(0, min(100, (r.score + 100) / 2 if -100 <= r.score <= 100 else r.score))
        fund_details = r.reasoning if hasattr(r, 'reasoning') else ""
    except: pass
    
    try:
        from src.agents.pillars.sentiment_pillar import SentimentPillar
        s = SentimentPillar(weight=0.20)
        r = await s.analyze(symbol, data)
        sent_score = max(0, min(100, (r.score + 100) / 2 if -100 <= r.score <= 100 else r.score))
        sent_details = r.reasoning if hasattr(r, 'reasoning') else ""
    except: pass
    
    try:
        from src.agents.pillars.ml_pillar import MLPillar
        m = MLPillar(weight=0.15)
        r = await m.analyze(symbol, data)
        ml_score = max(0, min(100, (r.score + 100) / 2 if -100 <= r.score <= 100 else r.score))
        ml_details = r.reasoning if hasattr(r, 'reasoning') else ""
    except: pass
    
    # Calculate final score
    final = tech_normalized * 0.30 + fund_score * 0.25 + sent_score * 0.20 + 50 * 0.10 + ml_score * 0.15
    
    if final >= 70: decision = 'STRONG_BUY'
    elif final >= 55: decision = 'BUY'
    elif final >= 45: decision = 'HOLD'
    elif final >= 30: decision = 'SELL'
    else: decision = 'STRONG_SELL'
    
    # Generate detailed analysis using LLM
    analysis_text = await generate_analysis_text(
        symbol, current_price, change_pct, high_52w, low_52w,
        tech_normalized, fund_score, sent_score, ml_score, final, decision,
        tech_score.reasoning if hasattr(tech_score, 'reasoning') else "",
        fund_details, sent_details, levels
    )
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'currentPrice': current_price,
        'change': round(change_pct, 2),
        'high52w': high_52w,
        'low52w': low_52w,
        'avgVolume': avg_volume,
        'finalScore': round(final),
        'decision': decision,
        'confidence': 85,
        'pillars': [
            {'name': 'Technical', 'score': round(tech_normalized), 'weight': 30, 'details': (tech_score.reasoning or '')[:2000]},
            {'name': 'Fundamental', 'score': round(fund_score), 'weight': 25, 'details': fund_details[:2000] if fund_details else 'Financial health'},
            {'name': 'Sentiment', 'score': round(sent_score), 'weight': 20, 'details': sent_details[:2000] if sent_details else 'Social sentiment'},
            {'name': 'News', 'score': 50, 'weight': 10, 'details': 'News impact'},
            {'name': 'ML Adaptive', 'score': round(ml_score), 'weight': 15, 'details': ml_details[:2000] if ml_details else 'Pattern recognition'}
        ],
        'levels': levels,
        'analysis': analysis_text
    }


async def generate_analysis_text(symbol, price, change, high, low, tech, fund, sent, ml, final, decision, tech_details, fund_details, sent_details, levels):
    """Generate written analysis using LLM."""
    
    api_key = os.getenv('GROK_API_KEY') or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        return generate_fallback_analysis(symbol, price, change, tech, fund, sent, final, decision, levels)
    
    # Determine API
    if api_key.startswith("xai-"):
        api_url = "https://api.x.ai/v1/chat/completions"
        model = "grok-3-latest"
    else:
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        model = "google/gemini-2.0-flash-exp:free"
    
    prompt = f"""Tu es un analyste financier senior. Rédige une analyse complète et professionnelle pour {symbol}.

DONNÉES:
- Prix actuel: ${price:.2f} ({'+' if change >= 0 else ''}{change:.2f}%)
- Plus haut 52 semaines: ${high:.2f}
- Plus bas 52 semaines: ${low:.2f}
- Score technique: {tech:.0f}/100
- Score fondamental: {fund:.0f}/100  
- Score sentiment: {sent:.0f}/100
- Score ML: {ml:.0f}/100
- Score final: {final:.0f}/100
- Recommandation: {decision}

NIVEAUX DE TRADING CALCULÉS:
- Stop Loss: ${levels['stopLoss']:.2f} ({levels['stopLossPercent']:.1f}%)
- Take Profit 1: ${levels['takeProfit1']:.2f} (+{levels['takeProfit1Percent']:.1f}%)
- Take Profit 2: ${levels['takeProfit2']:.2f} (+{levels['takeProfit2Percent']:.1f}%)
- Ratio Risk/Reward: {levels['riskRewardRatio']}
- ATR (volatilité): ${levels['atr']:.2f}
- Support: ${levels['support']:.2f}
- Résistance: ${levels['resistance']:.2f}

Détails techniques: {tech_details[:2000]}
Détails fondamentaux: {fund_details[:2000]}
Détails sentiment: {sent_details[:2000]}

CONSIGNES:
Rédige une analyse en français, structurée ainsi:

**Contexte de marché:** (2-3 phrases sur la position actuelle)

**Analyse technique:** (tendance, signaux clés, momentum)

**Analyse fondamentale:** (santé financière, valorisation)

**Sentiment & Social:** (ce que disent les investisseurs)

**🎯 Niveaux de trading suggérés:**
- Stop Loss: [niveau et justification]
- Take Profit 1: [niveau et justification]  
- Take Profit 2: [niveau et justification]
- Ratio Risk/Reward: [évaluation]

**Conclusion:** (synthèse et recommandation argumentée)

Sois concis mais précis. Les niveaux de trading sont ESSENTIELS - c'est notre valeur ajoutée."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            if not api_key.startswith("xai-"):
                headers["HTTP-Referer"] = "https://tradingbot.local"
            
            response = await client.post(api_url, headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10000,
                "temperature": 0.7
            })
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return generate_fallback_analysis(symbol, price, change, tech, fund, sent, final, decision, levels)


def generate_fallback_analysis(symbol, price, change, tech, fund, sent, final, decision, levels):
    """Generate basic analysis without LLM."""
    
    trend = "haussière" if change > 0 else "baissière" if change < 0 else "neutre"
    tech_view = "positif" if tech >= 55 else "négatif" if tech <= 45 else "neutre"
    fund_view = "solide" if fund >= 55 else "fragile" if fund <= 45 else "moyenne"
    sent_view = "optimiste" if sent >= 55 else "pessimiste" if sent <= 45 else "mitigé"
    
    reco_text = {
        'STRONG_BUY': "Achat fort recommandé",
        'BUY': "Opportunité d'achat",
        'HOLD': "Conserver la position",
        'SELL': "Alléger la position",
        'STRONG_SELL': "Vente recommandée"
    }.get(decision, "Position neutre")
    
    return f"""**Contexte de marché:**
{symbol} affiche une tendance {trend} avec un prix actuel de ${price:.2f} ({'+' if change >= 0 else ''}{change:.2f}%). 

**Analyse technique:**
Les indicateurs techniques montrent un signal {tech_view} (score: {tech:.0f}/100). La dynamique de prix suggère une continuation de la tendance actuelle.

**Analyse fondamentale:**
La santé financière de l'entreprise apparaît {fund_view} (score: {fund:.0f}/100). Les métriques de valorisation sont à surveiller.

**Sentiment & Social:**
Le sentiment du marché est {sent_view} (score: {sent:.0f}/100).

**🎯 Niveaux de trading suggérés:**
- **Stop Loss:** ${levels['stopLoss']:.2f} ({levels['stopLossPercent']:.1f}%) - basé sur 2x ATR et support récent
- **Take Profit 1:** ${levels['takeProfit1']:.2f} (+{levels['takeProfit1Percent']:.1f}%) - objectif conservateur
- **Take Profit 2:** ${levels['takeProfit2']:.2f} (+{levels['takeProfit2Percent']:.1f}%) - objectif ambitieux
- **Ratio Risk/Reward:** {levels['riskRewardRatio']}
- **Support:** ${levels['support']:.2f} | **Résistance:** ${levels['resistance']:.2f}

**Conclusion:**
Avec un score global de {final:.0f}/100, notre analyse suggère: **{reco_text}**. Cette recommandation est basée sur l'ensemble des facteurs analysés."""


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: detailed_analyze.py SYMBOL'}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    result = asyncio.run(detailed_analyze(symbol))
    print(json.dumps(result, ensure_ascii=False))
