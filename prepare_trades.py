#!/usr/bin/env python3
"""
Midas - Prepare Trades Script v2
Scans full universe (US + Europe) with user-configurable filters.
Streams progress to logs for real-time visibility.
"""
import asyncio
import json
import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/app")

from datetime import datetime
from pathlib import Path
import logging

# Setup logging to stderr for real-time visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

async def get_universe(max_stocks: int = None, max_price: float = None, markets: list = None):
    """Get stock universe with optional filters."""
    symbols = []
    
    try:
        # Try to use the UniverseScanner
        from src.intelligence.universe_scanner import UniverseScanner, UniverseConfig
        
        config = UniverseConfig()
        scanner = UniverseScanner(config)
        
        logger.info("Loading full universe (US + Europe)...")
        snapshot = await scanner.scan_full_universe()
        
        all_symbols = snapshot.us_symbols + snapshot.eu_symbols
        logger.info(f"Universe loaded: {len(all_symbols)} symbols")
        
        # Apply filters
        for s in all_symbols:
            # Filter by max price (capital)
            if max_price and s.price > max_price:
                continue
            # Filter by markets
            if markets and s.market not in markets:
                continue
            symbols.append({
                "symbol": s.symbol,
                "name": s.name,
                "price": s.price,
                "market": s.market,
                "sector": s.sector
            })
        
        # Limit count if specified
        if max_stocks and len(symbols) > max_stocks:
            symbols = symbols[:max_stocks]
            
    except Exception as e:
        logger.warning(f"UniverseScanner failed: {e}, using fallback list")
        # Fallback to default list
        symbols = get_fallback_universe(max_stocks, max_price)
    
    return symbols


def get_fallback_universe(max_stocks: int = None, max_price: float = None):
    """Fallback universe when scanner fails."""
    # Extended US + Europe list
    STOCKS = [
        # US Tech
        {"symbol": "AAPL", "name": "Apple", "market": "US"},
        {"symbol": "MSFT", "name": "Microsoft", "market": "US"},
        {"symbol": "GOOGL", "name": "Alphabet", "market": "US"},
        {"symbol": "AMZN", "name": "Amazon", "market": "US"},
        {"symbol": "NVDA", "name": "NVIDIA", "market": "US"},
        {"symbol": "META", "name": "Meta", "market": "US"},
        {"symbol": "TSLA", "name": "Tesla", "market": "US"},
        {"symbol": "AVGO", "name": "Broadcom", "market": "US"},
        {"symbol": "AMD", "name": "AMD", "market": "US"},
        {"symbol": "CRM", "name": "Salesforce", "market": "US"},
        {"symbol": "ADBE", "name": "Adobe", "market": "US"},
        {"symbol": "NFLX", "name": "Netflix", "market": "US"},
        {"symbol": "INTC", "name": "Intel", "market": "US"},
        {"symbol": "QCOM", "name": "Qualcomm", "market": "US"},
        {"symbol": "ORCL", "name": "Oracle", "market": "US"},
        # US Finance
        {"symbol": "JPM", "name": "JPMorgan", "market": "US"},
        {"symbol": "V", "name": "Visa", "market": "US"},
        {"symbol": "MA", "name": "Mastercard", "market": "US"},
        {"symbol": "BAC", "name": "Bank of America", "market": "US"},
        {"symbol": "GS", "name": "Goldman Sachs", "market": "US"},
        # US Healthcare
        {"symbol": "UNH", "name": "UnitedHealth", "market": "US"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "market": "US"},
        {"symbol": "PFE", "name": "Pfizer", "market": "US"},
        {"symbol": "LLY", "name": "Eli Lilly", "market": "US"},
        {"symbol": "MRK", "name": "Merck", "market": "US"},
        # US Consumer
        {"symbol": "WMT", "name": "Walmart", "market": "US"},
        {"symbol": "HD", "name": "Home Depot", "market": "US"},
        {"symbol": "COST", "name": "Costco", "market": "US"},
        {"symbol": "NKE", "name": "Nike", "market": "US"},
        {"symbol": "MCD", "name": "McDonalds", "market": "US"},
        # US Energy/Industrial
        {"symbol": "XOM", "name": "ExxonMobil", "market": "US"},
        {"symbol": "CVX", "name": "Chevron", "market": "US"},
        {"symbol": "CAT", "name": "Caterpillar", "market": "US"},
        {"symbol": "BA", "name": "Boeing", "market": "US"},
        {"symbol": "HON", "name": "Honeywell", "market": "US"},
        # Europe - CAC 40
        {"symbol": "MC.PA", "name": "LVMH", "market": "EU"},
        {"symbol": "OR.PA", "name": "LOreal", "market": "EU"},
        {"symbol": "TTE.PA", "name": "TotalEnergies", "market": "EU"},
        {"symbol": "SAN.PA", "name": "Sanofi", "market": "EU"},
        {"symbol": "AIR.PA", "name": "Airbus", "market": "EU"},
        {"symbol": "SU.PA", "name": "Schneider", "market": "EU"},
        {"symbol": "BNP.PA", "name": "BNP Paribas", "market": "EU"},
        {"symbol": "AI.PA", "name": "Air Liquide", "market": "EU"},
        {"symbol": "KER.PA", "name": "Kering", "market": "EU"},
        {"symbol": "RMS.PA", "name": "Hermes", "market": "EU"},
        # Europe - DAX
        {"symbol": "SAP.DE", "name": "SAP", "market": "EU"},
        {"symbol": "SIE.DE", "name": "Siemens", "market": "EU"},
        {"symbol": "ALV.DE", "name": "Allianz", "market": "EU"},
        {"symbol": "DTE.DE", "name": "Deutsche Telekom", "market": "EU"},
        {"symbol": "BMW.DE", "name": "BMW", "market": "EU"},
        {"symbol": "VOW3.DE", "name": "Volkswagen", "market": "EU"},
        {"symbol": "MRK.DE", "name": "Merck KGaA", "market": "EU"},
        {"symbol": "BAS.DE", "name": "BASF", "market": "EU"},
        # Europe - Other
        {"symbol": "ASML.AS", "name": "ASML", "market": "EU"},
        {"symbol": "NESN.SW", "name": "Nestle", "market": "EU"},
        {"symbol": "NOVN.SW", "name": "Novartis", "market": "EU"},
        {"symbol": "ROG.SW", "name": "Roche", "market": "EU"},
        {"symbol": "SHEL.L", "name": "Shell", "market": "EU"},
        {"symbol": "AZN.L", "name": "AstraZeneca", "market": "EU"},
        {"symbol": "ULVR.L", "name": "Unilever", "market": "EU"},
    ]
    
    if max_stocks:
        return STOCKS[:max_stocks]
    return STOCKS


async def analyze_symbol(symbol: str, data: dict = None) -> dict:
    """Analyze a single symbol and return score."""
    try:
        from src.data.market_data import MarketDataFetcher
        from src.agents.pillars.technical_pillar import TechnicalPillar
        
        fetcher = MarketDataFetcher()
        df = fetcher.get_historical_data(symbol, period="3mo", interval="1d")
        
        if df is None or len(df) < 20:
            return None
        
        current_price = float(df["Close"].iloc[-1])
        pillar_data = {"df": df, "symbol": symbol}
        
        # Technical score
        tech = TechnicalPillar(weight=0.30)
        tech_result = await tech.analyze(symbol, pillar_data)
        tech_score = max(0, min(100, (tech_result.score + 100) / 2))
        
        # Other pillars
        fund_score = sent_score = ml_score = news_score = 50
        
        try:
            from src.agents.pillars.fundamental_pillar import FundamentalPillar
            f = FundamentalPillar(weight=0.25)
            r = await f.analyze(symbol, pillar_data)
            fund_score = max(0, min(100, (r.score + 100) / 2))
        except: pass
        
        try:
            from src.agents.pillars.sentiment_pillar import SentimentPillar
            s = SentimentPillar(weight=0.20)
            r = await s.analyze(symbol, pillar_data)
            sent_score = max(0, min(100, (r.score + 100) / 2))
        except: pass
        
        try:
            from src.agents.pillars.news_pillar import NewsPillar
            n = NewsPillar(weight=0.10)
            r = await n.analyze(symbol, pillar_data)
            news_score = max(0, min(100, (r.score + 100) / 2))
        except: pass
        
        try:
            from src.agents.pillars.ml_pillar import MLPillar
            m = MLPillar(weight=0.15)
            r = await m.analyze(symbol, pillar_data)
            ml_score = max(0, min(100, (r.score + 100) / 2))
        except: pass
        
        # Final score
        final = (
            tech_score * 0.30 + 
            fund_score * 0.25 + 
            sent_score * 0.15 + 
            news_score * 0.10 + 
            ml_score * 0.20
        )
        
        if final >= 75: decision = "STRONG_BUY"
        elif final >= 70: decision = "BUY"
        elif final >= 60: decision = "HOLD"
        elif final >= 50: decision = "WEAK"
        else: decision = "AVOID"
        
        return {
            "symbol": symbol,
            "name": data.get("name", symbol) if data else symbol,
            "market": data.get("market", "US") if data else "US",
            "score": round(final, 1),
            "decision": decision,
            "confidence": round(min(tech_score, fund_score, sent_score), 0),
            "price": round(current_price, 2),
            "pillars": {
                "technical": round(tech_score, 1),
                "fundamental": round(fund_score, 1),
                "sentiment": round(sent_score, 1),
                "news": round(news_score, 1),
                "ml": round(ml_score, 1)
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None


async def prepare_trades(max_stocks: int = None, max_price: float = None, min_score: int = 60):
    """Run analysis and return top candidates."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "candidates": [],
        "status": "running",
        "total_scanned": 0,
        "config": {
            "max_stocks": max_stocks,
            "max_price": max_price,
            "min_score": min_score
        }
    }
    
    # Progress file for real-time updates
    progress_file = Path("/tmp/midas-prepare-progress.json")
    
    try:
        # Get universe
        universe = await get_universe(max_stocks, max_price)
        results["total_scanned"] = len(universe)
        
        logger.info(f"Starting analysis of {len(universe)} stocks...")
        
        candidates = []
        
        for i, stock in enumerate(universe):
            symbol = stock["symbol"] if isinstance(stock, dict) else stock
            name = stock.get("name", symbol) if isinstance(stock, dict) else symbol
            
            # Log progress
            if (i + 1) % 10 == 0 or i == 0:
                pct = round((i + 1) / len(universe) * 100)
                logger.info(f"[{i+1}/{len(universe)}] ({pct}%) Analyzing {symbol}...")
                
                # Update progress file
                progress = {
                    "current": i + 1,
                    "total": len(universe),
                    "percent": pct,
                    "current_symbol": symbol,
                    "candidates_found": len(candidates)
                }
                progress_file.write_text(json.dumps(progress))
            
            result = await analyze_symbol(symbol, stock if isinstance(stock, dict) else None)
            
            if result and result["score"] >= min_score:
                candidates.append(result)
                logger.info(f"  ✓ {symbol}: {result[score]} ({result[decision]})")
            
            await asyncio.sleep(0.2)  # Rate limiting
        
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        results["candidates"] = candidates[:20]  # Top 20
        results["all_candidates_count"] = len(candidates)
        results["status"] = "success"
        
        logger.info('='*50)
        logger.info(f"DONE! Found {len(candidates)} candidates with score >= {min_score}")
        if candidates:
            logger.info(f"Top 5:")
            for c in candidates[:5]:
                logger.info(f"  {c['symbol']}: {c['score']} ({c['decision']})")
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()
    
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description="Midas - Prepare Trades")
    parser.add_argument("--max-stocks", type=int, default=None, help="Max stocks to analyze (default: all)")
    parser.add_argument("--max-price", type=float, default=None, help="Max stock price (capital filter)")
    parser.add_argument("--min-score", type=int, default=60, help="Min score to include (default: 60)")
    args = parser.parse_args()
    
    asyncio.run(prepare_trades(args.max_stocks, args.max_price, args.min_score))


if __name__ == "__main__":
    main()
