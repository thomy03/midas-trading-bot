#!/usr/bin/env python3
"""
Market Analysis Report Generator
Generates human-readable investment reports using LLM.
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
import httpx

sys.path.insert(0, '/app')

# Try to import from the app
try:
    from src.data.market_data import MarketDataFetcher
    from src.agents.state import get_state_manager
except ImportError:
    pass


async def generate_report() -> dict:
    """Generate a comprehensive market analysis report."""
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('GROK_API_KEY')
    model = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-exp:free')
    
    if not api_key:
        return {"error": "No API key configured"}
    
    # Collect market data
    market_data = await collect_market_data()
    
    # Generate report using LLM
    report = await generate_llm_report(api_key, model, market_data)
    
    return report


async def collect_market_data() -> dict:
    """Collect all relevant market data for the report."""
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%d %B %Y"),
        "market_hours": "closed" if datetime.now().hour < 9 or datetime.now().hour >= 16 else "open"
    }
    
    # Try to get scan results
    try:
        scan_file = Path("/app/data/latest_scan.json")
        if scan_file.exists():
            with open(scan_file) as f:
                data["scan_results"] = json.load(f)
    except:
        data["scan_results"] = {}
    
    # Try to get Grok sentiment
    try:
        grok_file = Path("/app/data/grok_sentiment.json")
        if grok_file.exists():
            with open(grok_file) as f:
                data["grok_sentiment"] = json.load(f)
    except:
        data["grok_sentiment"] = {}
    
    # Try to get portfolio state
    try:
        state_file = Path("/app/data/agent_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                data["portfolio"] = {
                    "capital": state.get("current_capital", 1500),
                    "positions": state.get("open_positions", []),
                    "trades_today": state.get("trades_today", 0)
                }
    except:
        data["portfolio"] = {"capital": 1500, "positions": [], "trades_today": 0}
    
    # Try to get top opportunities
    try:
        opps_file = Path("/app/data/opportunities.json")
        if opps_file.exists():
            with open(opps_file) as f:
                data["opportunities"] = json.load(f)
    except:
        data["opportunities"] = []
    
    # Try to get recent analyses
    try:
        analyses_dir = Path("/app/data/analyses")
        if analyses_dir.exists():
            recent = []
            for f in sorted(analyses_dir.glob("*.json"), reverse=True)[:10]:
                with open(f) as file:
                    recent.append(json.load(file))
            data["recent_analyses"] = recent
    except:
        data["recent_analyses"] = []
    
    return data


async def generate_llm_report(api_key: str, model: str, market_data: dict) -> dict:
    """Use LLM to generate a well-written report."""
    
    # Determine API endpoint
    if api_key.startswith("xai-"):
        api_url = "https://api.x.ai/v1/chat/completions"
        model = "grok-3-latest"
    else:
        api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    report_date = market_data.get('date', "Aujourd'hui")
    prompt = f"""Tu es un analyste financier senior. Génère un rapport d'analyse de marché professionnel en français.

DONNÉES DU SCAN:
{json.dumps(market_data, indent=2, default=str)}

CONSIGNES:
- Écris un rapport structuré et argumenté
- Utilise un ton professionnel mais accessible
- Justifie chaque recommandation
- Identifie les risques
- Donne ton avis d'expert

FORMAT DU RAPPORT (en Markdown):

# 📈 Rapport d'Analyse - {report_date}

## 1. Contexte de Marché
[Analyse du sentiment global, conditions macro-économiques]

## 2. Analyse Sectorielle
[Quels secteurs sont favorables/défavorables et pourquoi]

## 3. Opportunités Identifiées
[Liste des meilleures opportunités avec justification détaillée pour chacune]

## 4. État du Portefeuille
[Analyse des positions actuelles, performance]

## 5. Recommandations
[Actions concrètes à prendre avec argumentation]

## 6. Risques & Points de Vigilance
[Facteurs de risque à surveiller]

## 7. Conclusion
[Synthèse et perspectives à court terme]

---
*Rapport généré automatiquement par TradingBot V5*
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            if not api_key.startswith("xai-"):
                headers["HTTP-Referer"] = "https://tradingbot.local"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Tu es un analyste financier expert. Tu rédiges des rapports d'analyse de marché professionnels, clairs et argumentés."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10000,
                "temperature": 0.7
            }
            
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            report_content = result["choices"][0]["message"]["content"]
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "date": market_data.get("date"),
                "content": report_content,
                "market_data": market_data
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def save_report(report: dict):
    """Save report to file."""
    reports_dir = Path("/app/data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"report_{date_str}.json"
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Also save markdown version
    if report.get("content"):
        md_file = reports_dir / f"report_{date_str}.md"
        with open(md_file, "w") as f:
            f.write(report["content"])
    
    return str(report_file)


if __name__ == "__main__":
    report = asyncio.run(generate_report())
    
    if report.get("success"):
        filepath = save_report(report)
        print(json.dumps({
            "success": True,
            "filepath": filepath,
            "content": report["content"]
        }, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False))
