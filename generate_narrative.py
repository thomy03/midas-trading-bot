#!/usr/bin/env python3
"""
Generate a narrative analysis report for a symbol.
Uses all 5 pillars + Grok/Gemini for explanations.
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, '/app')

from src.agents.reasoning_engine import ReasoningEngine
from src.intelligence.grok_scanner import GrokScanner


async def generate_narrative(symbol: str) -> dict:
    """Generate a complete narrative report for a symbol."""
    
    # Initialize engines
    engine = ReasoningEngine()
    grok = GrokScanner()
    await grok.initialize()
    
    try:
        # Get pillar analysis
        result = await engine.analyze(symbol)
        
        # Get Grok deep dive for context
        grok_insight = None
        if grok.is_available():
            grok_insight = await grok.analyze_symbol(symbol)
        
        # Build comprehensive report
        report = {
            "symbol": symbol,
            "timestamp": result.timestamp if hasattr(result, 'timestamp') else "",
            "score_total": round(result.total_score, 1),
            "signal": result.signal.value if hasattr(result.signal, 'value') else str(result.signal),
            "resume_executif": "",
            "pillars": {},
            "grok_analysis": None,
            "risques": [],
            "catalyseurs": [],
            "recommendation": ""
        }
        
        # Process each pillar with detailed explanation
        pillar_explanations = []
        for pr in result.pillar_results:
            pillar_data = {
                "score": round(pr.score, 1),
                "signal": pr.signal.value if hasattr(pr.signal, 'value') else str(pr.signal),
                "confidence": round(pr.confidence * 100),
                "reasoning": pr.reasoning,
                "factors": []
            }
            
            # Extract key factors
            if hasattr(pr, 'factors') and pr.factors:
                for factor in pr.factors[:5]:  # Top 5 factors
                    if isinstance(factor, dict):
                        pillar_data["factors"].append({
                            "name": factor.get("indicator") or factor.get("metric") or factor.get("source") or factor.get("category", ""),
                            "impact": factor.get("score", 0),
                            "detail": factor.get("message", "")
                        })
            
            report["pillars"][pr.pillar_name.lower()] = pillar_data
            pillar_explanations.append(f"{pr.pillar_name}: {pr.reasoning[:200]}")
        
        # Add Grok analysis if available
        if grok_insight:
            report["grok_analysis"] = {
                "sentiment": grok_insight.sentiment,
                "sentiment_score": round(grok_insight.sentiment_score, 2),
                "summary": grok_insight.summary,
                "key_points": grok_insight.key_points[:5],
                "catalysts": grok_insight.catalysts[:3],
                "risks": grok_insight.risk_factors[:3]
            }
            report["catalyseurs"] = grok_insight.catalysts[:3]
            report["risques"] = grok_insight.risk_factors[:3]
        
        # Generate executive summary
        tech_score = report["pillars"].get("technical", {}).get("score", 0)
        fund_score = report["pillars"].get("fundamental", {}).get("score", 0)
        sent_score = report["pillars"].get("sentiment", {}).get("score", 0)
        news_score = report["pillars"].get("news", {}).get("score", 0)
        
        summary_parts = []
        
        # Technical summary
        if tech_score > 20:
            summary_parts.append(f"Techniquement HAUSSIER (score {tech_score}/100)")
        elif tech_score < -20:
            summary_parts.append(f"Techniquement BAISSIER (score {tech_score}/100)")
        else:
            summary_parts.append(f"Techniquement NEUTRE (score {tech_score}/100)")
        
        # Fundamental summary
        if fund_score > 30:
            summary_parts.append(f"fondamentaux SOLIDES ({fund_score}/100)")
        elif fund_score < 10:
            summary_parts.append(f"fondamentaux FAIBLES ({fund_score}/100)")
        else:
            summary_parts.append(f"fondamentaux MOYENS ({fund_score}/100)")
        
        # Sentiment summary
        if sent_score > 40:
            summary_parts.append(f"sentiment TRÈS POSITIF sur les réseaux ({sent_score}/100)")
        elif sent_score > 20:
            summary_parts.append(f"sentiment POSITIF ({sent_score}/100)")
        elif sent_score < -20:
            summary_parts.append(f"sentiment NÉGATIF ({sent_score}/100)")
        else:
            summary_parts.append(f"sentiment NEUTRE ({sent_score}/100)")
        
        report["resume_executif"] = f"${symbol}: " + ", ".join(summary_parts) + "."
        
        # Final recommendation
        if result.total_score >= 70:
            report["recommendation"] = f"ACHAT FORT recommandé - Score global de {report['score_total']}/100 avec convergence positive des piliers."
        elif result.total_score >= 55:
            report["recommendation"] = f"ACHAT suggéré - Score de {report['score_total']}/100, opportunité intéressante à surveiller."
        elif result.total_score >= 45:
            report["recommendation"] = f"NEUTRE - Score de {report['score_total']}/100, attendre de meilleurs signaux."
        else:
            report["recommendation"] = f"ÉVITER - Score faible de {report['score_total']}/100, risque élevé."
        
        return report
        
    finally:
        await grok.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python generate_narrative.py SYMBOL"}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    try:
        report = asyncio.run(generate_narrative(symbol))
        print(json.dumps(report, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
