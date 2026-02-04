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


async def generate_narrative(symbol: str) -> dict:
    """Generate a complete narrative report for a symbol."""
    
    # Initialize engine
    engine = ReasoningEngine()
    
    try:
        # Get pillar analysis
        result = await engine.analyze(symbol)
        
        # Use built-in to_dict
        report = result.to_dict()
        
        # Add French summary
        tech = report['pillar_scores']['technical']['score']
        fund = report['pillar_scores']['fundamental']['score']
        sent = report['pillar_scores']['sentiment']['score']
        news = report['pillar_scores']['news']['score']
        total = report['total_score']
        
        # Generate French executive summary
        parts = []
        
        if tech > 20:
            parts.append(f"signal technique HAUSSIER ({tech:.0f}/100)")
        elif tech < -20:
            parts.append(f"signal technique BAISSIER ({tech:.0f}/100)")
        else:
            parts.append(f"signal technique NEUTRE ({tech:.0f}/100)")
            
        if fund > 30:
            parts.append(f"fondamentaux SOLIDES ({fund:.0f}/100)")
        elif fund > 0:
            parts.append(f"fondamentaux CORRECTS ({fund:.0f}/100)")
        else:
            parts.append(f"fondamentaux FAIBLES ({fund:.0f}/100)")
            
        if sent > 40:
            parts.append(f"sentiment TRÈS POSITIF ({sent:.0f}/100)")
        elif sent > 10:
            parts.append(f"sentiment POSITIF ({sent:.0f}/100)")
        elif sent < -10:
            parts.append(f"sentiment NÉGATIF ({sent:.0f}/100)")
        else:
            parts.append(f"sentiment NEUTRE ({sent:.0f}/100)")
        
        report['resume_francais'] = f"${symbol}: " + ", ".join(parts) + f". Score global: {total:.1f}/100."
        
        # Add recommendation in French
        decision = report['decision']
        if decision == 'strong_buy':
            report['recommendation_fr'] = f"🟢 ACHAT FORT - Score de {total:.1f}/100 avec forte convergence des indicateurs."
        elif decision == 'buy':
            report['recommendation_fr'] = f"🟢 ACHAT - Score de {total:.1f}/100, opportunité intéressante."
        elif decision == 'hold':
            report['recommendation_fr'] = f"🟡 NEUTRE - Score de {total:.1f}/100, conserver ou attendre."
        elif decision == 'sell':
            report['recommendation_fr'] = f"🔴 VENTE - Score de {total:.1f}/100, signaux négatifs."
        else:
            report['recommendation_fr'] = f"🔴 VENTE FORTE - Score de {total:.1f}/100, éviter."
        
        return report
        
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python generate_narrative.py SYMBOL"}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    try:
        report = asyncio.run(generate_narrative(symbol))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
