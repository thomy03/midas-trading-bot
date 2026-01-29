"""
Report Generator - Génération de rapports .md
Crée des rapports lisibles au format Markdown.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ReportGenerator:
    """Génère des rapports Markdown pour le bot de trading."""

    def __init__(self, reports_dir: str = "data/reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate_daily_report(
        self,
        discovery_result: Optional[Dict] = None,
        analysis_result: Optional[Dict] = None,
        trading_result: Optional[Dict] = None,
        chain_of_thought: Optional[List[str]] = None
    ) -> str:
        """
        Génère un rapport quotidien complet.

        Returns:
            Chemin du fichier .md généré
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        lines = [
            f"# 📊 Rapport Trading - {date_str}",
            f"",
            f"*Généré à {time_str}*",
            f"",
        ]

        # === Chain of Thought ===
        if chain_of_thought:
            lines.extend([
                "## 🧠 Raisonnement (Chain of Thought)",
                "",
            ])
            for thought in chain_of_thought[-20:]:  # Last 20 thoughts
                lines.append(f"- {thought}")
            lines.append("")

        # === Discovery ===
        if discovery_result:
            lines.extend(self._format_discovery(discovery_result))

        # === Analysis ===
        if analysis_result:
            lines.extend(self._format_analysis(analysis_result))

        # === Trading ===
        if trading_result:
            lines.extend(self._format_trading(trading_result))

        # Footer
        lines.extend([
            "",
            "---",
            f"*Rapport généré par TradingBot V4.1 - {now.isoformat()}*"
        ])

        # Save report
        content = "\n".join(lines)
        filename = f"report_{now.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.reports_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        # Also save as latest
        latest_path = os.path.join(self.reports_dir, "latest_report.md")
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return filepath

    def _format_discovery(self, result: Dict) -> List[str]:
        """Formate la section Discovery."""
        lines = [
            "## 🔍 Phase Discovery",
            "",
        ]

        # Social Trending
        social = result.get('social_trending', [])
        if social:
            lines.append(f"### 🌐 Social Trending ({len(social)} symboles)")
            lines.append("")
            lines.append(f"**Top 10:** {', '.join(social[:10])}")
            lines.append("")

        # Grok Insights
        grok = result.get('grok_insights', [])
        if grok:
            lines.append(f"### 🐦 Grok/X Insights ({len(grok)} analyses)")
            lines.append("")
            for insight in grok[:5]:
                if isinstance(insight, dict):
                    topic = insight.get('topic', 'N/A')
                    sentiment = insight.get('sentiment', 'N/A')
                    symbols = insight.get('symbols', [])
                    lines.append(f"- **{topic}**: {sentiment}")
                    if symbols:
                        lines.append(f"  - Symboles: {', '.join(symbols[:5])}")
            lines.append("")

        # Volume Anomalies
        anomalies = result.get('volume_anomalies', [])
        if anomalies:
            lines.append(f"### 📈 Anomalies de Volume ({len(anomalies)})")
            lines.append("")
            lines.append(f"{', '.join(anomalies[:10])}")
            lines.append("")

        # Watchlist
        watchlist = result.get('watchlist', [])
        lines.append(f"### 📋 Watchlist Finale: {len(watchlist)} symboles")
        lines.append("")

        return lines

    def _format_analysis(self, result: Dict) -> List[str]:
        """Formate la section Analysis."""
        lines = [
            "## 📊 Phase Analyse",
            "",
        ]

        # Market Sentiment
        sentiment = result.get('market_sentiment', 0)
        sentiment_emoji = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "🟡"
        lines.append(f"**Sentiment Marché:** {sentiment_emoji} {sentiment:+.2f}")
        lines.append("")

        # Market Regime
        regime = result.get('market_regime', 'UNKNOWN')
        lines.append(f"**Régime:** {regime}")
        lines.append("")

        # Trends
        trends = result.get('trends', [])
        if trends:
            lines.append(f"### 📈 Tendances Détectées ({len(trends)})")
            lines.append("")
            for trend in trends[:5]:
                name = trend.get('name', 'N/A')
                strength = trend.get('strength', 'N/A')
                confidence = trend.get('confidence', 0)
                lines.append(f"- **{name}** - Force: {strength}, Confiance: {confidence:.0%}")
            lines.append("")

        # Narratives
        narratives = result.get('narratives', [])
        if narratives:
            lines.append(f"### 📰 Narratifs Emergents ({len(narratives)})")
            lines.append("")
            for narrative in narratives[:3]:
                name = narrative.get('name', 'N/A')
                desc = narrative.get('description', '')[:100]
                lines.append(f"- **{name}**: {desc}")
            lines.append("")

        # Focus Symbols
        focus = result.get('focus_symbols', [])
        if focus:
            lines.append(f"### 🎯 Focus Symbols ({len(focus)})")
            lines.append("")
            lines.append(f"{', '.join(focus[:15])}")
            lines.append("")

        return lines

    def _format_trading(self, result: Dict) -> List[str]:
        """Formate la section Trading."""
        lines = [
            "## 💹 Phase Trading",
            "",
        ]

        scanned = result.get('scanned_symbols', 0)
        signals = result.get('signals_found', 0)
        executed = result.get('trades_executed', 0)
        rejected = result.get('trades_rejected', 0)

        lines.extend([
            f"- **Symboles scannés:** {scanned}",
            f"- **Signaux trouvés:** {signals}",
            f"- **Trades exécutés:** {executed}",
            f"- **Trades rejetés:** {rejected}",
            "",
        ])

        # Alerts
        alerts = result.get('alerts', [])
        if alerts:
            lines.append(f"### 🚨 Alertes ({len(alerts)})")
            lines.append("")
            lines.append("| Symbole | Signal | Score | Prix |")
            lines.append("|---------|--------|-------|------|")
            for alert in alerts[:10]:
                symbol = alert.get('symbol', 'N/A')
                signal = alert.get('signal', 'N/A')
                score = alert.get('confidence_score', 0)
                price = alert.get('price', 0)
                lines.append(f"| {symbol} | {signal} | {score}/100 | ${price:.2f} |")
            lines.append("")

        return lines

    def generate_thinking_log(self, thoughts: List[Dict]) -> str:
        """
        Génère un fichier de log des pensées/raisonnements.

        Args:
            thoughts: Liste de pensées avec timestamp et contenu

        Returns:
            Chemin du fichier généré
        """
        now = datetime.now()
        filename = f"thinking_{now.strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.reports_dir, filename)

        lines = [
            f"# 🧠 Log de Raisonnement - {now.strftime('%Y-%m-%d')}",
            "",
        ]

        for thought in thoughts:
            ts = thought.get('timestamp', '')
            content = thought.get('content', '')
            category = thought.get('category', 'general')
            lines.append(f"### [{ts}] {category.upper()}")
            lines.append(f"{content}")
            lines.append("")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return filepath


# Singleton
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Retourne le singleton ReportGenerator."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
