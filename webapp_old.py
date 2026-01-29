"""
TradingBot V4.1 - Modern Web Dashboard
Interface web moderne avec NiceGUI pour controler le robot de trading.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta, time
from typing import Optional, Dict, List, Literal
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nicegui import ui, app
from dotenv import load_dotenv

load_dotenv(override=True)  # Override system env vars with .env values


NotifyType = Literal['positive', 'negative', 'warning', 'info', 'ongoing']

def safe_notify(message: str, type: NotifyType = 'info', **kwargs):
    """
    Safe notification that works in both UI context and background tasks.
    Falls back to logging if no UI context is available.
    """
    try:
        ui.notify(message, type=type, **kwargs)
    except Exception:
        # No UI context (background task) - log instead
        level = {'positive': 'info', 'negative': 'error', 'warning': 'warning', 'info': 'info'}.get(type, 'info')
        getattr(logger, level)(f"[Notification] {message}")

# V4.3 - Persistent Analysis Store
from src.intelligence.analysis_store import get_analysis_store, AnalysisStore

# V4.5 - Explanation Helper
from src.utils.explanation_helper import (
    get_four_pillars_explanation,
    get_score_interpretation,
    get_discovery_explanation,
    get_guardrails_explanation,
    get_timeline_explanation
)

# V4.6 - Narrative Generator for LLM-generated reports
from src.intelligence.narrative_generator import NarrativeGenerator, get_narrative_generator

# V4.6 - Reasoning Graph for neural network-like visualization
from src.agents.reasoning_graph import (
    ReasoningGraph, ReasoningGraphBuilder, ReasoningGraphStore,
    get_graph_builder, get_graph_store, NodeType
)


# =============================================================================
# V4.5 - SCORE CONVERSION HELPERS
# =============================================================================
# Internal scores use -100 to +100 range
# Display scores use 0 to 100 range for better UX

def convert_score_to_display(internal_score: float) -> float:
    """Convert internal [-100, +100] score to display [0, 100]

    Examples:
        -100 -> 0 (very bearish)
        0 -> 50 (neutral)
        +100 -> 100 (very bullish)

    Also handles already-converted scores (0-100 range).
    """
    if internal_score is None:
        return 50  # Default neutral

    # Detect if score is already in display format (0-100)
    # Internal scores are -100 to +100, so negative = definitely internal
    if internal_score < 0:
        # Internal format: convert
        converted = (internal_score + 100) / 2
    elif internal_score > 100:
        # Invalid: cap at 100
        converted = 100
    else:
        # Could be either format - assume internal if came from pillars
        # For total scores, we assume 0-100 range if positive
        converted = (internal_score + 100) / 2 if internal_score <= 100 else internal_score

    # Clamp to valid range
    return round(max(0, min(100, converted)), 1)


def convert_pillar_score_to_display(internal_score: float, data_quality: float = None) -> float:
    """Convert internal pillar score to display [0, 25] scale

    Each pillar contributes max 25 points to total score.
    Internal: -100 to +100 -> Display: 0 to 25

    Examples:
        -100 -> 0 (very negative contribution)
        0 -> 12.5 (neutral) - but only if data_quality > 0
        +100 -> 25 (max positive contribution)
        None or 0 with data_quality=0 -> 0 (no data)

    ROBUST: Also handles scores that are already in various formats.
    """
    # V4.8: If no data (None or 0 with data_quality=0), return 0 not 12.5
    if internal_score is None:
        return 0.0  # No data = 0, not neutral 12.5

    # If score is exactly 0 and data_quality is 0, it's "no data" not "neutral"
    if internal_score == 0 and (data_quality is not None and data_quality == 0):
        return 0.0

    # Detect score format and convert appropriately
    if internal_score < 0:
        # Definitely internal format (-100 to 0): convert to 0-12.5
        converted = ((internal_score + 100) / 200) * 25
    elif internal_score > 100:
        # Score > 100 is invalid, must be a bug - cap at 25
        converted = 25
    elif internal_score > 25:
        # Score between 25 and 100: likely internal format (0 to +100)
        # Convert: 0->12.5, 100->25
        converted = ((internal_score + 100) / 200) * 25
    else:
        # Score 0-25: assume already in display format
        converted = internal_score

    # Clamp to valid range [0, 25]
    return round(max(0, min(25, converted)), 1)


def convert_total_score(pillar_scores: list) -> float:
    """Calculate total display score from pillar scores

    Args:
        pillar_scores: List of internal pillar scores [-100, +100]

    Returns:
        Total display score [0, 100]
    """
    if not pillar_scores:
        return 50  # Default neutral
    # Each pillar is already weighted at 25%, so we sum the display scores
    display_scores = [convert_pillar_score_to_display(s) for s in pillar_scores]
    return round(sum(display_scores), 1)


# =============================================================================
# MODERN CSS STYLING (Glassmorphism + Gradients)
# =============================================================================

MODERN_CSS = '''
<style>
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: rgba(15, 23, 42, 0.8);
        --bg-card: rgba(30, 41, 59, 0.6);
        --border-color: rgba(71, 85, 105, 0.4);
        --accent-green: #10b981;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-orange: #f97316;
    }

    body {
        background: linear-gradient(135deg, #0a0f1a 0%, #1e1b4b 50%, #0a0f1a 100%) !important;
        background-attachment: fixed !important;
        min-height: 100vh;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(71, 85, 105, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15) !important;
    }

    .stat-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(71, 85, 105, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stat-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
    }

    .gradient-text-green {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .gradient-text-blue {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .gradient-text-purple {
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .modern-btn {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }

    .modern-btn:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.4) 0%, rgba(139, 92, 246, 0.4) 100%) !important;
        transform: translateY(-1px) !important;
    }

    .nav-btn {
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        margin: 2px 0 !important;
    }

    .nav-btn:hover {
        background: rgba(99, 102, 241, 0.2) !important;
    }

    .header-gradient {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.95) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(71, 85, 105, 0.3) !important;
    }

    .sidebar-gradient {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 15, 26, 0.98) 100%) !important;
        backdrop-filter: blur(10px) !important;
    }

    .pulse-green {
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    }

    .glow-border {
        position: relative;
    }

    .glow-border::before {
        content: '';
        position: absolute;
        inset: -1px;
        border-radius: inherit;
        padding: 1px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.5) 0%, rgba(139, 92, 246, 0.5) 100%);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .glow-border:hover::before {
        opacity: 1;
    }

    /* Pipeline step cards */
    .pipeline-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }

    .pipeline-card:hover {
        transform: scale(1.02) !important;
    }

    /* Table styling */
    .q-table {
        background: transparent !important;
    }

    .q-table th {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #94a3b8 !important;
    }

    .q-table td {
        background: rgba(30, 41, 59, 0.4) !important;
        border-color: rgba(71, 85, 105, 0.3) !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.7);
    }

    /* V4.5 Timeline Styles for Flux de Pensées */
    .timeline-container {
        position: relative;
        padding-left: 40px;
    }

    .timeline-container::before {
        content: '';
        position: absolute;
        left: 16px;
        top: 0;
        bottom: 0;
        width: 2px;
        background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        border-radius: 2px;
    }

    .timeline-item {
        position: relative;
        padding-bottom: 16px;
        animation: fadeInUp 0.3s ease-out;
    }

    .timeline-item:last-child {
        padding-bottom: 0;
    }

    .timeline-dot {
        position: absolute;
        left: -32px;
        top: 4px;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        border: 2px solid #1e293b;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
        z-index: 1;
    }

    .timeline-dot.discovery { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); }
    .timeline-dot.social { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .timeline-dot.grok { background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); }
    .timeline-dot.news { background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); }
    .timeline-dot.insight { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }
    .timeline-dot.decision { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }
    .timeline-dot.error { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .timeline-dot.wait { background: linear-gradient(135deg, #64748b 0%, #475569 100%); }
    .timeline-dot.general { background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%); }

    .timeline-content {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 12px 16px;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }

    .timeline-content:hover {
        background: rgba(30, 41, 59, 0.95);
        transform: translateX(4px);
    }

    .timeline-content.discovery { border-left-color: #3b82f6; }
    .timeline-content.social { border-left-color: #10b981; }
    .timeline-content.grok { border-left-color: #06b6d4; }
    .timeline-content.news { border-left-color: #eab308; }
    .timeline-content.insight { border-left-color: #f97316; }
    .timeline-content.decision { border-left-color: #8b5cf6; }
    .timeline-content.error { border-left-color: #ef4444; }
    .timeline-content.wait { border-left-color: #64748b; }
    .timeline-content.general { border-left-color: #94a3b8; }

    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }

    .timeline-category {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .timeline-time {
        font-size: 11px;
        color: #64748b;
        font-family: monospace;
    }

    .timeline-text {
        font-size: 14px;
        line-height: 1.5;
        color: #e2e8f0;
    }

    .timeline-section-header {
        position: relative;
        padding: 8px 0;
        margin-bottom: 8px;
    }

    .timeline-section-header::after {
        content: '';
        position: absolute;
        left: -24px;
        right: 0;
        bottom: 0;
        height: 1px;
        background: linear-gradient(90deg, #6366f1 0%, transparent 100%);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* V4.6 Structured Thoughts Styles */
    .thought-phase-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
    }

    .thought-phase-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .thought-phase-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }

    .thought-phase-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .thought-phase-icon.discovery { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); }
    .thought-phase-icon.social { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .thought-phase-icon.grok { background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); }
    .thought-phase-icon.news { background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%); }
    .thought-phase-icon.insight { background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); }
    .thought-phase-icon.decision { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }

    .thought-section {
        margin-bottom: 16px;
    }

    .thought-section-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .thought-bullet-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .thought-bullet-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 12px;
        margin-bottom: 4px;
        background: rgba(51, 65, 85, 0.4);
        border-radius: 8px;
        font-size: 14px;
        color: #e2e8f0;
        line-height: 1.5;
    }

    .thought-bullet-item::before {
        content: '•';
        color: #6366f1;
        font-weight: bold;
        font-size: 16px;
        line-height: 1.3;
    }

    .thought-bullet-item.highlight {
        background: rgba(99, 102, 241, 0.15);
        border-left: 3px solid #6366f1;
    }

    .thought-bullet-item.warning {
        background: rgba(234, 179, 8, 0.1);
        border-left: 3px solid #eab308;
    }

    .thought-bullet-item.success {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
    }

    .thought-bullet-item.error {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
    }

    .thought-key-insight {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }

    .thought-key-insight-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .thought-explore-box {
        background: rgba(6, 182, 212, 0.1);
        border: 1px dashed rgba(6, 182, 212, 0.4);
        border-radius: 10px;
        padding: 14px;
        margin-top: 12px;
    }

    .thought-explore-title {
        font-size: 12px;
        font-weight: 600;
        color: #22d3ee;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .thought-timestamp {
        font-size: 11px;
        color: #64748b;
        font-family: monospace;
    }

    .thought-empty-state {
        text-align: center;
        padding: 60px 20px;
    }

    .thought-empty-icon {
        font-size: 64px;
        margin-bottom: 16px;
        opacity: 0.5;
    }
</style>
'''


def apply_modern_styles():
    """Apply modern CSS styles to the page"""
    ui.add_head_html(MODERN_CSS)


def create_help_button(title: str, content: str, icon: str = 'help_outline'):
    """
    V4.5: Create a help button that shows explanation in a dialog.

    Args:
        title: Dialog title
        content: Markdown content to display
        icon: Icon name (default: help_outline)
    """
    def show_help():
        with ui.dialog() as dlg, ui.card().classes('glass-card w-[600px] max-h-[80vh]'):
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label(f'📚 {title}').classes('text-white text-xl font-bold')
                ui.button(icon='close', on_click=dlg.close).props('flat round size=sm').classes('text-slate-400')

            with ui.scroll_area().classes('max-h-[60vh]'):
                ui.markdown(content).classes('text-slate-200 prose prose-invert prose-sm')

            ui.button('Compris !', on_click=dlg.close, icon='check').classes('mt-4 w-full')
        dlg.open()

    return ui.button(icon=icon, on_click=show_help).props('flat round size=sm').classes('text-blue-400 hover:text-blue-300').tooltip(f'Aide: {title}')


# =============================================================================
# GLOBAL STATE
# =============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.bot_running = False
        self.bot_status = "Arrete"
        self.bot_mode = "OFF"  # OFF, PRE_MARKET, MARKET, AFTER_HOURS
        self.last_scan = None
        self.last_discovery = None
        self.alerts: List[Dict] = []
        self.premarket_queue: List[Dict] = []  # Alerts found outside market hours
        self.trades: List[Dict] = []
        self.capital = float(os.getenv('TRADING_CAPITAL', 1500))
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.components_status = {}
        self.logs: List[str] = []
        self.agent = None
        self._task = None
        self._discovery_task = None
        # Chain of Thought - Reasoning display
        self.chain_of_thought: List[Dict] = []
        # Discovery/Analysis results for display
        self.last_discovery_result: Optional[Dict] = None
        self.last_analysis_result: Optional[Dict] = None
        self.last_trading_result: Optional[Dict] = None
        # V4.2 - Narrative reports (human-readable analysis)
        self.narrative_reports: Dict[str, str] = {}  # symbol -> markdown report

        # V4.3 - SYMBOL JOURNEY TRACKING (Complete Chain of Thought per symbol)
        # Chaque symbole a son propre parcours avec toutes les étapes détaillées
        self.symbol_journeys: Dict[str, Dict] = {}
        # Structure:
        # {
        #     "NVDA": {
        #         "symbol": "NVDA",
        #         "current_status": "signal",  # discovered, analyzing, signal, bought, sold, rejected
        #         "current_score": 78,
        #         "first_seen": "2024-12-29T10:00:00",
        #         "last_update": "2024-12-29T10:30:00",
        #         "journey": [
        #             {
        #                 "step": "discovery",
        #                 "timestamp": "...",
        #                 "title": "Détecté via Reddit + Grok",
        #                 "reasoning": "47 mentions/h sur r/wallstreetbets (baseline: 12). Thèmes: Blackwell, CES 2025",
        #                 "data": {"reddit_mentions": 47, "sentiment": 0.72, "themes": ["Blackwell", "CES"]}
        #             },
        #             {
        #                 "step": "analysis",
        #                 "timestamp": "...",
        #                 "title": "Analyse 4 Piliers",
        #                 "reasoning": "Technical: 68/100 (RSI breakout confirmé). Fundamental: 75/100 (P/E: 45, croissance 35%)...",
        #                 "data": {"technical": 68, "fundamental": 75, "sentiment": 80, "news": 72, "total": 74}
        #             },
        #             {
        #                 "step": "signal",
        #                 "timestamp": "...",
        #                 "title": "Signal BUY généré",
        #                 "reasoning": "Score 74/100 > seuil 55. Convergence 4/4 piliers positifs.",
        #                 "data": {"signal": "BUY", "score": 74, "confidence": 0.82}
        #             }
        #         ]
        #     }
        # }

        self.focus_symbols: List[str] = []  # Symbols selected for active analysis
        self.watchlist: List[str] = []  # Watchlist symbols

        # V4.3 - Recently analyzed symbols (to avoid repetition)
        self.recently_analyzed: Dict[str, str] = {}  # symbol -> last_analysis_timestamp

        # V4.2 - Capital/PnL history for visualization
        self.capital_history: List[Dict] = []  # [{timestamp, capital, event, symbol, details}]
        self.actions_timeline: List[Dict] = []  # [{timestamp, action, symbol, price, reason}]

state = AppState()


# =============================================================================
# SYMBOL JOURNEY TRACKING HELPERS - V4.3 PERSISTENT STORE
# =============================================================================

# Initialize the persistent analysis store
_analysis_store: Optional[AnalysisStore] = None

def _get_store() -> AnalysisStore:
    """Get or create the analysis store singleton"""
    global _analysis_store
    if _analysis_store is None:
        _analysis_store = get_analysis_store()
    return _analysis_store


def load_persisted_journeys():
    """
    V4.5: Load persisted journeys into in-memory state at startup.
    This ensures previously analyzed symbols are visible after app restart.
    """
    store = _get_store()
    persistent = store.get_all_journeys()

    loaded_count = 0
    for symbol, journey in persistent.items():
        if symbol not in state.symbol_journeys:
            # Convert SymbolJourney to dict format
            state.symbol_journeys[symbol] = journey.to_dict()
            loaded_count += 1

    return loaded_count


def add_journey_step(symbol: str, step: str, title: str, reasoning: str, data: Dict = None):
    """
    Ajoute une étape au parcours d'un symbole.
    V4.3: Uses persistent AnalysisStore instead of in-memory dict.

    Args:
        symbol: Ticker du symbole
        step: Type d'étape (discovery, analysis, signal, buy, sell, reject)
        title: Titre court de l'étape
        reasoning: Explication détaillée de la décision
        data: Données associées (scores, métriques, etc.)
    """
    store = _get_store()
    store.add_journey_step(
        symbol=symbol,
        step=step,
        title=title,
        reasoning=reasoning,
        data=data
    )

    # Also update in-memory state for real-time display
    timestamp = datetime.now().isoformat()

    # V4.5: Convert internal score to display format [0-100]
    def get_display_score(d):
        if not d:
            return 50  # Neutral default
        raw = d.get('score', d.get('total', None))
        if raw is None:
            return 50
        return convert_score_to_display(raw)

    if symbol not in state.symbol_journeys:
        state.symbol_journeys[symbol] = {
            "symbol": symbol,
            "current_status": step,
            "current_score": get_display_score(data),
            "first_seen": timestamp,
            "last_update": timestamp,
            "journey": []
        }

    journey_entry = {
        "step": step,
        "timestamp": timestamp,
        "title": title,
        "reasoning": reasoning,
        "data": data or {}
    }

    state.symbol_journeys[symbol]["journey"].append(journey_entry)
    state.symbol_journeys[symbol]["current_status"] = step
    state.symbol_journeys[symbol]["last_update"] = timestamp

    # V4.5: Update score with display format conversion
    if data:
        if 'score' in data or 'total' in data:
            state.symbol_journeys[symbol]["current_score"] = get_display_score(data)


def get_journey_for_symbol(symbol: str) -> Optional[Dict]:
    """
    Récupère le parcours complet d'un symbole.
    V4.3: Checks persistent store first, then in-memory.
    """
    # First check in-memory for real-time data
    if symbol in state.symbol_journeys:
        return state.symbol_journeys.get(symbol)

    # Fall back to persistent store
    store = _get_store()
    journey = store.get_journey(symbol)
    if journey:
        return journey.to_dict()
    return None


def get_all_journeys_sorted() -> List[Dict]:
    """
    Retourne tous les parcours triés par dernière mise à jour.
    V4.3: Combines persistent store with in-memory data.
    """
    store = _get_store()

    # V4.5: Get from persistent store (increased limit from 100 to 500)
    persistent_journeys = store.get_journeys_sorted(max_count=500)
    result = {j.symbol: j.to_dict() for j in persistent_journeys}

    # Merge with in-memory (in-memory takes precedence for fresh data)
    for symbol, journey in state.symbol_journeys.items():
        # Use .get() to avoid KeyError if last_update doesn't exist
        journey_update = journey.get("last_update") or journey.get("updated_at") or ""
        result_update = result.get(symbol, {}).get("updated_at") or result.get(symbol, {}).get("last_update") or ""
        if symbol not in result or journey_update > result_update:
            result[symbol] = journey

    # Sort by last update
    journeys = list(result.values())
    # Handle both "last_update" and "updated_at" keys
    def get_update_time(j):
        return j.get("last_update") or j.get("updated_at") or ""
    return sorted(journeys, key=get_update_time, reverse=True)


def get_analysis_history(symbol: str, limit: int = 10) -> List[Dict]:
    """
    V4.3: Get historical analyses for a symbol from persistent store.
    Allows viewing how analysis has evolved over time.
    """
    store = _get_store()
    return store.get_analysis_history(symbol, limit)


def get_score_trend(symbol: str, days: int = 30) -> List[Dict]:
    """
    V4.3: Get score trend for a symbol over time.
    Useful for visualizing analysis evolution.
    """
    store = _get_store()
    return store.get_score_trend(symbol, days)


def compare_symbol_analyses(symbol: str) -> Optional[Dict]:
    """
    V4.3: Compare current vs previous analysis.
    Returns delta for each pillar score.
    """
    store = _get_store()
    return store.compare_analyses(symbol)


# =============================================================================
# V4.6 - NARRATIVE REPORT GENERATION VIA LLM
# =============================================================================

# Cache for generated narratives (symbol -> narrative_data)
_narrative_cache: Dict[str, Dict] = {}
_narrative_generator: Optional[NarrativeGenerator] = None


async def generate_symbol_narrative(symbol: str, journey: Dict) -> Dict:
    """
    Generate a narrative analysis report for a symbol using LLM.

    Returns a dict with:
    - watchlist_summary: Why this symbol caught attention
    - pillar_explanations: Detailed explanation for each pillar
    - decision_reasoning: Why buy/not buy
    - reservations: Risks and concerns
    - exit_triggers: When to exit
    """
    global _narrative_generator

    # Check cache first
    cache_key = f"{symbol}_{journey.get('last_update', '')[:10]}"
    if cache_key in _narrative_cache:
        return _narrative_cache[cache_key]

    # Initialize generator if needed
    if _narrative_generator is None:
        _narrative_generator = NarrativeGenerator()
        await _narrative_generator.initialize()

    # Extract data from journey
    steps = journey.get('journey', [])
    analysis_data = next((s.get('data', {}) for s in steps if s.get('step') == 'analysis'), {})
    discovery_data = next((s.get('data', {}) for s in steps if s.get('step') == 'discovery'), {})

    # Build technical/fundamental data from pillar_details
    pillar_details = analysis_data.get('pillar_details', {})

    technical_data = pillar_details.get('technical', {})
    fundamental_data = pillar_details.get('fundamental', {})
    news_data = pillar_details.get('news', {})

    # Generate narrative
    try:
        narrative = await _narrative_generator.generate_analysis(
            symbol=symbol,
            social_data=discovery_data.get('social', {}),
            grok_data=discovery_data.get('grok', {}),
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            news_data=news_data
        )

        result = {
            'watchlist_summary': narrative.watchlist_summary,
            'pillar_explanations': narrative.pillar_explanations,
            'decision_reasoning': narrative.decision_reasoning,
            'reservations': narrative.reservations,
            'exit_triggers': narrative.exit_triggers,
            'decision': narrative.decision,
            'timestamp': narrative.timestamp
        }

        # Cache the result
        _narrative_cache[cache_key] = result

        # Also store in state for persistence
        state.narrative_reports[symbol] = narrative.to_markdown()

        return result

    except Exception as e:
        logger.error(f"Failed to generate narrative for {symbol}: {e}")
        return {
            'watchlist_summary': f"Analyse de {symbol} en cours...",
            'pillar_explanations': {},
            'decision_reasoning': "Génération du rapport en cours...",
            'reservations': "",
            'exit_triggers': [],
            'decision': 'WATCH',
            'error': str(e)
        }


def get_cached_narrative(symbol: str) -> Optional[Dict]:
    """Get a cached narrative if available"""
    for key, narrative in _narrative_cache.items():
        if key.startswith(symbol + "_"):
            return narrative
    return None


# =============================================================================
# MARKET HOURS UTILITIES
# =============================================================================

# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# Market hours (US Eastern)
MARKET_OPEN = time(9, 30)   # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
PREMARKET_START = time(4, 0)  # 4:00 AM ET (pre-market opens)

def get_current_market_mode() -> str:
    """Get current market mode based on time"""
    now_et = datetime.now(ET)
    current_time = now_et.time()
    weekday = now_et.weekday()

    # Weekend = AFTER_HOURS (market closed)
    if weekday >= 5:
        return "WEEKEND"

    # Market hours
    if MARKET_OPEN <= current_time < MARKET_CLOSE:
        return "MARKET"

    # Pre-market (4:00 AM - 9:30 AM ET)
    if PREMARKET_START <= current_time < MARKET_OPEN:
        return "PRE_MARKET"

    # After hours
    return "AFTER_HOURS"

def is_market_open() -> bool:
    """Check if US market is currently open"""
    return get_current_market_mode() == "MARKET"

def get_time_to_market_open() -> Optional[timedelta]:
    """Get time remaining until market opens"""
    now_et = datetime.now(ET)
    current_time = now_et.time()
    weekday = now_et.weekday()

    # If market is open, return None
    if is_market_open():
        return None

    # Calculate next market open
    if weekday >= 5:  # Weekend
        days_until_monday = 7 - weekday
        next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
    elif current_time >= MARKET_CLOSE:
        # After close, next open is tomorrow (or Monday)
        if weekday == 4:  # Friday
            next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=3)
        else:
            next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
    else:
        # Before open today
        next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    return next_open - now_et

# =============================================================================
# BOT CONTROL
# =============================================================================

async def initialize_agent():
    """Initialize the MarketAgent"""
    try:
        from src.agents.orchestrator import MarketAgent, OrchestratorConfig
        config = OrchestratorConfig(
            initial_capital=state.capital,
            ibkr_port=int(os.getenv('IBKR_PORT', 7497))
        )
        state.agent = MarketAgent(config)
        await state.agent.initialize()
        add_log("[OK] Agent initialise")
        return True
    except Exception as e:
        add_log(f"[ERREUR] Initialisation agent: {e}")
        return False

async def start_bot():
    """Start the trading bot"""
    if state.bot_running:
        return

    state.bot_status = "Demarrage..."
    add_log("Demarrage du bot...")

    # V4.9.5: Add narrative startup thoughts
    add_thought("🤖 Bonjour! Je me réveille et prépare mon analyse...", "startup")
    add_thought("   → Vérification des connexions API (Gemini, Grok, yfinance)...", "startup")

    if state.agent is None:
        add_thought("   → Initialisation du cerveau (ReasoningEngine 4 piliers)...", "startup")
        success = await initialize_agent()
        if not success:
            add_thought("❌ Oups, je n'arrive pas à m'initialiser. Vérifiez les API keys.", "error")
            state.bot_status = "Erreur"
            return

    state.bot_running = True
    state.bot_status = "Actif"
    add_log("[OK] Bot demarre")

    # V4.9.6: AUTOMATIC DISCOVERY AT STARTUP
    add_thought("🔍 Lancement de la phase Discovery...", "discovery")
    add_thought("   → Scan X/Twitter via Grok, news via Gemini, social via StockTwits...", "discovery")

    try:
        # Run discovery to populate focus_symbols
        discovery_result = await state.agent.run_discovery_phase()
        state.last_discovery = datetime.now()
        state.last_discovery_result = discovery_result

        watchlist = discovery_result.get('watchlist', [])
        social = discovery_result.get('social_trending', [])
        grok = discovery_result.get('grok_insights', [])

        # Update focus_symbols with discovered symbols
        if watchlist:
            state.focus_symbols = watchlist[:50]
            state.agent.state_manager.update_focus_symbols(state.focus_symbols)
            add_thought(f"✅ Discovery terminée: {len(watchlist)} symboles découverts!", "discovery")
            if social:
                add_thought(f"   → Social trending: {', '.join(social[:5])}...", "social")
            if grok:
                add_thought(f"   → Grok insights: {len(grok)} tendances X/Twitter", "grok")
        else:
            # Fallback to watchlist if no discovery results
            if state.watchlist:
                state.focus_symbols = [w['symbol'] for w in state.watchlist[:50]]
                state.agent.state_manager.update_focus_symbols(state.focus_symbols)
                add_thought(f"   → Pas de nouvelles découvertes, utilisation de la watchlist ({len(state.focus_symbols)} symboles)", "discovery")
            else:
                add_thought("   → Aucun symbole découvert. Ajoutez des symboles à la watchlist.", "warning")
    except Exception as e:
        add_thought(f"⚠ Discovery error: {e}", "error")
        add_log(f"Discovery startup error: {e}")
        # Fallback to watchlist
        if state.watchlist:
            state.focus_symbols = [w['symbol'] for w in state.watchlist[:50]]
            state.agent.state_manager.update_focus_symbols(state.focus_symbols)
            add_thought(f"   → Fallback: utilisation de la watchlist ({len(state.focus_symbols)} symboles)", "discovery")

    # V4.9.5: More narrative startup
    add_thought("✅ Parfait! Je suis prêt à analyser le marché.", "startup")
    watchlist_count = len(state.watchlist) if state.watchlist else 0
    focus_count = len(state.focus_symbols) if state.focus_symbols else 0
    if focus_count > 0:
        add_thought(f"   → J'ai {focus_count} symboles en focus et {watchlist_count} en watchlist.", "startup")
        add_thought("   → Je vais commencer mon analyse. Voyons ce que le marché nous réserve!", "startup")
    else:
        add_thought("   → Hmm, je n'ai aucun symbole à analyser. Ajoutez-en à la watchlist.", "startup")

    # Start main loop in background
    state._task = asyncio.create_task(run_bot_loop())

async def stop_bot():
    """Stop the trading bot"""
    if not state.bot_running:
        return

    state.bot_status = "Arret..."
    add_log("Arret du bot...")

    if state._task:
        state._task.cancel()
        try:
            await state._task
        except asyncio.CancelledError:
            pass

    if state.agent:
        await state.agent.stop()

    state.bot_running = False
    state.bot_status = "Arrete"
    add_log("[OK] Bot arrete")

async def run_bot_loop():
    """
    Main bot loop - runs 24/7 with different modes

    V4.9.7: Simplified loop with dynamic intervals:
    - MARKET mode: Scan every 15 minutes (with trade execution)
    - Other modes: Scan every 30 minutes (analysis only)

    Note: First scan is done in start_bot() immediately after clicking "Démarrer"
    """
    add_log("[OK] Bot 24/7 demarre")

    while state.bot_running:
        try:
            # Update current market mode
            mode = get_current_market_mode()
            state.bot_mode = mode

            now_et = datetime.now(ET)
            add_log(f"Mode: {mode} | Heure ET: {now_et.strftime('%H:%M')}")

            if state.agent:
                # ══════════════════════════════════════════════════════════════
                # V4.9.7: UNIFIED SCAN LOGIC (All modes use run_trading_scan)
                # ══════════════════════════════════════════════════════════════

                # V4.9.7: Dynamic intervals - 15 min if MARKET, 30 min otherwise
                scan_interval = 900 if mode == "MARKET" else 1800  # 15 min vs 30 min

                if mode == "MARKET":
                    add_log("━━━ TRADING SCAN ━━━")
                    add_log("🟢 Marché OUVERT - Scan complet avec exécution possible")
                    add_thought("═══ PHASE TRADING - MARCHÉ OUVERT ═══", "decision")
                    add_thought("Le marché US est ouvert. Analyse complète avec possibilité d'exécution.", "decision")
                else:
                    add_log(f"━━━ ANALYSE {mode} ━━━")
                    add_log(f"🌙 Marché FERMÉ ({mode}) - Analyse sans exécution")
                    add_thought(f"═══ PHASE ANALYSE - MARCHÉ FERMÉ ({mode}) ═══", "decision")
                    add_thought("Analyse complète pour préparer l'ouverture. Pas d'exécution possible.", "decision")

                # Process pre-market queue if market just opened
                if mode == "MARKET" and state.premarket_queue:
                    add_log(f"Traitement de {len(state.premarket_queue)} alertes pre-market...")
                    add_thought(f"📋 {len(state.premarket_queue)} symboles en file d'attente pré-market à analyser", "decision")
                    for alert in state.premarket_queue:
                        alert['source'] = 'PRE_MARKET'
                        state.alerts.insert(0, alert)
                    state.premarket_queue.clear()

                # V4.9.7: Use run_trading_scan() which includes Discovery (V4.9.6)
                # Note: run_trading_scan() handles symbols synchronization internally
                result = await run_trading_scan()
                state.last_scan = datetime.now()
                state.last_trading_result = result

                # V4.9.7: Guard against None result
                if result is None:
                    result = {}

                # ══════════════════════════════════════════════════════════════
                # EXPLICATIONS DETAILLEES DU SCORING
                # ══════════════════════════════════════════════════════════════
                add_thought("═══ RESULTATS DU SCAN ═══", "decision")

                analyzed_symbols = result.get('analyzed_symbols', [])
                if analyzed_symbols:
                    add_thought(f"📊 {len(analyzed_symbols)} symboles analysés par le ReasoningEngine", "decision")

                # Expliquer chaque symbole analysé + JOURNEY TRACKING
                # NOTE: run_trading_scan() already adds narrative thoughts, we just do journey/watchlist here
                scoring_details = result.get('scoring_details', {})
                for symbol, details in list(scoring_details.items())[:10]:  # Top 10
                    score = details.get('total_score', 0)
                    technical = details.get('technical', 0)
                    fundamental = details.get('fundamental', 0)
                    sentiment = details.get('sentiment', 0)
                    news = details.get('news', 0)
                    decision = details.get('decision', 'hold')

                    # V4.9.8: Removed duplicate add_narrative_thoughts() - already done in run_trading_scan()

                    # V4.2 - Track analysis in watchlist
                    update_watchlist_status(symbol, "analyzed", score=score)
                    add_action_event("analyze", symbol, reason=f"Score: {score}/100")

                    # === V4.3: TRACK ANALYSIS JOURNEY STEP WITH DETAILED REASONING ===
                    pillar_details = details.get('pillar_details', {})
                    key_factors = details.get('key_factors', [])
                    risk_factors = details.get('risk_factors', [])
                    reasoning_summary = details.get('reasoning_summary', '')

                    tech_detail = pillar_details.get('technical', {})
                    fund_detail = pillar_details.get('fundamental', {})
                    sent_detail = pillar_details.get('sentiment', {})
                    news_detail = pillar_details.get('news', {})

                    convergence = sum(1 for p in [technical, fundamental, sentiment, news] if p >= 15)

                    reasoning_text = f"Analyse ReasoningEngine (4 piliers):\n\n"
                    tech_reasoning = tech_detail.get('reasoning', f"Score: {technical}/25")
                    reasoning_text += f"🔧 TECHNICAL ({technical}/25):\n{tech_reasoning}\n\n"
                    fund_reasoning = fund_detail.get('reasoning', f"Score: {fundamental}/25")
                    reasoning_text += f"📊 FUNDAMENTAL ({fundamental}/25):\n{fund_reasoning}\n\n"
                    sent_reasoning = sent_detail.get('reasoning', f"Score: {sentiment}/25")
                    reasoning_text += f"💬 SENTIMENT ({sentiment}/25):\n{sent_reasoning}\n\n"
                    news_reasoning = news_detail.get('reasoning', f"Score: {news}/25")
                    reasoning_text += f"📰 NEWS ({news}/25):\n{news_reasoning}\n\n"
                    reasoning_text += f"CONVERGENCE: {convergence}/4 piliers positifs (>=15/25)\n"
                    reasoning_text += f"SCORE TOTAL: {score}/100"

                    add_journey_step(
                        symbol=symbol,
                        step="analysis",
                        title=f"Score {score}/100 - {convergence}/4 piliers",
                        reasoning=reasoning_text,
                        data={
                            "total": score,
                            "technical": technical,
                            "fundamental": fundamental,
                            "sentiment": sentiment,
                            "news": news,
                            "convergence": convergence,
                            "pillar_details": pillar_details,
                            "key_factors": key_factors,
                            "risk_factors": risk_factors,
                            "reasoning_summary": reasoning_summary
                        }
                    )

                # Process alerts - V4.9.8: Alerts already added to state.alerts by run_trading_scan()
                # We just do journey/watchlist tracking here
                if result.get('alerts'):
                    add_thought(f"✅ {len(result['alerts'])} signaux dépassent le seuil de confiance", "decision")
                    for alert in result['alerts']:
                        # V4.9.8: Removed duplicate state.alerts.insert() - already done in run_trading_scan()
                        symbol = alert.get('symbol', 'N/A')
                        score = alert.get('confidence_score', 0)
                        signal = alert.get('signal', 'N/A')
                        add_thought(f"   → {symbol}: {signal} (score: {score}/100)", "decision")

                        update_watchlist_status(symbol, "signal", score=score, signal=signal)
                        add_action_event("signal", symbol, reason=f"{signal} - Score: {score}/100")

                        add_journey_step(
                            symbol=symbol,
                            step="signal",
                            title=f"Signal {signal} ({score}/100)",
                            reasoning=f"Signal {signal} généré!\n• Score {score}/100 dépasse le seuil\n• Mode: {mode}",
                            data={"signal": signal, "score": score, "confidence": alert.get('confidence', 0.5)}
                        )
                else:
                    add_thought("❌ Aucun signal ne dépasse le seuil de confiance (default: 55/100)", "decision")

                # Track rejections
                rejected = result.get('rejected_symbols', {})
                if rejected:
                    add_thought("═══ SYMBOLES REJETES ═══", "decision")
                    for symbol, reason in list(rejected.items())[:10]:
                        add_thought(f"   ❌ {symbol}: {reason}", "decision")
                        update_watchlist_status(symbol, "rejected")
                        add_action_event("reject", symbol, reason=reason[:100])
                        add_journey_step(
                            symbol=symbol,
                            step="reject",
                            title=f"Rejeté: {reason[:40]}...",
                            reasoning=f"Signal rejeté.\nRaison: {reason}",
                            data={"reason": reason}
                        )

                # Guardrails checks
                guardrails_blocked = result.get('guardrails_blocked', [])
                if guardrails_blocked:
                    add_thought("🛡️ GUARDRAILS - Trades bloqués:", "error")
                    for blocked in guardrails_blocked:
                        add_thought(f"   🚫 {blocked.get('symbol', 'N/A')}: {blocked.get('reason', 'Unknown')}", "error")

                # Summary
                signals = result.get('signals_found', 0)
                executed = result.get('trades_executed', 0)
                add_log(f"[OK] Scan: {signals} signaux, {executed} executes")
                add_thought(f"📊 Résumé: {signals} signaux trouvés, {executed} exécutés", "decision")

                # Narrative reports
                narrative_reports = result.get('narrative_reports', {})
                if narrative_reports:
                    add_thought(f"📝 {len(narrative_reports)} rapport(s) narratif(s)", "decision")
                    state.narrative_reports.update(narrative_reports)

                # Notify on high-score signals
                await notify_high_score_alerts(result.get('alerts', []))

                # ══════════════════════════════════════════════════════════════
                # V4.9.7: DYNAMIC WAIT INTERVAL
                # ══════════════════════════════════════════════════════════════
                interval_min = scan_interval // 60
                add_log(f"⏳ Prochain scan dans {interval_min} min")
                add_thought(f"En attente du prochain cycle de scan ({interval_min} minutes)", "wait")

                # Update status before waiting
                update_components_status()

                # Wait with heartbeat every 5 minutes
                heartbeats = scan_interval // 300  # Number of 5-min intervals
                for i in range(heartbeats):
                    await asyncio.sleep(300)  # 5 minutes
                    remaining = (heartbeats - i - 1) * 5
                    if remaining > 0 and state.bot_running:
                        add_log(f"💓 Heartbeat | Prochain scan dans {remaining} min")
                    if not state.bot_running:
                        break

            else:
                add_log("[ERREUR] Agent non initialise")
                await asyncio.sleep(60)

        except asyncio.CancelledError:
            break
        except Exception as e:
            add_log(f"[ERREUR] {e}")
            import traceback
            add_log(f"   Traceback: {traceback.format_exc()[:300]}")
            await asyncio.sleep(60)

    state.bot_mode = "OFF"
    add_log("Bot arrete")


# ══════════════════════════════════════════════════════════════
# END OF run_bot_loop() - V4.9.7 Simplified
# ══════════════════════════════════════════════════════════════


# OLD CODE REMOVED (PRE_MARKET and AFTER_HOURS branches)
# The unified run_trading_scan() now handles all modes with:
# - Discovery via Grok, Gemini, Social Scanner (V4.9.6)
# - 15 min interval for MARKET mode
# - 30 min interval for other modes
# This simplification was requested by user for consistent behavior


# _old_premarket_code_placeholder() - DELETED in V4.9.7
# See run_trading_scan() for unified Discovery + Scan logic


async def notify_premarket_signals(signals: List[Dict]):
    """Send Telegram notification for pre-market signals"""
    if not signals:
        return

    import httpx
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        return

    symbols = [s.get('symbol', 'N/A') for s in signals[:10]]
    message = f"🌅 <b>PRE-MARKET FOCUS</b>\n\n📋 Symboles à surveiller:\n" + "\n".join([f"• {s}" for s in symbols])

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# V4.9.7: Old PRE_MARKET and AFTER_HOURS code has been removed
# All modes now use run_trading_scan() with dynamic intervals
# ══════════════════════════════════════════════════════════════


async def notify_high_score_alerts(alerts: List[Dict]):
    """Send Telegram notification for high-score alerts"""
    if not alerts:
        return

    high_score_alerts = [a for a in alerts if a.get('confidence_score', 0) >= 70]
    if not high_score_alerts:
        return

    import httpx
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        return

    for alert in high_score_alerts:
        message = (
            f"🚀 <b>SIGNAL {alert.get('signal', 'N/A')}</b>\n\n"
            f"📈 {alert.get('symbol', 'N/A')}\n"
            f"💯 Score: {alert.get('confidence_score', 0)}/100\n"
            f"💵 Entry: ${alert.get('entry_price', 'N/A')}\n"
            f"🛑 Stop: ${alert.get('stop_loss', 'N/A')}"
        )

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
                )
        except Exception:
            pass


async def notify_premarket_signals(alerts: List[Dict]):
    """Send Telegram notification for pre-market signals"""
    if not alerts:
        return

    import httpx
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        return

    symbols = [a.get('symbol', 'N/A') for a in alerts[:5]]  # Top 5
    message = (
        f"☀️ <b>PRE-MARKET SIGNALS</b>\n\n"
        f"📋 {len(alerts)} signaux detectes:\n"
        f"{'  '.join(symbols)}\n\n"
        f"⏰ En attente de l'ouverture du marche"
    )

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            )
    except Exception:
        pass

def update_components_status():
    """Update components status from agent"""
    if state.agent:
        status = state.agent.get_status()
        state.components_status = status.get('components', {})

        guardrails = status.get('guardrails', {})
        capital_info = guardrails.get('capital', {})
        state.capital = capital_info.get('current', state.capital)
        state.daily_pnl = guardrails.get('daily', {}).get('pnl', 0)

def add_log(message: str):
    """Add a log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.logs.insert(0, f"[{timestamp}] {message}")
    state.logs = state.logs[:100]  # Keep last 100


def add_thought(content: str, category: str = "general"):
    """Add a chain-of-thought entry for reasoning display"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.chain_of_thought.insert(0, {
        "timestamp": timestamp,
        "content": content,
        "category": category
    })
    state.chain_of_thought = state.chain_of_thought[:50]  # Keep last 50


async def add_narrative_thoughts(symbol: str, analysis_data: Dict, decision: str):
    """
    V4.9.5: Generate human-like narrative thoughts for a symbol analysis.
    Shows the bot "thinking" like a human trader would.
    """
    technical = analysis_data.get('technical', 0)
    fundamental = analysis_data.get('fundamental', 0)
    sentiment = analysis_data.get('sentiment', 0)
    news = analysis_data.get('news', 0)
    total_score = analysis_data.get('total_score', 50)
    pillar_details = analysis_data.get('pillar_details', {})

    # Technical analysis narrative
    tech_reasoning = pillar_details.get('technical', {}).get('reasoning', '')
    if technical > 15:
        add_thought(f"🔍 {symbol}: Hmm, les indicateurs techniques sont encourageants...", "analysis")
        add_thought(f"   → Les EMAs montrent une tendance haussière. Voyons si le momentum confirme.", "analysis")
    elif technical < 10:
        add_thought(f"🔍 {symbol}: Les indicateurs techniques me rendent prudent...", "analysis")
        add_thought(f"   → Je vois des signaux de faiblesse. Peut-être pas le bon moment.", "analysis")
    else:
        add_thought(f"🔍 {symbol}: Techniquement, c'est neutre. J'ai besoin d'autres confirmations.", "analysis")

    # Fundamental analysis narrative
    if fundamental > 15:
        add_thought(f"📊 Les fondamentaux de {symbol} sont solides. Bonne santé financière.", "analysis")
    elif fundamental < 10:
        add_thought(f"📊 Attention, les fondamentaux de {symbol} sont préoccupants...", "analysis")

    # Sentiment narrative
    if sentiment > 15:
        add_thought(f"💬 Le sentiment social sur {symbol} est très positif! Les traders en parlent.", "social")
    elif sentiment < 10:
        add_thought(f"💬 Peu d'enthousiasme sur les réseaux pour {symbol}. Pas de buzz.", "social")

    # News narrative
    if news > 15:
        add_thought(f"📰 Des news positives pour {symbol}! Ça pourrait être un catalyseur.", "news")
    elif news < 10 and news > 0:
        add_thought(f"📰 Les news sur {symbol} sont plutôt négatives. Je reste vigilant.", "news")

    # Decision narrative
    if total_score >= 70:
        add_thought(f"💡 CONCLUSION {symbol}: Score {total_score}/100 - C'est prometteur!", "decision")
        add_thought(f"   → Tous les piliers convergent. Je considère un achat.", "decision")
    elif total_score >= 55:
        add_thought(f"💡 CONCLUSION {symbol}: Score {total_score}/100 - Intéressant mais prudence.", "decision")
        add_thought(f"   → Signal d'achat modéré. Je surveille de près.", "decision")
    else:
        add_thought(f"💡 CONCLUSION {symbol}: Score {total_score}/100 - Je passe pour l'instant.", "decision")
        add_thought(f"   → Pas assez de conviction. J'attends de meilleures conditions.", "decision")


def add_capital_event(event_type: str, amount: float, symbol: str = None, details: str = ""):
    """Track capital changes over time"""
    state.capital_history.append({
        "timestamp": datetime.now().isoformat(),
        "capital": state.capital,
        "event": event_type,  # "start", "trade_open", "trade_close", "pnl_update"
        "symbol": symbol,
        "amount": amount,
        "details": details
    })
    # Keep last 500 events
    state.capital_history = state.capital_history[-500:]


def add_action_event(action: str, symbol: str = None, price: float = None, reason: str = ""):
    """Track trading actions for timeline visualization"""
    state.actions_timeline.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,  # "discovery", "watchlist_add", "analyze", "buy", "sell", "reject"
        "symbol": symbol,
        "price": price,
        "reason": reason
    })
    # Keep last 200 actions
    state.actions_timeline = state.actions_timeline[-200:]


def structure_thoughts() -> Dict[str, Dict]:
    """
    V4.6: Structure chain of thought into organized phases with sections.

    Returns a dictionary of phases, each containing:
    - observations: What the bot is seeing
    - hypotheses: What the bot is thinking
    - insights: Key discoveries
    - decisions: Actions taken or planned
    - explore: Ideas to investigate further
    """
    if not state.chain_of_thought:
        return {}

    # Phase definitions with their categories
    phase_config = {
        'discovery': {
            'title': 'Phase Discovery',
            'emoji': '🔍',
            'color': 'blue',
            'categories': ['discovery', 'social', 'grok'],
            'observations': [],
            'hypotheses': [],
            'insights': [],
            'decisions': [],
            'explore': [],
            'timestamps': []
        },
        'analysis': {
            'title': 'Phase Analyse',
            'emoji': '📊',
            'color': 'orange',
            'categories': ['news', 'insight'],
            'observations': [],
            'hypotheses': [],
            'insights': [],
            'decisions': [],
            'explore': [],
            'timestamps': []
        },
        'trading': {
            'title': 'Phase Trading',
            'emoji': '🎯',
            'color': 'purple',
            'categories': ['decision'],
            'observations': [],
            'hypotheses': [],
            'insights': [],
            'decisions': [],
            'explore': [],
            'timestamps': []
        },
        'status': {
            'title': 'Statut',
            'emoji': '⏳',
            'color': 'slate',
            'categories': ['wait', 'error', 'general'],
            'observations': [],
            'hypotheses': [],
            'insights': [],
            'decisions': [],
            'explore': [],
            'timestamps': []
        }
    }

    # Classify thoughts into phases and sections
    for thought in state.chain_of_thought:
        content = thought.get('content', '')
        category = thought.get('category', 'general')
        timestamp = thought.get('timestamp', '')

        # Skip headers (═══)
        if content.startswith('═══'):
            continue

        # Find which phase this belongs to
        target_phase = None
        for phase_key, phase_data in phase_config.items():
            if category in phase_data['categories']:
                target_phase = phase_key
                break

        if not target_phase:
            target_phase = 'status'

        phase = phase_config[target_phase]
        phase['timestamps'].append(timestamp)

        # Classify into sections based on content patterns
        content_lower = content.lower()
        clean_content = content.strip()

        # Remove leading symbols/indentation for cleaner display
        if clean_content.startswith('   '):
            clean_content = clean_content.strip()

        # Insights: Key discoveries, scores, important findings
        if any(x in content_lower for x in ['score', 'détecté', 'trouvé', 'résultat', '✅', 'signaux', 'analysé']):
            phase['insights'].append({
                'content': clean_content,
                'timestamp': timestamp,
                'type': 'success' if '✅' in content else 'highlight'
            })
        # Hypotheses: Reasoning, analysis in progress
        elif any(x in content_lower for x in ['analyse', 'utilisation', 'lancement', 'vérifi', 'calcul']):
            phase['hypotheses'].append({
                'content': clean_content,
                'timestamp': timestamp,
                'type': 'normal'
            })
        # Explore: Suggestions, next steps
        elif any(x in content_lower for x in ['possib', 'suggér', 'explorer', 'considér', 'potentiel', 'à surveiller']):
            phase['explore'].append({
                'content': clean_content,
                'timestamp': timestamp,
                'type': 'normal'
            })
        # Decisions: Actions taken or rejected
        elif any(x in content_lower for x in ['achat', 'vente', 'rejet', 'bloqué', 'exécuté', '❌', '🛡️', 'décision']):
            phase['decisions'].append({
                'content': clean_content,
                'timestamp': timestamp,
                'type': 'error' if '❌' in content or '🛡️' in content else 'success' if 'exécuté' in content_lower else 'normal'
            })
        # Observations: Everything else (what the bot sees)
        else:
            phase['observations'].append({
                'content': clean_content,
                'timestamp': timestamp,
                'type': 'warning' if '⚠️' in content else 'normal'
            })

    # Filter out empty phases and add time ranges
    result = {}
    for phase_key, phase_data in phase_config.items():
        has_content = any([
            phase_data['observations'],
            phase_data['hypotheses'],
            phase_data['insights'],
            phase_data['decisions'],
            phase_data['explore']
        ])
        if has_content:
            # Get time range
            if phase_data['timestamps']:
                phase_data['time_range'] = f"{min(phase_data['timestamps'])} - {max(phase_data['timestamps'])}"
            else:
                phase_data['time_range'] = ''
            result[phase_key] = phase_data

    return result


def update_watchlist(symbols: List[str], source: str):
    """Update watchlist with new symbols from a source"""
    existing_symbols = {w['symbol'] for w in state.watchlist}
    timestamp = datetime.now().isoformat()

    for symbol in symbols:
        if symbol not in existing_symbols:
            state.watchlist.append({
                'symbol': symbol,
                'source': source,
                'status': 'discovered',  # discovered -> analyzing -> signal -> traded/rejected
                'added_at': timestamp,
                'score': None,
                'signal': None,
                'last_update': timestamp
            })
            add_action_event("watchlist_add", symbol, reason=f"Source: {source}")

    # Keep watchlist manageable
    state.watchlist = state.watchlist[-300:]


def update_watchlist_status(symbol: str, status: str, score: float = None, signal: str = None):
    """Update status of a symbol in watchlist"""
    for item in state.watchlist:
        if item['symbol'] == symbol:
            item['status'] = status
            item['last_update'] = datetime.now().isoformat()
            if score is not None:
                item['score'] = score
            if signal is not None:
                item['signal'] = signal
            break


def generate_report():
    """Generate a markdown report from current state"""
    try:
        from src.intelligence.report_generator import get_report_generator
        generator = get_report_generator()

        # Extract thoughts as list of strings
        thoughts = [t['content'] for t in state.chain_of_thought]

        filepath = generator.generate_daily_report(
            discovery_result=state.last_discovery_result,
            analysis_result=state.last_analysis_result,
            trading_result=state.last_trading_result,
            chain_of_thought=thoughts
        )
        add_log(f"[OK] Rapport généré: {filepath}")
        return filepath
    except Exception as e:
        add_log(f"[ERREUR] Génération rapport: {e}")
        return None

# =============================================================================
# TELEGRAM
# =============================================================================

async def send_telegram_test():
    """Send a test Telegram message"""
    import httpx

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        safe_notify("Telegram non configure", type='negative')
        return

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "Test depuis le Dashboard TradingBot V4.1",
                    "parse_mode": "HTML"
                }
            )
            if response.status_code == 200:
                safe_notify("Message envoye!", type='positive')
            else:
                safe_notify(f"Erreur: {response.text}", type='negative')
    except Exception as e:
        safe_notify(f"Erreur: {e}", type='negative')

# =============================================================================
# DISCOVERY PHASE
# =============================================================================

async def run_discovery():
    """Run discovery phase manually"""
    if state.agent is None:
        success = await initialize_agent()
        if not success:
            safe_notify("Impossible d'initialiser l'agent", type='negative')
            return

    add_log("Lancement phase Discovery...")
    safe_notify("Discovery en cours...", type='info')

    try:
        result = await state.agent.run_discovery_phase()
        add_log(f"Discovery termine: {result.get('new_symbols', 0)} nouveaux symboles")
        safe_notify(f"Discovery: {result.get('new_symbols', 0)} symboles trouves", type='positive')
    except Exception as e:
        add_log(f"[ERREUR] Discovery: {e}")
        safe_notify(f"Erreur: {e}", type='negative')

async def run_trading_scan():
    """Run a single trading scan with automatic discovery"""
    if state.agent is None:
        success = await initialize_agent()
        if not success:
            safe_notify("Impossible d'initialiser l'agent", type='negative')
            return

    add_log("Lancement scan trading...")
    safe_notify("Scan en cours...", type='info')

    # V4.9.6: Narrative thoughts - Discovery first!
    add_thought("🔍 Scan lancé! Je vais d'abord chercher les opportunités...", "analysis")

    try:
        discovered_symbols = []

        # V4.9.6: AUTOMATIC DISCOVERY - Grok (X/Twitter) + Gemini (News)
        add_thought("📡 Phase 1: Découverte des opportunités...", "analysis")

        # 1. Grok Scanner (X/Twitter)
        if state.agent.grok_scanner:
            try:
                add_thought("   🐦 Scan X/Twitter via Grok...", "analysis")
                grok_insights = await state.agent.grok_scanner.search_financial_trends()
                grok_symbols = []
                for insight in grok_insights:
                    if hasattr(insight, 'symbols') and insight.symbols:
                        grok_symbols.extend(insight.symbols)
                    if hasattr(insight, 'symbol') and insight.symbol:
                        grok_symbols.append(insight.symbol)
                discovered_symbols.extend(grok_symbols)
                if grok_symbols:
                    add_thought(f"   ✓ Grok: {len(grok_symbols)} symboles trending sur X ({', '.join(grok_symbols[:5])}...)", "analysis")
                else:
                    add_thought("   → Grok: Pas de tendance forte détectée sur X", "analysis")
            except Exception as e:
                add_thought(f"   ⚠ Grok error: {e}", "warning")
                add_log(f"Grok scan error: {e}")
        else:
            add_thought("   → Grok non configuré (ajoutez GROK_API_KEY dans .env)", "warning")

        # 2. Trend Discovery (News via Gemini)
        if state.agent.trend_discovery:
            try:
                add_thought("   📰 Analyse des news financières via Gemini...", "analysis")
                anomalies = await state.agent.trend_discovery.detect_volume_anomalies()
                volume_symbols = [s for s, _ in anomalies[:15]]
                discovered_symbols.extend(volume_symbols)
                if volume_symbols:
                    add_thought(f"   ✓ Gemini: {len(volume_symbols)} anomalies de volume ({', '.join(volume_symbols[:5])}...)", "analysis")
            except Exception as e:
                add_thought(f"   ⚠ Trend discovery error: {e}", "warning")
        else:
            add_thought("   → Trend Discovery non configuré", "warning")

        # 3. Social Scanner (StockTwits/Reddit)
        if state.agent.social_scanner:
            try:
                add_thought("   💬 Scan StockTwits/Reddit...", "analysis")
                social_result = await state.agent.social_scanner.full_scan()
                if hasattr(social_result, 'get_top_symbols'):
                    social_symbols = social_result.get_top_symbols(15)
                elif hasattr(social_result, 'trending_symbols'):
                    social_symbols = [t.symbol for t in social_result.trending_symbols[:15]]
                else:
                    social_symbols = []
                discovered_symbols.extend(social_symbols)
                if social_symbols:
                    add_thought(f"   ✓ Social: {len(social_symbols)} symboles trending ({', '.join(social_symbols[:5])}...)", "analysis")
            except Exception as e:
                add_thought(f"   ⚠ Social scan error: {e}", "warning")

        # Remove duplicates
        discovered_symbols = list(set(discovered_symbols))

        # V4.9.6: Merge discovered + watchlist
        symbols_to_scan = discovered_symbols.copy()

        # Add watchlist symbols (that aren't already discovered)
        if state.watchlist:
            watchlist_symbols = [w['symbol'] for w in state.watchlist[:50]]
            for s in watchlist_symbols:
                if s not in symbols_to_scan:
                    symbols_to_scan.append(s)

        add_thought(f"📊 Phase 2: Analyse de {len(symbols_to_scan)} symboles...", "analysis")
        if discovered_symbols:
            add_thought(f"   → {len(discovered_symbols)} découverts + {len(symbols_to_scan) - len(discovered_symbols)} de ma watchlist", "analysis")

        if not symbols_to_scan:
            add_thought("❌ Oups, je n'ai aucun symbole à analyser! Ajoutez-en à la watchlist.", "error")
            safe_notify("Ajoutez des symboles à la watchlist avant de scanner", type='warning')
            return

        add_thought("   → Mes 4 piliers: Technical + Fundamental + Sentiment + News", "analysis")

        # Sync to orchestrator
        state.agent.state_manager.update_focus_symbols(symbols_to_scan)
        add_log(f"   Synchronisation: {len(symbols_to_scan)} symboles vers orchestrator")

        result = await state.agent.run_trading_scan()
        state.last_scan = datetime.now()

        signals = result.get('signals_found', 0)
        executed = result.get('trades_executed', 0)
        analyzed = result.get('analyzed_symbols', [])
        scoring_details = result.get('scoring_details', {})

        # V4.9.5: Narrative thoughts for each analyzed symbol
        for symbol, details in list(scoring_details.items())[:5]:  # Top 5
            await add_narrative_thoughts(symbol, details, details.get('decision', 'hold'))

        if result.get('alerts'):
            for alert in result['alerts']:
                state.alerts.insert(0, {
                    'time': datetime.now().isoformat(),
                    **alert
                })

        # V4.9.5: Narrative conclusion
        if signals > 0:
            add_thought(f"✅ Scan terminé! J'ai trouvé {signals} signal(s) intéressant(s).", "decision")
            add_thought(f"   → Regardez la page 'Alertes' pour voir mes recommandations.", "decision")
        else:
            add_thought(f"🤔 Scan terminé. Aucun signal fort pour le moment.", "decision")
            add_thought(f"   → {len(analyzed)} symboles analysés, mais rien ne passe mon seuil de 55/100.", "decision")
            add_thought(f"   → Je continue de surveiller. La patience est une vertu en trading!", "decision")

        add_log(f"Scan termine: {signals} signaux, {executed} trades")
        safe_notify(f"{signals} signaux, {executed} executes", type='positive')

        # V4.9.7: Return result for run_bot_loop()
        return result

    except Exception as e:
        add_thought(f"❌ Erreur pendant le scan: {e}", "error")
        add_log(f"[ERREUR] Scan: {e}")
        safe_notify(f"Erreur: {e}", type='negative')
        return {}  # Return empty dict on error

# =============================================================================
# MAIN PAGE
# =============================================================================

@ui.page('/')
def main_page():
    """Main dashboard page"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient border-b border-slate-700/50'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('show_chart', size='lg').classes('text-green-400')
            ui.label('TradingBot V4.1').classes('text-xl font-bold text-white')
            ui.space()

            # Market mode indicator
            with ui.row().classes('items-center gap-2 mr-4'):
                mode_icon = ui.icon('access_time', size='sm')
                mode_label = ui.label('--').classes('text-sm text-white')

                def update_mode():
                    mode = get_current_market_mode()
                    mode_colors = {
                        'MARKET': ('text-green-400', '🟢 MARKET OPEN'),
                        'PRE_MARKET': ('text-yellow-400', '🌅 PRE-MARKET'),
                        'AFTER_HOURS': ('text-blue-400', '🌙 AFTER-HOURS'),
                        'WEEKEND': ('text-purple-400', '📅 WEEKEND'),
                    }
                    color, text = mode_colors.get(mode, ('text-slate-400', mode))
                    mode_label.text = text
                    mode_label.classes(color, remove='text-green-400 text-yellow-400 text-blue-400 text-purple-400 text-slate-400')

                ui.timer(10, update_mode)
                update_mode()

            # Status indicator
            with ui.row().classes('items-center gap-2'):
                status_icon = ui.icon('circle', size='sm')
                status_label = ui.label(state.bot_status).classes('text-white')

                def update_status():
                    if state.bot_running:
                        status_icon.classes('text-green-500', remove='text-red-500 text-yellow-500')
                    else:
                        status_icon.classes('text-red-500', remove='text-green-500 text-yellow-500')
                    status_label.text = state.bot_status

                ui.timer(1, update_status)

    # Sidebar
    with ui.left_drawer().classes('sidebar-gradient border-r border-slate-700/30'):
        with ui.column().classes('p-4 gap-1'):
            ui.label('Navigation').classes('text-xs uppercase tracking-wider text-slate-500 mb-3 px-3')

            nav_items = [
                ('Dashboard', 'dashboard', '/'),
                ('Watchlist', 'list_alt', '/watchlist'),
                ('Capital & PnL', 'trending_up', '/capital'),
                ('Raisonnement', 'psychology', '/reasoning'),
                ('Alertes', 'notifications', '/alerts'),
                ('Rapports', 'description', '/reports'),
                ('Configuration', 'settings', '/settings'),
                ('Logs', 'article', '/logs'),
            ]

            for name, icon, path in nav_items:
                ui.button(name, icon=icon, on_click=lambda p=path: ui.navigate.to(p)).props('flat').classes('w-full justify-start text-white nav-btn')

    # Main content
    with ui.column().classes('w-full p-6 gap-6'):

        # Control Panel
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('smart_toy', size='md').classes('text-indigo-400')
                ui.label('Contrôle du Bot').classes('text-lg font-semibold text-white')

            with ui.row().classes('gap-3 flex-wrap'):
                start_btn = ui.button('Démarrer', icon='play_arrow', on_click=start_bot).classes('modern-btn text-green-400')
                stop_btn = ui.button('Arrêter', icon='stop', on_click=stop_bot).classes('modern-btn text-red-400')
                ui.button('Discovery', icon='explore', on_click=run_discovery).classes('modern-btn text-blue-400')
                ui.button('Scan Manuel', icon='radar', on_click=run_trading_scan).classes('modern-btn text-purple-400')
                ui.button('Test Telegram', icon='send', on_click=send_telegram_test).classes('modern-btn text-cyan-400')

                def update_buttons():
                    start_btn.set_enabled(not state.bot_running)
                    stop_btn.set_enabled(state.bot_running)

                ui.timer(1, update_buttons)

        # Stats Cards - Grid responsive
        with ui.row().classes('w-full gap-4 flex-wrap'):
            # Capital
            with ui.card().classes('flex-1 min-w-[200px] stat-card p-5'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('account_balance_wallet', size='sm').classes('text-emerald-400')
                    ui.label('Capital').classes('text-xs uppercase tracking-wider text-slate-500')
                capital_label = ui.label(f'{state.capital:,.0f} EUR').classes('text-2xl font-bold gradient-text-green mt-2')

                def update_capital():
                    capital_label.text = f'{state.capital:,.0f} EUR'
                ui.timer(5, update_capital)

            # Daily P&L
            with ui.card().classes('flex-1 min-w-[200px] stat-card p-5'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('trending_up', size='sm').classes('text-blue-400')
                    ui.label('P&L Jour').classes('text-xs uppercase tracking-wider text-slate-500')
                pnl_label = ui.label(f'{state.daily_pnl:+.2f} EUR').classes('text-2xl font-bold text-green-400 mt-2')

                def update_pnl():
                    color = 'text-green-400' if state.daily_pnl >= 0 else 'text-red-400'
                    pnl_label.text = f'{state.daily_pnl:+.2f} EUR'
                    pnl_label.classes(color, remove='text-green-400 text-red-400')
                ui.timer(5, update_pnl)

            # Alerts count
            with ui.card().classes('flex-1 min-w-[200px] stat-card p-5'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('notifications_active', size='sm').classes('text-amber-400')
                    ui.label('Alertes').classes('text-xs uppercase tracking-wider text-slate-500')
                alerts_label = ui.label(str(len(state.alerts))).classes('text-2xl font-bold text-amber-400 mt-2')

                def update_alerts_count():
                    alerts_label.text = str(len(state.alerts))
                ui.timer(5, update_alerts_count)

            # Last scan
            with ui.card().classes('flex-1 min-w-[200px] stat-card p-5'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('schedule', size='sm').classes('text-indigo-400')
                    ui.label('Dernier Scan').classes('text-xs uppercase tracking-wider text-slate-500')
                scan_label = ui.label('Jamais').classes('text-xl font-bold gradient-text-blue mt-2')

                def update_scan():
                    if state.last_scan:
                        delta = datetime.now() - state.last_scan
                        if delta.seconds < 60:
                            scan_label.text = "À l'instant"
                        elif delta.seconds < 3600:
                            scan_label.text = f'Il y a {delta.seconds // 60} min'
                        else:
                            scan_label.text = state.last_scan.strftime('%H:%M')
                ui.timer(10, update_scan)

        # Pre-market info card (shown when market is closed)
        with ui.card().classes('w-full glass-card p-5') as premarket_card:
            with ui.row().classes('w-full justify-between items-center'):
                with ui.column().classes('gap-1'):
                    ui.label('Prochaine Ouverture').classes('text-sm text-slate-400')
                    time_to_open_label = ui.label('--').classes('text-lg font-bold text-yellow-400')
                    now_et_label = ui.label('--').classes('text-xs text-slate-500')

                with ui.column().classes('gap-1 text-right'):
                    ui.label('File Pre-Market').classes('text-sm text-slate-400')
                    queue_label = ui.label('0 signaux').classes('text-lg font-bold text-blue-400')

            def update_premarket_info():
                mode = get_current_market_mode()
                now_et = datetime.now(ET)
                now_et_label.text = f"Heure NY: {now_et.strftime('%H:%M')}"

                if mode == "MARKET":
                    time_to_open_label.text = "Marche OUVERT"
                    time_to_open_label.classes('text-green-400', remove='text-yellow-400')
                else:
                    time_to_open = get_time_to_market_open()
                    if time_to_open:
                        hours = int(time_to_open.total_seconds() // 3600)
                        minutes = int((time_to_open.total_seconds() % 3600) // 60)
                        time_to_open_label.text = f"Dans {hours}h{minutes:02d}m"
                    time_to_open_label.classes('text-yellow-400', remove='text-green-400')

                queue_label.text = f"{len(state.premarket_queue)} signaux"

            ui.timer(30, update_premarket_info)
            update_premarket_info()

        # Two columns
        with ui.row().classes('w-full gap-4'):
            # Components Status
            with ui.card().classes('flex-1 glass-card p-5'):
                with ui.row().classes('items-center gap-3 mb-4'):
                    ui.icon('memory', size='sm').classes('text-indigo-400')
                    ui.label('Composants').classes('text-lg font-semibold text-white')

                components = [
                    ('ReasoningEngine', 'reasoning_engine'),
                    ('TradeMemory', 'trade_memory'),
                    ('TrendDiscovery', 'trend_discovery'),
                    ('SocialScanner', 'social_scanner'),
                    ('GrokScanner', 'grok_scanner'),
                    ('IBKR Executor', 'ibkr_executor'),
                ]

                component_icons = {}
                with ui.column().classes('gap-2'):
                    for name, key in components:
                        with ui.row().classes('items-center gap-2'):
                            icon = ui.icon('check_circle', size='sm').classes('text-slate-500')
                            component_icons[key] = icon
                            ui.label(name).classes('text-sm text-white')

                def update_components():
                    for key, icon in component_icons.items():
                        active = state.components_status.get(key, False)
                        if active:
                            icon.classes('text-green-500', remove='text-slate-500 text-red-500')
                        else:
                            icon.classes('text-slate-500', remove='text-green-500 text-red-500')

                ui.timer(5, update_components)

            # Recent Alerts
            with ui.card().classes('flex-1 glass-card p-5'):
                with ui.row().classes('items-center gap-3 mb-4'):
                    ui.icon('notifications_active', size='sm').classes('text-amber-400')
                    ui.label('Alertes Recentes').classes('text-lg font-semibold text-white')

                alerts_container = ui.column().classes('gap-2 max-h-64 overflow-auto')

                def update_alerts():
                    alerts_container.clear()
                    with alerts_container:
                        if not state.alerts:
                            ui.label('Aucune alerte').classes('text-slate-400')
                        else:
                            for alert in state.alerts[:5]:
                                with ui.card().classes('w-full p-3 bg-slate-700/50 border border-slate-600/50 rounded-lg'):
                                    with ui.row().classes('justify-between items-center'):
                                        with ui.column().classes('gap-0'):
                                            ui.label(alert.get('symbol', 'N/A')).classes('font-bold text-white')
                                            signal = alert.get('signal', 'N/A')
                                            signal_color = 'text-green-400' if 'BUY' in signal else 'text-yellow-400'
                                            ui.label(signal).classes(f'text-sm {signal_color}')
                                        ui.label(f"{alert.get('confidence_score', 0)}/100").classes('text-lg font-bold text-white')

                ui.timer(10, update_alerts)
                update_alerts()

        # Logs Panel
        with ui.card().classes('w-full glass-card p-5'):
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('terminal', size='sm').classes('text-slate-400')
                ui.label('Logs').classes('text-lg font-semibold text-white')

            logs_container = ui.column().classes('gap-2 max-h-64 overflow-auto font-mono text-base')

            def update_logs():
                logs_container.clear()
                with logs_container:
                    for log in state.logs[:15]:
                        if '[OK]' in log:
                            color = 'text-green-400'
                        elif '[ERREUR]' in log:
                            color = 'text-red-400'
                        else:
                            color = 'text-slate-300'
                        ui.label(log).classes(color)

            ui.timer(2, update_logs)
            update_logs()


@ui.page('/alerts')
def alerts_page():
    """Alerts page"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('notifications', size='lg').classes('text-yellow-400')
            ui.label('Alertes').classes('text-xl font-bold text-white')
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6'):
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('notifications_active', size='sm').classes('text-amber-400')
                ui.label('Toutes les Alertes').classes('text-lg font-semibold text-white')

            if not state.alerts:
                ui.label('Aucune alerte').classes('text-slate-400')
            else:
                columns = [
                    {'name': 'time', 'label': 'Heure', 'field': 'time', 'sortable': True},
                    {'name': 'symbol', 'label': 'Symbole', 'field': 'symbol', 'sortable': True},
                    {'name': 'signal', 'label': 'Signal', 'field': 'signal'},
                    {'name': 'score', 'label': 'Score', 'field': 'score', 'sortable': True},
                ]

                rows = []
                for alert in state.alerts:
                    rows.append({
                        'time': alert.get('time', '')[:19] if alert.get('time') else '',
                        'symbol': alert.get('symbol', 'N/A'),
                        'signal': alert.get('signal', 'N/A'),
                        'score': alert.get('confidence_score', 0),
                    })

                ui.table(columns=columns, rows=rows).classes('w-full')


@ui.page('/settings')
def settings_page():
    """Settings page"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('settings', size='lg').classes('text-blue-400')
            ui.label('Configuration').classes('text-xl font-bold text-white')
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6 gap-4'):
        # Capital
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('Capital').classes('text-lg font-bold text-white')
            capital_input = ui.number('Capital initial (EUR)', value=state.capital, min=100, max=1000000).classes('w-64')

            def save_capital():
                state.capital = capital_input.value
                ui.notify('Capital mis a jour', type='positive')

            ui.button('Sauvegarder', on_click=save_capital).classes('mt-2')

        # Telegram
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('Telegram').classes('text-lg font-bold text-white mb-2')

            token = os.getenv('TELEGRAM_BOT_TOKEN', '')
            chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

            with ui.row().classes('gap-4 items-center'):
                ui.icon('check_circle' if token else 'cancel').classes('text-green-500' if token else 'text-red-500')
                ui.label(f"Bot Token: {'Configure' if token else 'Non configure'}").classes('text-white')

            with ui.row().classes('gap-4 items-center'):
                ui.icon('check_circle' if chat_id else 'cancel').classes('text-green-500' if chat_id else 'text-red-500')
                ui.label(f"Chat ID: {chat_id if chat_id else 'Non configure'}").classes('text-white')

            ui.button('Tester Telegram', on_click=send_telegram_test).classes('mt-2')

        # IBKR
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('Interactive Brokers').classes('text-lg font-bold text-white mb-2')

            host = os.getenv('IBKR_HOST', '127.0.0.1')
            port = os.getenv('IBKR_PORT', '7497')

            ui.label(f"Host: {host}").classes('text-white')
            ui.label(f"Port: {port} ({'Paper Trading' if port == '7497' else 'Live Trading'})").classes('text-white')

            async def test_ibkr():
                try:
                    from ib_insync import IB
                    ib = IB()
                    await ib.connectAsync(host, int(port), clientId=99)
                    accounts = ib.managedAccounts()
                    ib.disconnect()
                    safe_notify(f"Connecte! Compte: {accounts[0]}", type='positive')
                except Exception as e:
                    safe_notify(f"Erreur: {e}", type='negative')

            ui.button('Tester Connexion IBKR', on_click=test_ibkr).classes('mt-2')


@ui.page('/logs')
def logs_page():
    """Logs page"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('article', size='lg').classes('text-slate-400')
            ui.label('Logs').classes('text-xl font-bold text-white')
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6'):
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('justify-between items-center mb-4'):
                ui.label('Logs Complets').classes('text-lg font-bold text-white')
                ui.button('Effacer', icon='delete', on_click=lambda: state.logs.clear()).props('flat').classes('text-red-400')

            logs_container = ui.column().classes('gap-2 font-mono text-base max-h-[500px] overflow-auto')

            def update_logs():
                logs_container.clear()
                with logs_container:
                    for log in state.logs:
                        if '[OK]' in log:
                            color = 'text-green-400'
                        elif '[ERREUR]' in log:
                            color = 'text-red-400'
                        else:
                            color = 'text-slate-300'
                        ui.label(log).classes(color)

            ui.timer(2, update_logs)
            update_logs()


@ui.page('/reasoning')
def reasoning_page():
    """Chain of Thought / Reasoning page"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('psychology', size='lg').classes('text-purple-400')
            ui.label('Raisonnement (Chain of Thought)').classes('text-xl font-bold text-white')
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6 gap-4'):
        # Explanation card
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('🧠 Comment le bot pense').classes('text-lg font-bold text-white mb-2')
            ui.label(
                "Cette page affiche le flux de raisonnement du bot en temps réel. "
                "Chaque entrée représente une étape du processus de décision: "
                "analyse sociale, insights Grok, analyse des news, construction de la watchlist..."
            ).classes('text-slate-400 text-sm')

        # Chain of Thought cards by category
        categories = {
            'discovery': ('🔍', 'Discovery', 'text-blue-400'),
            'social': ('🌐', 'Social Media', 'text-green-400'),
            'grok': ('🐦', 'Grok/X', 'text-cyan-400'),
            'news': ('📰', 'News/LLM', 'text-yellow-400'),
            'insight': ('💡', 'Insights', 'text-orange-400'),
            'decision': ('🎯', 'Décisions', 'text-purple-400'),
            'error': ('❌', 'Erreurs', 'text-red-400'),
            'wait': ('⏳', 'Attente', 'text-slate-400'),
            'general': ('💭', 'Général', 'text-slate-300'),
        }

        # V4.6: Redesigned as Structured Thoughts
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('justify-between items-center mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('🧠 Flux de Pensées').classes('text-xl font-bold text-white')
                    create_help_button('Flux de Pensées', get_timeline_explanation())
                with ui.row().classes('gap-2'):
                    # Phase indicators
                    phase_indicators = [
                        ('🔍', 'Discovery', 'text-blue-400'),
                        ('📊', 'Analyse', 'text-orange-400'),
                        ('🎯', 'Trading', 'text-purple-400'),
                        ('⏳', 'Statut', 'text-slate-400'),
                    ]
                    for emoji, label, color in phase_indicators:
                        with ui.element('div').classes('flex items-center gap-1'):
                            ui.label(emoji).classes('text-sm')
                            ui.label(label).classes(f'text-xs {color} hidden md:inline')
                    ui.button(icon='delete', on_click=lambda: state.chain_of_thought.clear()).props('flat round size=sm').classes('text-red-400')

            # Structured thoughts container
            thoughts_container = ui.element('div').classes('max-h-[700px] overflow-auto pr-4')

            def render_bullet_item(item: dict):
                """Render a single bullet point item"""
                item_type = item.get('type', 'normal')
                content = item.get('content', '')
                classes = 'thought-bullet-item'
                if item_type == 'success':
                    classes += ' success'
                elif item_type == 'warning':
                    classes += ' warning'
                elif item_type == 'error':
                    classes += ' error'
                elif item_type == 'highlight':
                    classes += ' highlight'
                with ui.element('div').classes(classes):
                    ui.label(content).classes('text-sm')

            def render_section(title: str, icon: str, items: list, section_type: str = 'normal'):
                """Render a section with bullet points"""
                if not items:
                    return

                with ui.element('div').classes('thought-section'):
                    with ui.element('div').classes('thought-section-title'):
                        ui.label(icon).classes('text-sm')
                        ui.label(title)

                    for item in items[:8]:  # Limit to 8 items per section
                        render_bullet_item(item)

            def update_thoughts():
                thoughts_container.clear()
                with thoughts_container:
                    if not state.chain_of_thought:
                        # Empty state
                        with ui.element('div').classes('thought-empty-state'):
                            ui.label('🤖').classes('thought-empty-icon')
                            ui.label('En attente des premières pensées...').classes('text-slate-400 italic text-lg mb-2')
                            ui.label('Démarrez le bot pour voir le flux de raisonnement structuré.').classes('text-slate-500 text-sm')
                    else:
                        # Get structured thoughts
                        structured = structure_thoughts()

                        if not structured:
                            # Fallback if structuring fails
                            ui.label('Chargement des pensées...').classes('text-slate-400')
                            return

                        # Phase colors and icons
                        phase_styles = {
                            'discovery': {'icon': '🔍', 'color': 'blue'},
                            'analysis': {'icon': '📊', 'color': 'orange'},
                            'trading': {'icon': '🎯', 'color': 'purple'},
                            'status': {'icon': '⏳', 'color': 'slate'}
                        }

                        # Render each phase as a card
                        for phase_key, phase_data in structured.items():
                            style = phase_styles.get(phase_key, {'icon': '💭', 'color': 'slate'})

                            with ui.element('div').classes('thought-phase-card'):
                                # Phase header
                                with ui.element('div').classes('thought-phase-header'):
                                    with ui.element('div').classes(f'thought-phase-icon {phase_key}'):
                                        ui.label(style['icon'])
                                    with ui.column().classes('gap-0'):
                                        ui.label(phase_data['title']).classes('text-white font-bold text-lg')
                                        if phase_data.get('time_range'):
                                            ui.label(phase_data['time_range']).classes('thought-timestamp')

                                # Observations section (Ce que le bot observe)
                                if phase_data['observations']:
                                    render_section('Observations', '👁️', phase_data['observations'])

                                # Hypotheses section (Ce que le bot pense)
                                if phase_data['hypotheses']:
                                    render_section('Raisonnement', '🧩', phase_data['hypotheses'])

                                # Key Insights (highlighted)
                                if phase_data['insights']:
                                    with ui.element('div').classes('thought-key-insight'):
                                        with ui.element('div').classes('thought-key-insight-title'):
                                            ui.label('💡')
                                            ui.label('Découvertes Clés')
                                        for item in phase_data['insights'][:5]:
                                            render_bullet_item(item)

                                # Decisions section
                                if phase_data['decisions']:
                                    render_section('Décisions', '✅', phase_data['decisions'])

                                # Ideas to explore
                                if phase_data['explore']:
                                    with ui.element('div').classes('thought-explore-box'):
                                        with ui.element('div').classes('thought-explore-title'):
                                            ui.label('🔮')
                                            ui.label('À Explorer')
                                        for item in phase_data['explore'][:3]:
                                            with ui.element('div').classes('text-sm text-cyan-200 py-1'):
                                                ui.label(f"→ {item.get('content', '')}")

            ui.timer(3, update_thoughts)
            update_thoughts()

        # Discovery Result Summary
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('items-center gap-2 mb-4'):
                ui.label('📊 Derniers Résultats Discovery').classes('text-lg font-bold text-white')
                create_help_button('Processus de Découverte', get_discovery_explanation())

            discovery_container = ui.column().classes('gap-2')

            def update_discovery_summary():
                discovery_container.clear()
                with discovery_container:
                    if state.last_discovery_result:
                        result = state.last_discovery_result
                        watchlist = result.get('watchlist', [])
                        social = result.get('social_trending', [])
                        grok = result.get('grok_insights', [])

                        with ui.row().classes('gap-4 flex-wrap'):
                            with ui.card().classes('p-3 bg-slate-700'):
                                ui.label('Watchlist').classes('text-xs text-slate-400')
                                ui.label(str(len(watchlist))).classes('text-xl font-bold text-blue-400')

                            with ui.card().classes('p-3 bg-slate-700'):
                                ui.label('Social Trending').classes('text-xs text-slate-400')
                                ui.label(str(len(social))).classes('text-xl font-bold text-green-400')

                            with ui.card().classes('p-3 bg-slate-700'):
                                ui.label('Grok Insights').classes('text-xs text-slate-400')
                                ui.label(str(len(grok))).classes('text-xl font-bold text-cyan-400')

                        if watchlist:
                            ui.label('Top 10 Watchlist:').classes('text-sm text-slate-300 mt-2')
                            ui.label(', '.join(watchlist[:10])).classes('text-sm text-white font-mono')
                    else:
                        ui.label('Pas encore de résultats. Démarrez le bot ou lancez un Discovery manuel.').classes('text-slate-400 italic')

            ui.timer(10, update_discovery_summary)
            update_discovery_summary()

        # V4.2 - Narrative Reports Section
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('📝 Rapports Narratifs (Analyse Étayée)').classes('text-lg font-bold text-white mb-4')
            ui.label(
                "Ces rapports expliquent POURQUOI un actif a été sélectionné, avec les thèmes, "
                "catalyseurs, et justifications pour chaque décision."
            ).classes('text-slate-400 text-sm mb-4')

            narrative_container = ui.column().classes('gap-4 max-h-[600px] overflow-auto')

            def update_narrative_reports():
                narrative_container.clear()
                with narrative_container:
                    if not state.narrative_reports:
                        ui.label('Aucun rapport narratif. Les rapports sont générés pendant les scans de trading pour les signaux actionables.').classes('text-slate-400 italic')
                    else:
                        # Display each narrative report
                        for symbol, report in state.narrative_reports.items():
                            with ui.expansion(f'📈 {symbol}', icon='analytics').classes('w-full bg-slate-700'):
                                ui.markdown(report).classes('text-white p-4')

            ui.button('Rafraîchir', icon='refresh', on_click=update_narrative_reports).props('flat').classes('text-blue-400 mb-2')
            update_narrative_reports()
            ui.timer(15, update_narrative_reports)

        # ══════════════════════════════════════════════════════════════
        # V4.8: ARBRE DÉCISIONNEL - Flux de pensée par symbole
        # ══════════════════════════════════════════════════════════════
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('account_tree', size='sm').classes('text-purple-400')
                ui.label('🌳 Arbre Décisionnel - Flux de Pensée').classes('text-lg font-bold text-white')

            ui.label(
                "Visualisation du parcours réel de chaque symbole: Discovery → Watchlist → Analyse → Décision → Position"
            ).classes('text-slate-400 text-sm mb-4')

            decision_tree_container = ui.element('div').classes('w-full')

            # V4.8: Persistent dialog for pillar details on /reasoning page
            with ui.dialog() as reasoning_pillar_dialog:
                with ui.card().classes('glass-card w-[600px] max-h-[80vh] overflow-auto p-4'):
                    reasoning_dialog_title = ui.label('').classes('text-white text-xl font-bold mb-4')
                    reasoning_dialog_total = ui.label('').classes('text-2xl font-bold')
                    ui.separator().classes('my-4')
                    reasoning_pillars_grid = ui.element('div').classes('grid grid-cols-2 gap-4')
                    ui.button('Fermer', on_click=reasoning_pillar_dialog.close).classes('mt-4 w-full')

            def show_reasoning_pillar_details(symbol: str, analysis_data: Dict, total_score: float):
                """Show detailed pillar breakdown for /reasoning page"""
                reasoning_dialog_title.text = f'📊 Analyse 4 Piliers - {symbol}'

                score_color = 'text-green-400' if total_score >= 55 else 'text-yellow-400' if total_score >= 40 else 'text-red-400'
                reasoning_dialog_total.text = f'Score Total: {total_score}/100'
                reasoning_dialog_total.classes(score_color, remove='text-green-400 text-yellow-400 text-red-400')

                reasoning_pillars_grid.clear()
                with reasoning_pillars_grid:
                    pillar_details = analysis_data.get('pillar_details', {})

                    pillars = [
                        ('🔧 Technical', analysis_data.get('technical', 0), pillar_details.get('technical', {})),
                        ('📊 Fundamental', analysis_data.get('fundamental', 0), pillar_details.get('fundamental', {})),
                        ('💬 Sentiment', analysis_data.get('sentiment', 0), pillar_details.get('sentiment', {})),
                        ('📰 News', analysis_data.get('news', 0), pillar_details.get('news', {})),
                    ]

                    for name, raw_score, detail in pillars:
                        # Convert to display format
                        display_score = convert_pillar_score_to_display(raw_score)
                        is_pass = display_score >= 15
                        bg_class = 'bg-green-900/50' if is_pass else 'bg-red-900/50'
                        reasoning = detail.get('reasoning', 'Pas d\'analyse disponible')
                        signal = detail.get('signal', 'neutral')

                        with ui.card().classes(f'{bg_class} p-3'):
                            with ui.row().classes('justify-between items-center mb-2'):
                                ui.label(name).classes('text-white font-bold')
                                score_clr = 'text-green-400' if is_pass else 'text-red-400'
                                ui.label(f'{display_score}/25').classes(f'{score_clr} font-bold')

                            ui.label(signal.upper()).classes('text-xs text-slate-400 mb-2')
                            ui.label(reasoning[:150] + '...' if len(reasoning) > 150 else reasoning).classes('text-slate-300 text-xs')

                            # Show factors if available
                            factors = detail.get('factors', [])
                            if factors:
                                with ui.expansion('Facteurs', icon='list').classes('mt-2'):
                                    for f in factors[:3]:
                                        f_text = f.get('name', str(f)) if isinstance(f, dict) else str(f)
                                        ui.label(f'• {f_text}').classes('text-slate-400 text-xs')

                reasoning_pillar_dialog.open()

            def render_decision_tree():
                """Render the actual decision flow for recent symbols"""
                decision_tree_container.clear()
                with decision_tree_container:
                    journeys = list(state.symbol_journeys.items())

                    if not journeys:
                        with ui.element('div').classes('text-center py-12'):
                            ui.label('🤖').classes('text-6xl mb-4')
                            ui.label('En attente des premières analyses...').classes('text-slate-400 text-lg')
                            ui.label('Lancez le bot pour voir le flux de décision.').classes('text-slate-500 text-sm')
                        return

                    # CSS for decision tree
                    ui.add_head_html('''
                    <style>
                        .decision-flow {
                            display: flex;
                            flex-direction: column;
                            gap: 8px;
                            max-height: 600px;
                            overflow-y: auto;
                        }
                        .symbol-journey {
                            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
                            border: 1px solid rgba(148, 163, 184, 0.2);
                            border-radius: 12px;
                            padding: 16px;
                            transition: all 0.3s ease;
                        }
                        .symbol-journey:hover {
                            border-color: rgba(139, 92, 246, 0.5);
                            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.2);
                        }
                        .journey-header {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 12px;
                        }
                        .journey-symbol {
                            font-size: 18px;
                            font-weight: bold;
                            color: white;
                        }
                        .journey-status {
                            padding: 4px 12px;
                            border-radius: 20px;
                            font-size: 12px;
                            font-weight: 600;
                        }
                        .status-discovered { background: #3b82f6; color: white; }
                        .status-analyzing { background: #f59e0b; color: black; }
                        .status-signal { background: #22c55e; color: white; }
                        .status-bought { background: #8b5cf6; color: white; }
                        .status-rejected { background: #ef4444; color: white; }
                        .journey-flow {
                            display: flex;
                            align-items: center;
                            gap: 8px;
                            flex-wrap: wrap;
                        }
                        .flow-step {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            padding: 8px 12px;
                            background: rgba(71, 85, 105, 0.5);
                            border-radius: 8px;
                            min-width: 80px;
                        }
                        .flow-step.active {
                            background: linear-gradient(135deg, #8b5cf6, #6366f1);
                            box-shadow: 0 2px 10px rgba(139, 92, 246, 0.4);
                        }
                        .flow-step.passed {
                            background: rgba(34, 197, 94, 0.3);
                            border: 1px solid #22c55e;
                        }
                        .flow-step.failed {
                            background: rgba(239, 68, 68, 0.3);
                            border: 1px solid #ef4444;
                        }
                        .flow-arrow {
                            color: #6366f1;
                            font-size: 20px;
                        }
                        .flow-icon { font-size: 16px; }
                        .flow-label { font-size: 10px; color: #94a3b8; margin-top: 2px; }
                        .flow-value { font-size: 11px; color: white; font-weight: bold; }
                        .journey-reasoning {
                            margin-top: 12px;
                            padding: 10px;
                            background: rgba(0, 0, 0, 0.3);
                            border-radius: 8px;
                            border-left: 3px solid #8b5cf6;
                        }
                        .reasoning-title {
                            font-size: 11px;
                            color: #a78bfa;
                            font-weight: 600;
                            margin-bottom: 4px;
                        }
                        .reasoning-text {
                            font-size: 12px;
                            color: #cbd5e1;
                            line-height: 1.4;
                        }
                    </style>
                    ''')

                    with ui.element('div').classes('decision-flow'):
                        # Sort by most recent
                        sorted_journeys = sorted(
                            journeys,
                            key=lambda x: x[1].get('last_update', ''),
                            reverse=True
                        )[:10]  # Top 10 most recent

                        for symbol, journey in sorted_journeys:
                            status = journey.get('current_status', 'discovered')
                            steps = journey.get('journey', [])

                            # V4.8: Calculate score from pillar sums (consistent with /alerts page)
                            analysis_step = next((s for s in steps if s.get('step') == 'analysis'), None)
                            analysis_data_tmp = analysis_step.get('data', {}) if analysis_step else {}
                            t_tmp = convert_pillar_score_to_display(analysis_data_tmp.get('technical', 0))
                            f_tmp = convert_pillar_score_to_display(analysis_data_tmp.get('fundamental', 0))
                            s_tmp = convert_pillar_score_to_display(analysis_data_tmp.get('sentiment', 0))
                            n_tmp = convert_pillar_score_to_display(analysis_data_tmp.get('news', 0))
                            score = round(t_tmp + f_tmp + s_tmp + n_tmp, 1)

                            # Determine status class
                            status_class = {
                                'discovered': 'status-discovered',
                                'analyzing': 'status-analyzing',
                                'analyzed': 'status-analyzing',
                                'signal': 'status-signal',
                                'bought': 'status-bought',
                                'rejected': 'status-rejected'
                            }.get(status, 'status-discovered')

                            with ui.element('div').classes('symbol-journey'):
                                # Header
                                with ui.element('div').classes('journey-header'):
                                    ui.label(symbol).classes('journey-symbol')
                                    with ui.row().classes('items-center gap-2'):
                                        if score > 0:
                                            score_color = 'text-green-400' if score >= 55 else 'text-yellow-400' if score >= 40 else 'text-red-400'
                                            ui.label(f'{score}/100').classes(f'text-lg font-bold {score_color}')
                                        ui.element('span').classes(f'journey-status {status_class}').text = status.upper()

                                # Flow visualization
                                with ui.element('div').classes('journey-flow'):
                                    # Step 1: Discovery
                                    discovery_step = next((s for s in steps if s.get('step') == 'discovery'), None)
                                    step_class = 'passed' if discovery_step else ''
                                    with ui.element('div').classes(f'flow-step {step_class}'):
                                        ui.label('🔍').classes('flow-icon')
                                        ui.label('Discovery').classes('flow-label')
                                        if discovery_step:
                                            sources = discovery_step.get('data', {}).get('sources', [])
                                            ui.label(f'{len(sources)} src').classes('flow-value')

                                    ui.label('→').classes('flow-arrow')

                                    # Step 2: Watchlist
                                    watchlist_step = status in ['analyzing', 'analyzed', 'signal', 'bought']
                                    step_class = 'passed' if watchlist_step else ''
                                    with ui.element('div').classes(f'flow-step {step_class}'):
                                        ui.label('📋').classes('flow-icon')
                                        ui.label('Watchlist').classes('flow-label')

                                    ui.label('→').classes('flow-arrow')

                                    # Step 3: Analysis (4 pillars) - V4.8: CLICKABLE
                                    analysis_step = next((s for s in steps if s.get('step') == 'analysis'), None)
                                    step_class = 'active' if status == 'analyzing' else 'passed' if analysis_step else ''
                                    analysis_data = analysis_step.get('data', {}) if analysis_step else {}

                                    # Make clickable if has analysis
                                    click_class = 'cursor-pointer hover:scale-105 transition-transform' if analysis_step else ''
                                    with ui.element('div').classes(f'flow-step {step_class} {click_class}').on(
                                        'click',
                                        lambda sym=symbol, data=analysis_data, sc=score: show_reasoning_pillar_details(sym, data, sc) if data else None
                                    ):
                                        ui.label('📊').classes('flow-icon')
                                        ui.label('4 Piliers').classes('flow-label')
                                        if analysis_step:
                                            # V4.8: Convert scores to display format [0-25]
                                            t = convert_pillar_score_to_display(analysis_data.get('technical', 0))
                                            f = convert_pillar_score_to_display(analysis_data.get('fundamental', 0))
                                            s = convert_pillar_score_to_display(analysis_data.get('sentiment', 0))
                                            n = convert_pillar_score_to_display(analysis_data.get('news', 0))
                                            ui.label(f'T{t:.0f} F{f:.0f} S{s:.0f} N{n:.0f}').classes('flow-value')
                                            ui.label('(clic pour détails)').classes('text-xs text-slate-500')

                                    ui.label('→').classes('flow-arrow')

                                    # Step 4: Decision
                                    if score >= 55:
                                        step_class = 'passed'
                                        decision_icon = '✅'
                                        decision_label = 'BUY'
                                    elif score >= 40:
                                        step_class = ''
                                        decision_icon = '👀'
                                        decision_label = 'WATCH'
                                    elif score > 0:
                                        step_class = 'failed'
                                        decision_icon = '❌'
                                        decision_label = 'REJECT'
                                    else:
                                        step_class = ''
                                        decision_icon = '⏳'
                                        decision_label = 'PENDING'

                                    with ui.element('div').classes(f'flow-step {step_class}'):
                                        ui.label(decision_icon).classes('flow-icon')
                                        ui.label(decision_label).classes('flow-label')
                                        if score > 0:
                                            ui.label(f'{score}/100').classes('flow-value')

                                    # Step 5: Position (if bought)
                                    if status == 'bought':
                                        ui.label('→').classes('flow-arrow')
                                        with ui.element('div').classes('flow-step active'):
                                            ui.label('💰').classes('flow-icon')
                                            ui.label('Position').classes('flow-label')
                                            ui.label('OPEN').classes('flow-value')

                                # Last reasoning
                                if steps:
                                    last_step = steps[-1]
                                    reasoning = last_step.get('reasoning', '')
                                    if reasoning:
                                        with ui.element('div').classes('journey-reasoning'):
                                            ui.label('💭 Dernière réflexion:').classes('reasoning-title')
                                            # Truncate long reasoning
                                            short_reasoning = reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
                                            ui.label(short_reasoning).classes('reasoning-text')

            # V4.8: DISABLED auto-refresh - causes view reset when user is interacting
            # ui.timer(10, render_decision_tree)  # DISABLED
            render_decision_tree()

        # ══════════════════════════════════════════════════════════════
        # V4.8: SYSTÈME D'APPRENTISSAGE - Explication claire
        # ══════════════════════════════════════════════════════════════
        with ui.card().classes('w-full glass-card p-6'):
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('school', size='sm').classes('text-cyan-400')
                ui.label('🎓 Comment le Bot Apprend').classes('text-lg font-bold text-white')

            learning_container = ui.column().classes('gap-4')

            def render_learning_stats():
                """Render learning statistics"""
                learning_container.clear()
                with learning_container:
                    try:
                        from src.agents.reasoning_graph import get_graph_store

                        graph_store = get_graph_store()
                        vectors_with_outcomes = graph_store.get_all_vectors_with_outcomes()

                        total_graphs = len(vectors_with_outcomes)
                        profits = [v for v in vectors_with_outcomes if v[1] == 'profit']
                        losses = [v for v in vectors_with_outcomes if v[1] == 'loss']
                        total_closed = len(profits) + len(losses)

                        # Explanation of the learning process
                        with ui.card().classes('w-full p-4 bg-slate-800/50 rounded-lg'):
                            ui.label('📖 Le Cycle d\'Apprentissage').classes('text-sm font-bold text-white mb-3')

                            # Step-by-step process
                            steps = [
                                ('1️⃣', 'Analyse', 'Chaque analyse génère un vecteur 80D (sources + piliers + décision)', total_graphs > 0),
                                ('2️⃣', 'Trade', 'Si score >= 55, position ouverte avec entry/stop', any(j.get('current_status') == 'bought' for _, j in state.symbol_journeys.items()) if state.symbol_journeys else False),
                                ('3️⃣', 'Clôture', 'Trade fermé → Résultat (profit/loss) enregistré', total_closed > 0),
                                ('4️⃣', 'Feedback', 'Le vecteur est marqué avec l\'outcome réel', total_closed > 0),
                                ('5️⃣', 'Optimisation', 'Patterns gagnants identifiés → poids ajustés', total_closed >= 5),
                            ]

                            for icon, name, desc, completed in steps:
                                color = 'text-green-400' if completed else 'text-slate-500'
                                check = '✓' if completed else '○'
                                with ui.row().classes('items-start gap-2 mb-2'):
                                    ui.label(f'{check}').classes(f'{color} text-sm w-4')
                                    with ui.column().classes('gap-0'):
                                        ui.label(f'{icon} {name}').classes(f'text-sm font-semibold {"text-white" if completed else "text-slate-400"}')
                                        ui.label(desc).classes('text-xs text-slate-500')

                        # Current stats
                        with ui.row().classes('gap-4 flex-wrap mt-4'):
                            # Analyses
                            with ui.card().classes('p-4 bg-slate-700/50 rounded-lg flex-1 min-w-[120px]'):
                                ui.label('📊 Analyses').classes('text-xs text-slate-400')
                                ui.label(str(len(state.symbol_journeys))).classes('text-2xl font-bold text-blue-400')
                                ui.label('symboles traités').classes('text-xs text-slate-500')

                            # Vecteurs
                            with ui.card().classes('p-4 bg-slate-700/50 rounded-lg flex-1 min-w-[120px]'):
                                ui.label('🧮 Vecteurs').classes('text-xs text-slate-400')
                                ui.label(str(total_graphs)).classes('text-2xl font-bold text-purple-400')
                                ui.label('en base').classes('text-xs text-slate-500')

                            # Trades clôturés
                            with ui.card().classes('p-4 bg-slate-700/50 rounded-lg flex-1 min-w-[120px]'):
                                ui.label('📈 Clôturés').classes('text-xs text-slate-400')
                                ui.label(str(total_closed)).classes('text-2xl font-bold text-cyan-400')
                                ui.label(f'({len(profits)}W / {len(losses)}L)').classes('text-xs text-slate-500')

                            # Win rate
                            with ui.card().classes('p-4 bg-slate-700/50 rounded-lg flex-1 min-w-[120px]'):
                                ui.label('🏆 Win Rate').classes('text-xs text-slate-400')
                                if total_closed > 0:
                                    win_rate = len(profits) / total_closed * 100
                                    color = 'text-green-400' if win_rate > 50 else 'text-red-400'
                                    ui.label(f'{win_rate:.0f}%').classes(f'text-2xl font-bold {color}')
                                else:
                                    ui.label('—').classes('text-2xl font-bold text-slate-500')
                                ui.label('historique').classes('text-xs text-slate-500')

                        # Current learning status
                        with ui.card().classes('w-full p-4 bg-gradient-to-r from-slate-800/50 to-slate-700/50 rounded-lg mt-4'):
                            ui.label('📡 Statut Actuel').classes('text-sm font-bold text-white mb-2')

                            if total_closed == 0:
                                ui.label('⏳ En attente du premier trade clôturé pour commencer l\'apprentissage').classes('text-yellow-400 text-sm')
                                ui.label('Le bot analyse et trade, mais sans historique de résultats, il ne peut pas encore apprendre.').classes('text-slate-400 text-xs mt-1')
                            elif total_closed < 5:
                                ui.label(f'📥 Collecte de données: {total_closed}/5 trades clôturés').classes('text-orange-400 text-sm')
                                ui.label('Le bot a besoin d\'au moins 5 trades clôturés pour identifier des patterns.').classes('text-slate-400 text-xs mt-1')
                            else:
                                ui.label('✅ Apprentissage actif').classes('text-green-400 text-sm')
                                ui.label(f'Le bot ajuste ses poids basé sur {total_closed} trades historiques.').classes('text-slate-400 text-xs mt-1')

                                # Show learned insights
                                if len(profits) > len(losses):
                                    ui.label('💡 Insight: La stratégie actuelle est profitable').classes('text-green-400 text-xs mt-2')
                                else:
                                    ui.label('⚠️ Insight: La stratégie nécessite un ajustement').classes('text-red-400 text-xs mt-2')

                    except Exception as e:
                        ui.label(f'Erreur: {e}').classes('text-red-400')

            ui.timer(30, render_learning_stats)
            render_learning_stats()


@ui.page('/watchlist')
def watchlist_page():
    """Watchlist page - Decision Tree visualization with detailed factor breakdown"""

    ui.dark_mode(True)
    apply_modern_styles()

    # CSS for decision tree and factor cards (V4.6 Enhanced)
    ui.add_head_html('''
    <style>
        /* V4.6 Enhanced Decision Tree Styles */
        .decision-tree {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0;
            padding: 20px;
            position: relative;
        }

        /* Phase Container */
        .tree-phase {
            width: 100%;
            position: relative;
            padding: 16px 0;
        }

        .tree-phase-label {
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.2));
            border-left: 3px solid #8b5cf6;
            padding: 8px 12px;
            border-radius: 0 8px 8px 0;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #a78bfa;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: translateY(-50%) rotate(180deg);
        }

        .tree-phase-content {
            margin-left: 50px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .tree-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }

        .tree-node-box {
            padding: 14px 24px;
            border-radius: 12px;
            text-align: center;
            min-width: 150px;
            position: relative;
            z-index: 1;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .tree-node-box:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 30px rgba(0, 0, 0, 0.4);
        }

        .tree-node-box.root {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            border: 2px solid #60a5fa;
        }
        .tree-node-box.check {
            background: linear-gradient(135deg, #475569, #334155);
            border: 2px solid #64748b;
        }
        .tree-node-box.pass {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            border: 2px solid #4ade80;
        }
        .tree-node-box.fail {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            border: 2px solid #f87171;
        }
        .tree-node-box.decision {
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            border: 2px solid #a78bfa;
        }

        /* Enhanced Connectors with Arrows */
        .tree-connector {
            width: 3px;
            height: 24px;
            background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
            position: relative;
        }

        .tree-connector::after {
            content: '▼';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            color: #8b5cf6;
            font-size: 10px;
        }

        .tree-connector.data-flow::before {
            content: attr(data-label);
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            color: #94a3b8;
            white-space: nowrap;
        }

        /* Branch with Pillars */
        .tree-branch {
            display: flex;
            gap: 16px;
            position: relative;
            padding: 10px 0;
        }

        .tree-branch::before {
            content: '';
            position: absolute;
            top: 0;
            left: 10%;
            width: 80%;
            height: 3px;
            background: linear-gradient(90deg, transparent 0%, #6366f1 20%, #8b5cf6 80%, transparent 100%);
        }

        .tree-branch .tree-node::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 50%;
            width: 3px;
            height: 10px;
            background: #6366f1;
            transform: translateX(-50%);
        }

        /* Phase Flow Indicator */
        .phase-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 20px;
            margin: 8px 0;
        }

        .phase-flow-arrow {
            color: #6366f1;
            font-size: 18px;
            animation: pulse-arrow 1.5s infinite;
        }

        @keyframes pulse-arrow {
            0%, 100% { opacity: 0.5; transform: translateY(0); }
            50% { opacity: 1; transform: translateY(2px); }
        }

        .phase-flow-label {
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Pillar mini cards */
        .tree-node-box.pillar {
            min-width: 120px;
            padding: 10px 14px;
        }

        /* Factor Cards */
        .factor-positive {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.1));
            border-left: 3px solid #22c55e;
        }
        .factor-negative {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
            border-left: 3px solid #ef4444;
        }
        .factor-neutral {
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.2), rgba(202, 138, 4, 0.1));
            border-left: 3px solid #eab308;
        }

        /* Pillar Score Bar */
        .pillar-bar {
            height: 8px;
            border-radius: 4px;
            background: #1e293b;
            overflow: hidden;
        }
        .pillar-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        /* V4.6 Neural Network Graph Styles */
        .neural-graph {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            padding: 24px;
            min-height: 400px;
            position: relative;
            overflow-x: auto;
        }

        /* V4.7: Vertical layout for global reasoning network */
        .neural-graph-vertical {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
            min-height: 400px;
            position: relative;
        }

        .neural-graph-vertical .neural-layer {
            display: flex;
            flex-direction: row;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .neural-layer {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            min-width: 120px;
        }

        .neural-layer-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #94a3b8;
            margin-bottom: 8px;
        }

        .neural-connector {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 30px;
            margin: 8px 0;
        }

        .neural-connector::before {
            content: '';
            width: 2px;
            height: 100%;
            background: linear-gradient(180deg, #8b5cf6 0%, #6366f1 100%);
        }

        .neural-node {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            position: relative;
        }

        .neural-node:hover {
            transform: scale(1.15);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.5);
            z-index: 10;
        }

        .neural-node.source {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            border: 2px solid #60a5fa;
        }

        .neural-node.detection {
            background: linear-gradient(135deg, #10b981, #059669);
            border: 2px solid #34d399;
        }

        .neural-node.pillar {
            background: linear-gradient(135deg, #f97316, #ea580c);
            border: 2px solid #fb923c;
        }

        .neural-node.aggregation {
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            border: 2px solid #a78bfa;
        }

        .neural-node.decision {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            border: 2px solid #4ade80;
            width: 60px;
            height: 60px;
        }

        .neural-node.decision.reject {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            border: 2px solid #f87171;
        }

        .neural-node.decision.watch {
            background: linear-gradient(135deg, #eab308, #ca8a04);
            border: 2px solid #fde047;
        }

        .neural-node-value {
            font-size: 10px;
            font-weight: bold;
            color: white;
        }

        .neural-node-label {
            position: absolute;
            bottom: -20px;
            font-size: 9px;
            color: #94a3b8;
            white-space: nowrap;
            max-width: 80px;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .neural-edge {
            position: absolute;
            height: 2px;
            transform-origin: left center;
            z-index: 0;
        }

        .neural-edge.strong {
            background: linear-gradient(90deg, #22c55e, #4ade80);
            height: 3px;
        }

        .neural-edge.medium {
            background: linear-gradient(90deg, #eab308, #fde047);
        }

        .neural-edge.weak {
            background: linear-gradient(90deg, #64748b, #94a3b8);
            opacity: 0.5;
        }

        /* SVG connections */
        .neural-connections {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .neural-connections line {
            stroke-linecap: round;
        }

        .neural-tooltip {
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            max-width: 250px;
            z-index: 100;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        }
    </style>
    ''')

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('account_tree', size='lg').classes('text-blue-400')
            ui.label('Arbre de Décision').classes('text-xl font-bold text-white')
            create_help_button('Interprétation des Scores', get_score_interpretation())
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6 gap-4'):
        # Quick stats row
        with ui.row().classes('w-full gap-4 flex-wrap'):
            journeys = get_all_journeys_sorted()
            stats = {
                'total': len(journeys),
                'signal': sum(1 for j in journeys if j.get('current_status') == 'signal'),
                'rejected': sum(1 for j in journeys if j.get('current_status') == 'reject'),
            }

            stat_configs = [
                ('Symboles Suivis', stats['total'], 'visibility', 'bg-slate-700'),
                ('Signaux Actifs', stats['signal'], 'trending_up', 'bg-green-800'),
                ('Rejetés', stats['rejected'], 'block', 'bg-red-800'),
            ]

            for label, count, icon_name, bg in stat_configs:
                with ui.card().classes(f'{bg} p-4 glass-card'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.icon(icon_name).classes('text-white')
                        ui.label(label).classes('text-white text-sm')
                    ui.label(str(count)).classes('text-white text-2xl font-bold')

        # Filter
        with ui.card().classes('w-full glass-card p-4'):
            with ui.row().classes('gap-4 items-center'):
                status_filter = ui.toggle(['Tous', 'Signaux', 'Rejetés'], value='Tous').classes('text-white')
                ui.space()
                search_input = ui.input(placeholder='Rechercher...').classes('w-48')
                # V4.8: Manual refresh button (replaces auto-timer)
                ui.button('🔄', on_click=lambda: update_display()).props('flat round').classes('text-blue-400').tooltip('Rafraîchir la liste')

        # Main content container
        content_container = ui.column().classes('w-full gap-4')

        # V4.8: PERSISTENT DIALOGS - Created at page level to survive re-renders
        # Store current dialog data
        current_pillar_data = {'name': '', 'data': {}, 'score': 0}
        current_discovery_data = {'symbol': '', 'sources': [], 'reasons': []}
        current_factors_data = {'key': [], 'risk': []}

        # Pillar detail dialog
        with ui.dialog() as pillar_dialog:
            with ui.card().classes('glass-card w-[500px] max-h-[80vh] overflow-auto p-4'):
                pillar_title = ui.label('').classes('text-white text-xl font-bold')
                pillar_signal = ui.label('').classes('font-bold')
                ui.separator().classes('my-4')
                ui.label('Raisonnement:').classes('text-blue-400 font-bold mb-2')
                pillar_reasoning = ui.label('').classes('text-slate-200 text-sm whitespace-pre-wrap bg-slate-800 p-3 rounded')
                pillar_factors_container = ui.column().classes('mt-4')
                ui.button('Fermer', on_click=pillar_dialog.close).classes('mt-4 w-full')

        def show_pillar_details(pillar_name: str, pillar_data: Dict, score_val: float):
            """Show pillar details in persistent dialog"""
            reasoning = pillar_data.get('reasoning', f'Score: {score_val}/25')
            factors = pillar_data.get('factors', [])
            signal = pillar_data.get('signal', 'neutral')

            pillar_title.text = f'{pillar_name.upper()} - {score_val}/25'
            signal_color = 'text-green-400' if 'buy' in signal.lower() else 'text-red-400' if 'sell' in signal.lower() else 'text-yellow-400'
            pillar_signal.text = signal.upper()
            pillar_signal.classes(signal_color, remove='text-green-400 text-red-400 text-yellow-400')
            pillar_reasoning.text = reasoning

            pillar_factors_container.clear()
            if factors:
                with pillar_factors_container:
                    ui.label('Facteurs Contributifs:').classes('text-purple-400 font-bold mb-2')
                    for factor in factors[:5]:
                        if isinstance(factor, dict):
                            factor_name = factor.get('name', factor.get('factor', str(factor)))
                            with ui.row().classes('gap-2 items-center mb-1'):
                                ui.icon('arrow_right', size='xs').classes('text-purple-400')
                                ui.label(f"{factor_name}").classes('text-slate-300 text-sm')
                        else:
                            with ui.row().classes('gap-2 items-center mb-1'):
                                ui.icon('arrow_right', size='xs').classes('text-purple-400')
                                ui.label(str(factor)).classes('text-slate-300 text-sm')

            pillar_dialog.open()

        # Discovery detail dialog
        with ui.dialog() as discovery_dialog:
            with ui.card().classes('glass-card w-[400px] p-4'):
                discovery_title = ui.label('').classes('text-white text-xl font-bold mb-4')
                ui.label('Sources de détection:').classes('text-blue-400 font-bold mb-2')
                discovery_sources_container = ui.column()
                discovery_reasons_container = ui.column().classes('mt-3')
                ui.button('Fermer', on_click=discovery_dialog.close).classes('mt-4 w-full')

        def show_discovery_details(symbol: str, sources: List, reasons: List):
            """Show discovery details in persistent dialog"""
            discovery_title.text = f'Découverte: {symbol}'

            discovery_sources_container.clear()
            with discovery_sources_container:
                for src in sources:
                    with ui.row().classes('gap-2'):
                        ui.icon('check', size='xs').classes('text-blue-400')
                        ui.label(src).classes('text-slate-300')

            discovery_reasons_container.clear()
            if reasons:
                with discovery_reasons_container:
                    ui.label('Raisons:').classes('text-green-400 font-bold mb-2')
                    for reason in reasons[:3]:
                        ui.label(f"• {reason}").classes('text-slate-300 text-sm')

            discovery_dialog.open()

        # Factors detail dialog
        with ui.dialog() as factors_dialog:
            with ui.card().classes('glass-card w-[450px] p-4'):
                ui.label('Facteurs de Décision').classes('text-white text-xl font-bold mb-4')
                factors_key_container = ui.column()
                factors_risk_container = ui.column().classes('mt-3')
                ui.button('Fermer', on_click=factors_dialog.close).classes('mt-4 w-full')

        def show_factors_details(key_factors: List, risk_factors: List):
            """Show factors details in persistent dialog"""
            factors_key_container.clear()
            if key_factors:
                with factors_key_container:
                    ui.label('✅ Facteurs Positifs:').classes('text-green-400 font-bold mb-2')
                    for f in key_factors[:5]:
                        factor_text = f.get('description', str(f)) if isinstance(f, dict) else str(f)
                        ui.label(f"• {factor_text}").classes('text-slate-300 text-sm')

            factors_risk_container.clear()
            if risk_factors:
                with factors_risk_container:
                    ui.label('⚠️ Facteurs de Risque:').classes('text-red-400 font-bold mb-2')
                    for f in risk_factors[:5]:
                        ui.label(f"• {f}").classes('text-slate-300 text-sm')

            factors_dialog.open()

        def format_timestamp(ts_str: str) -> str:
            try:
                dt = datetime.fromisoformat(ts_str)
                return dt.strftime("%d/%m %H:%M")
            except:
                return ts_str[:16] if ts_str else ''

        def render_decision_tree(journey: Dict):
            """Render a visual decision tree for a symbol with clickable nodes showing reasoning"""
            symbol = journey.get('symbol', '?')
            status = journey.get('current_status', 'unknown')
            steps = journey.get('journey', [])

            # Extract data from journey steps
            discovery_data = next((s.get('data', {}) for s in steps if s.get('step') == 'discovery'), {})
            analysis_data = next((s.get('data', {}) for s in steps if s.get('step') == 'analysis'), {})

            # Get pillar scores and details (convert from internal -100/+100 to display 0-25)
            technical_raw = analysis_data.get('technical', 0)
            fundamental_raw = analysis_data.get('fundamental', 0)
            sentiment_raw = analysis_data.get('sentiment', 0)
            news_raw = analysis_data.get('news', 0)

            # V4.5: Convert to display scale [0-25] per pillar
            technical = convert_pillar_score_to_display(technical_raw)
            fundamental = convert_pillar_score_to_display(fundamental_raw)
            sentiment = convert_pillar_score_to_display(sentiment_raw)
            news = convert_pillar_score_to_display(news_raw)

            # V4.8: Total = SUM of pillar display scores (not the weighted_score from engine)
            # This ensures visual consistency: T + F + S + N = Total
            total = round(technical + fundamental + sentiment + news, 1)

            # Get detailed pillar data (V4.3)
            pillar_details = analysis_data.get('pillar_details', {})
            key_factors = analysis_data.get('key_factors', [])
            risk_factors = analysis_data.get('risk_factors', [])

            # Decision result - V4.8: Use TOTAL (pillar sum) threshold, not journey.current_score
            # A symbol with total >= 55 should show SIGNAL (pending execution), not REJETÉ
            is_signal = status in ['signal', 'buy'] or total >= 55
            if total >= 55:
                decision_text = 'SIGNAL' if status in ['signal', 'buy'] else 'EN ATTENTE'  # Pending market open
            else:
                decision_text = 'REJETÉ'

            # V4.8: Use persistent dialogs (defined at page level)

            with ui.column().classes('decision-tree'):
                # Root node: Symbol with discovery info
                with ui.element('div').classes('tree-node'):
                    sources = discovery_data.get('sources', ['Scan'])
                    discovery_reasoning = discovery_data.get('reasons', ['Détecté par scan'])

                    with ui.element('div').classes('tree-node-box root cursor-pointer').on(
                        'click', lambda sym=symbol, src=sources, rsn=discovery_reasoning: show_discovery_details(sym, src, rsn)
                    ):
                        ui.label(symbol).classes('text-white font-bold text-lg')
                        ui.label(f"via {', '.join(sources[:2])}").classes('text-blue-200 text-xs')
                        ui.label('(cliquez pour détails)').classes('text-blue-300 text-xs italic')

                    ui.element('div').classes('tree-connector')

                # Pillar nodes row - CLICKABLE with reasoning
                with ui.element('div').classes('tree-branch'):
                    pillars = [
                        ('Technical', technical, pillar_details.get('technical', {}), '🔧'),
                        ('Fundamental', fundamental, pillar_details.get('fundamental', {}), '📊'),
                        ('Sentiment', sentiment, pillar_details.get('sentiment', {}), '💬'),
                        ('News', news, pillar_details.get('news', {}), '📰'),
                    ]

                    for name, score_val, detail, emoji in pillars:
                        # V4.8: Detect if pillar has been analyzed
                        # Has reasoning OR score is not 0 (0 = no data)
                        has_reasoning = bool(detail.get('reasoning'))
                        is_analyzed = has_reasoning or score_val > 0
                        is_pass = score_val >= 15 if is_analyzed else False
                        reasoning_preview = detail.get('reasoning', '')[:50] + '...' if detail.get('reasoning', '') else ''

                        # V4.8: Get data_quality for this pillar (0-1, default 0.5)
                        data_quality = detail.get('data_quality', 0.5 if is_analyzed else 0)
                        data_quality_pct = int(data_quality * 100)

                        with ui.element('div').classes('tree-node'):
                            # Color: green=pass, red=fail, gray=not analyzed
                            if not is_analyzed:
                                box_class = 'tree-node-box check cursor-pointer'  # Gray/neutral
                            else:
                                box_class = f'tree-node-box {"pass" if is_pass else "fail"} cursor-pointer'

                            # V4.8: Use persistent dialog function
                            with ui.element('div').classes(box_class).on(
                                'click', lambda n=name, d=detail, s=score_val: show_pillar_details(n, d, s)
                            ):
                                ui.label(f'{emoji} {name}').classes('text-white font-bold text-sm')
                                # Show score (0 = no data)
                                if is_analyzed:
                                    ui.label(f'{score_val}/25').classes('text-white text-xs')
                                    # V4.8: Show data quality indicator
                                    quality_color = 'text-green-300' if data_quality >= 0.7 else 'text-yellow-300' if data_quality >= 0.4 else 'text-red-300'
                                    ui.label(f'📊 {data_quality_pct}%').classes(f'{quality_color} text-xs')
                                else:
                                    ui.label('0/25').classes('text-slate-400 text-xs')
                                # Show mini reasoning preview
                                if reasoning_preview:
                                    ui.label(reasoning_preview).classes('text-slate-300 text-xs max-w-[120px] truncate')
                                elif not is_analyzed:
                                    ui.label("Pas de données").classes('text-slate-400 text-xs italic')
                                ui.label('▼ détails').classes('text-white text-xs opacity-70')

                ui.element('div').classes('tree-connector')

                # Key factors summary (clickable) - V4.8: Use persistent dialog
                if key_factors or risk_factors:
                    with ui.element('div').classes('tree-node'):
                        with ui.element('div').classes('tree-node-box check cursor-pointer').on(
                            'click', lambda kf=key_factors, rf=risk_factors: show_factors_details(kf, rf)
                        ):
                            ui.label('Facteurs Clés').classes('text-white font-bold text-sm')
                            ui.label(f'+{len(key_factors)} / -{len(risk_factors)}').classes('text-white text-xs')

                        ui.element('div').classes('tree-connector')

                # Score aggregation
                with ui.element('div').classes('tree-node'):
                    with ui.element('div').classes('tree-node-box check'):
                        ui.label('Score Total').classes('text-white font-bold text-sm')
                        score_color = 'text-green-400' if total >= 55 else 'text-yellow-400' if total >= 40 else 'text-red-400'
                        ui.label(f'{total}/100').classes(f'{score_color} text-lg font-bold')
                        convergence = sum(1 for p in [technical, fundamental, sentiment, news] if p >= 15)
                        ui.label(f'{convergence}/4 piliers OK').classes('text-slate-300 text-xs')
                    ui.element('div').classes('tree-connector')

                # Threshold check
                with ui.element('div').classes('tree-node'):
                    threshold_pass = total >= 55
                    with ui.element('div').classes(f'tree-node-box {"pass" if threshold_pass else "fail"}'):
                        ui.label('Seuil >= 55?').classes('text-white font-bold text-sm')
                        ui.label('OUI ✓' if threshold_pass else 'NON ✗').classes('text-white text-xs font-bold')
                    ui.element('div').classes('tree-connector')

                # Final decision - V4.8: Color based on score threshold
                with ui.element('div').classes('tree-node'):
                    decision_class = 'pass' if total >= 55 else 'fail'
                    with ui.element('div').classes(f'tree-node-box {decision_class}'):
                        ui.label(decision_text).classes('text-white font-bold text-lg')
                        ui.label(f'Score: {total}/100').classes('text-white text-xs')

        def render_factor_breakdown(journey: Dict):
            """Render detailed factor breakdown with + and - factors"""
            steps = journey.get('journey', [])
            analysis_step = next((s for s in steps if s.get('step') == 'analysis'), None)

            if not analysis_step:
                ui.label('Pas encore analysé').classes('text-slate-400 italic')
                return

            data = analysis_step.get('data', {})
            reasoning = analysis_step.get('reasoning', '')

            # V4.5: Pillar scores with visual bars (convert from internal -100/+100 to display 0-25)
            pillars = [
                ('Technical', convert_pillar_score_to_display(data.get('technical', 0)), 25, 'blue'),
                ('Fundamental', convert_pillar_score_to_display(data.get('fundamental', 0)), 25, 'purple'),
                ('Sentiment', convert_pillar_score_to_display(data.get('sentiment', 0)), 25, 'yellow'),
                ('News', convert_pillar_score_to_display(data.get('news', 0)), 25, 'orange'),
            ]

            with ui.row().classes('items-center gap-2 mb-3'):
                ui.label('Scores des 4 Piliers').classes('text-white font-bold')
                create_help_button('Les 4 Piliers', get_four_pillars_explanation())

            for name, score, max_score, color in pillars:
                with ui.row().classes('w-full items-center gap-3 mb-2'):
                    ui.label(name).classes('text-slate-300 w-24 text-sm')

                    # Score bar
                    pct = (score / max_score) * 100
                    bar_color = '#22c55e' if pct >= 60 else '#eab308' if pct >= 40 else '#ef4444'

                    with ui.element('div').classes('pillar-bar flex-1'):
                        ui.element('div').classes('pillar-bar-fill').style(
                            f'width: {pct}%; background: {bar_color};'
                        )

                    score_color = 'text-green-400' if score >= 15 else 'text-yellow-400' if score >= 10 else 'text-red-400'
                    ui.label(f'{score}/{max_score}').classes(f'{score_color} w-16 text-right font-bold')

            ui.separator().classes('my-4')

            # Extract factors from reasoning
            ui.label('Facteurs de Décision').classes('text-white font-bold mb-3')

            # Parse reasoning into factors
            positive_factors = []
            negative_factors = []
            neutral_factors = []

            for line in reasoning.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Determine factor type based on keywords
                if any(kw in line.lower() for kw in ['fort', 'strong', 'positif', 'bullish', 'breakout', 'confirmé', 'croissant']):
                    positive_factors.append(line.replace('• ', ''))
                elif any(kw in line.lower() for kw in ['faible', 'weak', 'négatif', 'bearish', 'pas de', 'non', 'absent']):
                    negative_factors.append(line.replace('• ', ''))
                elif line.startswith('•'):
                    neutral_factors.append(line.replace('• ', ''))

            # Positive factors
            if positive_factors:
                with ui.card().classes('factor-positive p-3 mb-2'):
                    with ui.row().classes('gap-2 items-center mb-2'):
                        ui.icon('add_circle', color='green').classes('text-green-400')
                        ui.label('Facteurs Positifs').classes('text-green-400 font-bold')

                    for factor in positive_factors[:5]:
                        with ui.row().classes('gap-2 items-start'):
                            ui.icon('check', size='xs').classes('text-green-400 mt-1')
                            ui.label(factor).classes('text-slate-200 text-sm')

            # Negative factors
            if negative_factors:
                with ui.card().classes('factor-negative p-3 mb-2'):
                    with ui.row().classes('gap-2 items-center mb-2'):
                        ui.icon('remove_circle', color='red').classes('text-red-400')
                        ui.label('Facteurs Négatifs').classes('text-red-400 font-bold')

                    for factor in negative_factors[:5]:
                        with ui.row().classes('gap-2 items-start'):
                            ui.icon('close', size='xs').classes('text-red-400 mt-1')
                            ui.label(factor).classes('text-slate-200 text-sm')

            # Neutral/Info factors
            if neutral_factors:
                with ui.card().classes('factor-neutral p-3 mb-2'):
                    with ui.row().classes('gap-2 items-center mb-2'):
                        ui.icon('info', color='yellow').classes('text-yellow-400')
                        ui.label('Observations').classes('text-yellow-400 font-bold')

                    for factor in neutral_factors[:5]:
                        with ui.row().classes('gap-2 items-start'):
                            ui.icon('arrow_right', size='xs').classes('text-yellow-400 mt-1')
                            ui.label(factor).classes('text-slate-200 text-sm')

        def render_symbol_card(journey: Dict):
            """Render a complete symbol analysis card with tree and factors"""
            symbol = journey.get('symbol', '?')
            status = journey.get('current_status', 'unknown')
            first_seen = format_timestamp(journey.get('first_seen', ''))
            last_update = format_timestamp(journey.get('last_update', ''))

            # V4.8: Calculate score from pillar display values (same as decision tree)
            # This ensures header and tree show the same score
            steps = journey.get('journey', [])
            analysis_data = next((s.get('data', {}) for s in steps if s.get('step') == 'analysis'), {})

            technical_raw = analysis_data.get('technical', 0)
            fundamental_raw = analysis_data.get('fundamental', 0)
            sentiment_raw = analysis_data.get('sentiment', 0)
            news_raw = analysis_data.get('news', 0)

            # Convert to display scale [0-25] per pillar
            technical = convert_pillar_score_to_display(technical_raw)
            fundamental = convert_pillar_score_to_display(fundamental_raw)
            sentiment = convert_pillar_score_to_display(sentiment_raw)
            news = convert_pillar_score_to_display(news_raw)

            # Total = sum of pillar display scores (0-100)
            score = round(technical + fundamental + sentiment + news, 1)

            # Status config
            status_config = {
                'discovery': ('🔍', 'En découverte', 'border-blue-500', 'bg-blue-900'),
                'analysis': ('📊', 'En analyse', 'border-yellow-500', 'bg-yellow-900'),
                'signal': ('✅', 'SIGNAL ACTIF', 'border-green-500', 'bg-green-900'),
                'buy': ('💰', 'ACHETÉ', 'border-emerald-500', 'bg-emerald-900'),
                'sell': ('📤', 'VENDU', 'border-orange-500', 'bg-orange-900'),
                'reject': ('❌', 'Rejeté', 'border-red-500', 'bg-red-900'),
            }
            emoji, status_label, border_color, status_bg = status_config.get(
                status, ('•', status, 'border-slate-500', 'bg-slate-800')
            )

            score_color = 'text-green-400' if score >= 55 else 'text-yellow-400' if score >= 40 else 'text-red-400'

            with ui.card().classes(f'w-full glass-card border-l-4 {border_color}'):
                # Header
                with ui.row().classes('w-full justify-between items-center p-4 border-b border-slate-700'):
                    with ui.row().classes('gap-3 items-center'):
                        ui.label(f'{emoji} {symbol}').classes('text-2xl font-bold text-white')
                        with ui.element('span').classes(f'{status_bg} px-3 py-1 rounded-full'):
                            ui.label(status_label).classes('text-white text-sm font-medium')

                    with ui.column().classes('items-end'):
                        ui.label(f'{score}/100').classes(f'{score_color} text-3xl font-bold')
                        ui.label(f'Vu: {first_seen}').classes('text-slate-500 text-xs')

                # Tabs for Tree, Factors, Narrative Report, Neural Graph, and History
                with ui.tabs().classes('w-full') as tabs:
                    tree_tab = ui.tab('Arbre', icon='account_tree')
                    neural_tab = ui.tab('🧠 Graphe', icon='hub')
                    factors_tab = ui.tab('Facteurs', icon='list')
                    narrative_tab = ui.tab('📝 Rapport', icon='auto_awesome')
                    history_tab = ui.tab('📜 Historique', icon='history')

                with ui.tab_panels(tabs, value=tree_tab).classes('w-full'):
                    with ui.tab_panel(tree_tab):
                        with ui.scroll_area().classes('w-full'):
                            render_decision_tree(journey)

                    with ui.tab_panel(neural_tab):
                        # V4.6: Neural Network Graph Visualization
                        with ui.element('div').classes('p-4'):
                            ui.label('🧠 Graphe de Raisonnement').classes('text-white font-bold text-lg mb-2')
                            ui.label('Visualisation du cheminement de pensée comme un réseau neuronal').classes('text-slate-400 text-sm mb-4')

                            # Build the reasoning graph
                            builder = get_graph_builder()
                            graph = builder.build_from_journey(symbol, journey)

                            # Render the neural network visualization
                            with ui.element('div').classes('neural-graph bg-slate-900 rounded-xl'):
                                # Layer 1: Sources (Input)
                                source_nodes = [n for n in graph.nodes if n.node_type.value.startswith('source_')]
                                with ui.element('div').classes('neural-layer'):
                                    ui.label('Sources').classes('neural-layer-label')
                                    for node in source_nodes:
                                        with ui.element('div').classes('neural-node source').on(
                                            'click', lambda n=node: show_node_dialog(n)
                                        ):
                                            ui.label(f'{node.value*100:.0f}%').classes('neural-node-value')
                                            ui.label(node.label[:10]).classes('neural-node-label')

                                # Layer 2: Detection (Hidden 1)
                                detection_nodes = [n for n in graph.nodes if n.node_type.value.startswith('detection_')]
                                if detection_nodes:
                                    with ui.element('div').classes('neural-layer'):
                                        ui.label('Détection').classes('neural-layer-label')
                                        for node in detection_nodes:
                                            with ui.element('div').classes('neural-node detection').on(
                                                'click', lambda n=node: show_node_dialog(n)
                                            ):
                                                ui.label(f'{node.value*100:.0f}%').classes('neural-node-value')
                                                ui.label(node.label[:10]).classes('neural-node-label')

                                # Layer 3: Pillars (Hidden 2)
                                pillar_nodes = [n for n in graph.nodes if n.node_type.value.startswith('pillar_')]
                                with ui.element('div').classes('neural-layer'):
                                    ui.label('4 Piliers').classes('neural-layer-label')
                                    for node in pillar_nodes:
                                        with ui.element('div').classes('neural-node pillar').on(
                                            'click', lambda n=node: show_node_dialog(n)
                                        ):
                                            ui.label(f'{node.value*100:.0f}%').classes('neural-node-value')
                                            ui.label(node.label[:8]).classes('neural-node-label')

                                # Layer 4: Aggregation (Hidden 3)
                                agg_nodes = [n for n in graph.nodes if n.node_type.value.startswith('aggregation_')]
                                with ui.element('div').classes('neural-layer'):
                                    ui.label('Agrégation').classes('neural-layer-label')
                                    for node in agg_nodes:
                                        with ui.element('div').classes('neural-node aggregation').on(
                                            'click', lambda n=node: show_node_dialog(n)
                                        ):
                                            ui.label(f'{node.value*100:.0f}%').classes('neural-node-value')
                                            ui.label('Score').classes('neural-node-label')

                                # Layer 5: Decision (Output)
                                decision_nodes = [n for n in graph.nodes if n.node_type.value.startswith('decision_')]
                                with ui.element('div').classes('neural-layer'):
                                    ui.label('Décision').classes('neural-layer-label')
                                    for node in decision_nodes:
                                        decision_class = 'decision'
                                        if node.node_type == NodeType.DECISION_REJECT:
                                            decision_class += ' reject'
                                        elif node.node_type == NodeType.DECISION_WATCH:
                                            decision_class += ' watch'
                                        with ui.element('div').classes(f'neural-node {decision_class}').on(
                                            'click', lambda n=node: show_node_dialog(n)
                                        ):
                                            ui.label(node.label).classes('neural-node-value text-xs')

                            # Node dialog function
                            def show_node_dialog(node):
                                with ui.dialog() as dlg, ui.card().classes('glass-card w-[450px]'):
                                    ui.label(f'{node.label}').classes('text-white text-xl font-bold mb-2')
                                    ui.label(f'Type: {node.node_type.value}').classes('text-slate-400 text-sm mb-4')

                                    with ui.card().classes('bg-slate-800 p-3 mb-3'):
                                        ui.label('Valeur d\'activation').classes('text-blue-400 font-bold text-sm')
                                        with ui.row().classes('items-center gap-2'):
                                            ui.linear_progress(value=node.value).classes('flex-1')
                                            ui.label(f'{node.value*100:.1f}%').classes('text-white font-bold')

                                    ui.label('Raisonnement:').classes('text-purple-400 font-bold mb-1')
                                    ui.label(node.reasoning).classes('text-slate-300 text-sm whitespace-pre-wrap')

                                    if node.metadata:
                                        ui.label('Métadonnées:').classes('text-orange-400 font-bold mt-3 mb-1')
                                        for k, v in list(node.metadata.items())[:5]:
                                            ui.label(f'• {k}: {v}').classes('text-slate-400 text-xs')

                                    # Show embedding (first 10 values)
                                    ui.label('Embedding (extrait):').classes('text-cyan-400 font-bold mt-3 mb-1')
                                    non_zero = [(i, v) for i, v in enumerate(node.embedding) if v != 0]
                                    if non_zero:
                                        for i, v in non_zero[:5]:
                                            ui.label(f'dim[{i}] = {v:.3f}').classes('text-slate-500 text-xs font-mono')
                                    else:
                                        ui.label('Vecteur nul').classes('text-slate-500 text-xs')

                                    ui.button('Fermer', on_click=dlg.close).classes('mt-4 w-full')
                                dlg.open()

                            # Save graph for learning
                            store = get_graph_store()
                            graph_id = store.save(graph)

                            # Graph info
                            with ui.row().classes('mt-4 gap-4 text-xs text-slate-500'):
                                ui.label(f'Nodes: {len(graph.nodes)}')
                                ui.label(f'Edges: {len(graph.edges)}')
                                ui.label(f'Vector dim: 80')
                                ui.label(f'ID: {graph_id[:8]}...')

                    with ui.tab_panel(factors_tab):
                        with ui.column().classes('p-4'):
                            render_factor_breakdown(journey)

                    with ui.tab_panel(narrative_tab):
                        narrative_container = ui.column().classes('p-4 gap-4 w-full')

                        async def generate_and_display_narrative():
                            """Generate and display narrative report"""
                            # V4.8: Clear and show loading state
                            narrative_container.clear()
                            with narrative_container:
                                with ui.row().classes('items-center gap-2'):
                                    ui.spinner('dots', size='sm').classes('text-purple-400')
                                    ui.label('⏳ Génération du rapport en cours...').classes('text-slate-400')
                                ui.label('Appel à l\'API OpenRouter (Gemini)...').classes('text-slate-500 text-xs')

                            # V4.8: Force UI refresh to show loading state
                            await ui.run_javascript('null', timeout=0.1)

                            try:
                                narrative = await generate_symbol_narrative(symbol, journey)
                                ui.notify(f'✅ Rapport généré pour {symbol}', type='positive', timeout=3000)

                                narrative_container.clear()
                                with narrative_container:
                                    # Watchlist Summary
                                    if narrative.get('watchlist_summary'):
                                        with ui.card().classes('w-full bg-slate-800 p-4'):
                                            ui.label('🔍 POURQUOI CE SYMBOLE?').classes('text-blue-400 font-bold text-lg mb-2')
                                            ui.label(narrative['watchlist_summary']).classes('text-slate-200 text-sm leading-relaxed whitespace-pre-wrap')

                                    # Decision Reasoning
                                    if narrative.get('decision_reasoning'):
                                        decision = narrative.get('decision', 'WATCH')
                                        decision_color = 'bg-green-900' if decision == 'BUY' else 'bg-red-900' if decision == 'PASS' else 'bg-yellow-900'
                                        with ui.card().classes(f'w-full {decision_color} p-4'):
                                            ui.label(f'💰 DÉCISION: {decision}').classes('text-white font-bold text-lg mb-2')
                                            ui.label(narrative['decision_reasoning']).classes('text-slate-200 text-sm leading-relaxed whitespace-pre-wrap')

                                    # Pillar Explanations
                                    pillar_explanations = narrative.get('pillar_explanations', {})
                                    if pillar_explanations:
                                        with ui.card().classes('w-full bg-slate-800 p-4'):
                                            ui.label('📊 ANALYSE DES 4 PILIERS').classes('text-purple-400 font-bold text-lg mb-3')
                                            for pillar, explanation in pillar_explanations.items():
                                                if explanation:
                                                    pillar_emoji = {'technical': '🔧', 'fundamental': '📈', 'sentiment': '💬', 'news': '📰'}.get(pillar, '📊')
                                                    with ui.element('div').classes('mb-3'):
                                                        ui.label(f'{pillar_emoji} {pillar.capitalize()}').classes('text-slate-300 font-bold text-sm')
                                                        ui.label(explanation).classes('text-slate-400 text-sm ml-4 leading-relaxed')

                                    # Reservations / Risks
                                    if narrative.get('reservations'):
                                        with ui.card().classes('w-full bg-slate-800 border-l-4 border-orange-500 p-4'):
                                            ui.label('⚠️ RÉSERVES ET RISQUES').classes('text-orange-400 font-bold text-lg mb-2')
                                            ui.label(narrative['reservations']).classes('text-slate-200 text-sm leading-relaxed whitespace-pre-wrap')

                                    # Exit Triggers
                                    exit_triggers = narrative.get('exit_triggers', [])
                                    if exit_triggers:
                                        with ui.card().classes('w-full bg-slate-800 p-4'):
                                            ui.label('🚪 CONDITIONS DE SORTIE').classes('text-red-400 font-bold text-lg mb-2')
                                            for trigger in exit_triggers:
                                                with ui.row().classes('gap-2 items-center'):
                                                    ui.icon('exit_to_app', size='xs').classes('text-red-400')
                                                    ui.label(trigger).classes('text-slate-300 text-sm')

                                    # Error message if any
                                    if narrative.get('error'):
                                        ui.label(f"Note: {narrative['error']}").classes('text-yellow-500 text-xs mt-2 italic')

                                    # V4.8: Add regenerate button at end
                                    ui.button('🔄 Régénérer le rapport', on_click=generate_and_display_narrative).classes('mt-4')

                                # V4.8: Force UI refresh after content is added
                                await ui.run_javascript('null', timeout=0.1)

                            except Exception as e:
                                ui.notify(f'❌ Erreur génération rapport: {str(e)[:50]}', type='negative', timeout=5000)
                                narrative_container.clear()
                                with narrative_container:
                                    ui.label(f'❌ Erreur: {e}').classes('text-red-400')
                                    ui.label('Vérifiez que OPENROUTER_API_KEY est configuré dans .env').classes('text-slate-500 text-sm')
                                    ui.button('🔄 Réessayer', on_click=generate_and_display_narrative).classes('mt-2')

                        # Check for cached narrative first
                        cached = get_cached_narrative(symbol)
                        with narrative_container:
                            if cached and (cached.get('watchlist_summary') or cached.get('decision_reasoning')):
                                # Display cached narrative
                                if cached.get('watchlist_summary'):
                                    with ui.card().classes('w-full bg-slate-800 p-4'):
                                        ui.label('🔍 POURQUOI CE SYMBOLE?').classes('text-blue-400 font-bold text-lg mb-2')
                                        ui.label(cached['watchlist_summary']).classes('text-slate-200 text-sm leading-relaxed')

                                if cached.get('decision_reasoning'):
                                    decision = cached.get('decision', 'WATCH')
                                    decision_color = 'bg-green-900' if decision == 'BUY' else 'bg-red-900' if decision == 'PASS' else 'bg-yellow-900'
                                    with ui.card().classes(f'w-full {decision_color} p-4'):
                                        ui.label(f'💰 DÉCISION: {decision}').classes('text-white font-bold text-lg mb-2')
                                        ui.label(cached['decision_reasoning']).classes('text-slate-200 text-sm leading-relaxed')

                                # V4.8: Show pillar explanations if available
                                pillar_explanations = cached.get('pillar_explanations', {})
                                if pillar_explanations:
                                    with ui.card().classes('w-full bg-slate-800 p-4'):
                                        ui.label('📊 ANALYSE DES 4 PILIERS').classes('text-purple-400 font-bold text-lg mb-3')
                                        for pillar, explanation in pillar_explanations.items():
                                            if explanation:
                                                pillar_emoji = {'technical': '🔧', 'fundamental': '📈', 'sentiment': '💬', 'news': '📰'}.get(pillar, '📊')
                                                with ui.element('div').classes('mb-3'):
                                                    ui.label(f'{pillar_emoji} {pillar.capitalize()}').classes('text-slate-300 font-bold text-sm')
                                                    ui.label(explanation).classes('text-slate-400 text-sm ml-4 leading-relaxed')

                                if cached.get('reservations'):
                                    with ui.card().classes('w-full bg-slate-800 border-l-4 border-orange-500 p-4'):
                                        ui.label('⚠️ RÉSERVES ET RISQUES').classes('text-orange-400 font-bold text-lg mb-2')
                                        ui.label(cached['reservations']).classes('text-slate-200 text-sm leading-relaxed')

                                ui.button('🔄 Régénérer le rapport', on_click=generate_and_display_narrative).classes('mt-2')
                            else:
                                # V4.8: Clear instruction with API check
                                import os
                                has_api_key = bool(os.getenv('OPENROUTER_API_KEY'))
                                if has_api_key:
                                    ui.label('📝 Générez un rapport narratif détaillé via IA').classes('text-slate-400 text-sm mb-2')
                                    ui.label('Le rapport explique POURQUOI ce symbole est intéressant et donne une recommandation argumentée.').classes('text-slate-500 text-xs mb-4')
                                    ui.button('✨ Générer le rapport IA', on_click=generate_and_display_narrative).props('color=purple').classes('w-full')
                                else:
                                    with ui.card().classes('w-full bg-yellow-900/30 p-4 border border-yellow-600'):
                                        ui.label('⚠️ API non configurée').classes('text-yellow-400 font-bold mb-2')
                                        ui.label('Pour générer des rapports narratifs, configurez OPENROUTER_API_KEY dans votre fichier .env').classes('text-slate-300 text-sm')
                                        ui.label('Obtenez une clé sur: openrouter.ai').classes('text-blue-400 text-xs mt-2')

                    # V4.8: History tab - shows past analyses and score evolution
                    with ui.tab_panel(history_tab):
                        with ui.column().classes('p-4 gap-4 w-full'):
                            ui.label('📜 Historique des Analyses').classes('text-white font-bold text-lg mb-2')

                            # Get historical analyses from store
                            history = get_analysis_history(symbol, limit=20)

                            if not history:
                                with ui.card().classes('w-full bg-slate-800 p-4'):
                                    ui.label('Aucune analyse historique disponible pour ce symbole.').classes('text-slate-400 italic')
                                    ui.label('Les analyses seront enregistrées au fur et à mesure des scans.').classes('text-slate-500 text-xs mt-1')
                            else:
                                # Score evolution mini-chart
                                with ui.card().classes('w-full bg-slate-800 p-4'):
                                    ui.label('📈 Évolution du Score').classes('text-purple-400 font-bold mb-3')

                                    # Prepare chart data (oldest to newest)
                                    chart_data = list(reversed(history[:10]))  # Last 10, chronological order
                                    timestamps = []
                                    scores = []
                                    for h in chart_data:
                                        ts = h.get('timestamp', '')[:10]  # Date only
                                        timestamps.append(ts)
                                        scores.append(h.get('total_score', 0))

                                    # Simple score display with trend
                                    if len(scores) >= 2:
                                        trend = scores[-1] - scores[0]
                                        trend_icon = '📈' if trend > 0 else '📉' if trend < 0 else '➡️'
                                        trend_color = 'text-green-400' if trend > 0 else 'text-red-400' if trend < 0 else 'text-slate-400'
                                        ui.label(f'{trend_icon} Tendance: {trend:+.1f} points').classes(f'{trend_color} text-sm mb-2')

                                    # Visual score bars
                                    with ui.column().classes('gap-1'):
                                        for i, (ts, sc) in enumerate(zip(timestamps, scores)):
                                            pct = max(0, min(100, sc))  # Clamp to 0-100
                                            bar_color = '#22c55e' if sc >= 55 else '#eab308' if sc >= 40 else '#ef4444'
                                            with ui.row().classes('items-center gap-2'):
                                                ui.label(ts[5:]).classes('text-slate-500 text-xs w-12')  # MM-DD
                                                with ui.element('div').classes('flex-1 h-4 bg-slate-700 rounded'):
                                                    ui.element('div').style(
                                                        f'width: {pct}%; background: {bar_color}; height: 100%; border-radius: 4px;'
                                                    )
                                                ui.label(f'{sc:.1f}').classes('text-slate-300 text-xs w-10 text-right')

                                # Analysis history table
                                with ui.card().classes('w-full bg-slate-800 p-4 mt-4'):
                                    ui.label('📋 Détail des Analyses Passées').classes('text-blue-400 font-bold mb-3')

                                    # Header
                                    with ui.row().classes('w-full gap-2 border-b border-slate-600 pb-2 mb-2'):
                                        ui.label('Date').classes('text-slate-400 text-xs font-bold w-24')
                                        ui.label('Score').classes('text-slate-400 text-xs font-bold w-12')
                                        ui.label('Tech').classes('text-slate-400 text-xs font-bold w-10')
                                        ui.label('Fund').classes('text-slate-400 text-xs font-bold w-10')
                                        ui.label('Sent').classes('text-slate-400 text-xs font-bold w-10')
                                        ui.label('News').classes('text-slate-400 text-xs font-bold w-10')
                                        ui.label('Décision').classes('text-slate-400 text-xs font-bold flex-1')

                                    # Data rows (newest first)
                                    for h in history[:10]:
                                        ts = h.get('timestamp', '')[:16].replace('T', ' ')
                                        total = h.get('total_score', 0)
                                        tech = convert_pillar_score_to_display(h.get('technical_score', 0))
                                        fund = convert_pillar_score_to_display(h.get('fundamental_score', 0))
                                        sent = convert_pillar_score_to_display(h.get('sentiment_score', 0))
                                        news = convert_pillar_score_to_display(h.get('news_score', 0))
                                        decision = h.get('decision', '-')

                                        score_color = 'text-green-400' if total >= 55 else 'text-yellow-400' if total >= 40 else 'text-red-400'
                                        decision_color = 'text-green-400' if decision in ['buy', 'strong_buy'] else 'text-red-400' if decision in ['sell', 'strong_sell'] else 'text-slate-300'

                                        with ui.row().classes('w-full gap-2 py-1 border-b border-slate-700'):
                                            ui.label(ts[5:]).classes('text-slate-300 text-xs w-24')  # MM-DD HH:MM
                                            ui.label(f'{total:.0f}').classes(f'{score_color} text-xs font-bold w-12')
                                            ui.label(f'{tech:.0f}').classes('text-blue-300 text-xs w-10')
                                            ui.label(f'{fund:.0f}').classes('text-purple-300 text-xs w-10')
                                            ui.label(f'{sent:.0f}').classes('text-yellow-300 text-xs w-10')
                                            ui.label(f'{news:.0f}').classes('text-orange-300 text-xs w-10')
                                            ui.label(decision or '-').classes(f'{decision_color} text-xs flex-1')

                                    # Compare with previous if possible
                                    comparison = compare_symbol_analyses(symbol)
                                    if comparison:
                                        with ui.card().classes('w-full bg-slate-700 p-3 mt-4'):
                                            ui.label('🔄 Comparaison vs Analyse Précédente').classes('text-cyan-400 font-bold text-sm mb-2')

                                            deltas = [
                                                ('Total', comparison.get('total_delta', 0)),
                                                ('Technical', comparison.get('technical_delta', 0)),
                                                ('Fundamental', comparison.get('fundamental_delta', 0)),
                                                ('Sentiment', comparison.get('sentiment_delta', 0)),
                                                ('News', comparison.get('news_delta', 0)),
                                            ]

                                            with ui.row().classes('gap-4 flex-wrap'):
                                                for name, delta in deltas:
                                                    delta_color = 'text-green-400' if delta > 0 else 'text-red-400' if delta < 0 else 'text-slate-400'
                                                    delta_icon = '▲' if delta > 0 else '▼' if delta < 0 else '—'
                                                    ui.label(f'{name}: {delta_icon} {abs(delta):.1f}').classes(f'{delta_color} text-xs')

                                            if comparison.get('decision_changed'):
                                                ui.label(
                                                    f"⚠️ Décision changée: {comparison.get('previous_decision')} → {comparison.get('current_decision')}"
                                                ).classes('text-yellow-400 text-xs mt-2')

                # Footer
                ui.label(f'Dernière mise à jour: {last_update}').classes('text-slate-500 text-xs p-2 text-right')

        def update_display():
            content_container.clear()
            filter_val = status_filter.value
            search_val = search_input.value.upper() if search_input.value else ''

            filter_map = {'Tous': None, 'Signaux': 'signal', 'Rejetés': 'reject'}
            target_status = filter_map.get(filter_val)

            journeys = get_all_journeys_sorted()

            if target_status:
                journeys = [j for j in journeys if j.get('current_status') == target_status]
            if search_val:
                journeys = [j for j in journeys if search_val in j.get('symbol', '')]

            with content_container:
                if not journeys:
                    with ui.card().classes('w-full glass-card p-8 text-center'):
                        ui.icon('hourglass_empty', size='xl').classes('text-slate-500 mb-4')
                        ui.label('Aucun symbole analysé.').classes('text-slate-400 text-lg')
                        ui.label('Lancez le bot pour commencer à voir les arbres de décision.').classes('text-slate-500 text-sm')
                else:
                    for journey in journeys[:20]:
                        render_symbol_card(journey)

        status_filter.on_value_change(lambda: update_display())
        search_input.on_value_change(lambda: update_display())
        # V4.8: REMOVED auto-refresh timer - it was resetting tab selection to default
        # Users can manually refresh via filter change or page reload if needed
        # ui.timer(20, update_display)  # DISABLED - causes tab reset
        update_display()


@ui.page('/capital')
def capital_page():
    """Capital & PnL page - Shows capital evolution and trading timeline"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('trending_up', size='lg').classes('text-green-400')
            ui.label('Capital & PnL').classes('text-xl font-bold text-white')
            create_help_button('Sécurités (Guardrails)', get_guardrails_explanation())
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6 gap-4'):
        # Current Stats
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('flex-1 glass-card p-5'):
                ui.label('Capital Actuel').classes('text-sm text-slate-400')
                capital_display = ui.label(f'{state.capital:,.0f} EUR').classes('text-3xl font-bold text-green-400')

            with ui.card().classes('flex-1 glass-card p-5'):
                ui.label('P&L Journalier').classes('text-sm text-slate-400')
                pnl_display = ui.label(f'{state.daily_pnl:+.2f} EUR').classes('text-3xl font-bold text-green-400')

            with ui.card().classes('flex-1 glass-card p-5'):
                ui.label('Actions Aujourd\'hui').classes('text-sm text-slate-400')
                today_actions = len([a for a in state.actions_timeline if a['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))])
                ui.label(str(today_actions)).classes('text-3xl font-bold text-blue-400')

        def update_stats():
            capital_display.text = f'{state.capital:,.0f} EUR'
            color = 'text-green-400' if state.daily_pnl >= 0 else 'text-red-400'
            pnl_display.text = f'{state.daily_pnl:+.2f} EUR'
            pnl_display.classes(color, remove='text-green-400 text-red-400')

        ui.timer(5, update_stats)

        # Capital Evolution Chart (using Plotly via echart)
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('📈 Évolution du Capital').classes('text-lg font-bold text-white mb-4')

            # If no capital history, show message
            if not state.capital_history:
                # Initialize with current capital
                add_capital_event("start", state.capital, details="Session démarrée")

            chart_container = ui.column().classes('w-full h-64')

            def update_capital_chart():
                chart_container.clear()
                with chart_container:
                    if len(state.capital_history) < 2:
                        ui.label('Pas assez de données pour afficher le graphique. Les points seront ajoutés au fil des actions.').classes('text-slate-400 italic')
                    else:
                        # Build chart data
                        timestamps = [h['timestamp'][11:19] for h in state.capital_history[-50:]]  # Last 50 points, time only
                        capitals = [h['capital'] for h in state.capital_history[-50:]]

                        # Simple line chart using ui.chart (ECharts)
                        ui.echart({
                            'backgroundColor': 'transparent',
                            'xAxis': {'type': 'category', 'data': timestamps, 'axisLabel': {'color': '#94a3b8'}},
                            'yAxis': {'type': 'value', 'axisLabel': {'color': '#94a3b8'}, 'splitLine': {'lineStyle': {'color': '#334155'}}},
                            'series': [{'data': capitals, 'type': 'line', 'smooth': True, 'areaStyle': {'opacity': 0.3}, 'lineStyle': {'color': '#22c55e'}, 'itemStyle': {'color': '#22c55e'}}],
                            'tooltip': {'trigger': 'axis'},
                            'grid': {'left': '10%', 'right': '5%', 'top': '10%', 'bottom': '15%'}
                        }).classes('w-full h-full')

            ui.timer(30, update_capital_chart)
            update_capital_chart()

        # Actions Timeline
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('🕐 Timeline des Actions').classes('text-lg font-bold text-white mb-4')
            ui.label('Chaque action du bot est tracée ici avec son contexte.').classes('text-slate-400 text-sm mb-4')

            timeline_container = ui.column().classes('gap-3 max-h-[500px] overflow-auto')

            def update_timeline():
                timeline_container.clear()
                with timeline_container:
                    if not state.actions_timeline:
                        ui.label('Aucune action enregistrée. Démarrez le bot pour voir la timeline.').classes('text-slate-400 italic')
                    else:
                        action_icons = {
                            'discovery': ('🔍', 'bg-blue-900', 'text-blue-300'),
                            'watchlist_add': ('📋', 'bg-purple-900', 'text-purple-300'),
                            'analyze': ('📊', 'bg-yellow-900', 'text-yellow-300'),
                            'signal': ('✅', 'bg-green-900', 'text-green-300'),
                            'buy': ('💰', 'bg-green-800', 'text-green-200'),
                            'sell': ('💸', 'bg-orange-900', 'text-orange-300'),
                            'reject': ('❌', 'bg-red-900', 'text-red-300'),
                        }

                        # Show most recent first
                        for action in reversed(state.actions_timeline[-50:]):
                            icon, bg, color = action_icons.get(action['action'], ('•', 'bg-slate-700', 'text-slate-300'))

                            with ui.card().classes(f'w-full p-3 {bg} border border-slate-600'):
                                with ui.row().classes('gap-4 items-start'):
                                    ui.label(icon).classes('text-2xl')
                                    with ui.column().classes('flex-1 gap-1'):
                                        with ui.row().classes('justify-between items-center'):
                                            action_name = action['action'].upper().replace('_', ' ')
                                            ui.label(action_name).classes(f'font-bold {color}')
                                            # Parse timestamp
                                            ts = action['timestamp']
                                            time_str = ts[11:19] if len(ts) > 19 else ts
                                            ui.label(time_str).classes('text-xs text-slate-500')

                                        with ui.row().classes('gap-4'):
                                            if action.get('symbol'):
                                                ui.chip(action['symbol'], icon='analytics').classes('bg-slate-600 text-white text-xs')
                                            if action.get('price'):
                                                ui.label(f"${action['price']:.2f}").classes('text-sm text-slate-300')

                                        if action.get('reason'):
                                            ui.label(action['reason']).classes('text-sm text-slate-400')

            ui.timer(10, update_timeline)
            update_timeline()

        # Capital History Events
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('💼 Historique Capital').classes('text-lg font-bold text-white mb-4')

            history_container = ui.column().classes('gap-2 max-h-64 overflow-auto')

            def update_history():
                history_container.clear()
                with history_container:
                    if not state.capital_history:
                        ui.label('Aucun événement capital enregistré.').classes('text-slate-400 italic')
                    else:
                        for event in reversed(state.capital_history[-20:]):
                            with ui.row().classes('gap-4 items-center p-2 bg-slate-700 rounded'):
                                event_type = event['event']
                                color = 'text-green-400' if event.get('amount', 0) >= 0 else 'text-red-400'

                                ui.label(event['timestamp'][11:19]).classes('text-xs text-slate-500 w-16')
                                ui.label(event_type.upper()).classes('text-xs text-slate-300 w-24')
                                ui.label(f"{event['capital']:,.0f} EUR").classes(f'font-bold {color}')
                                if event.get('symbol'):
                                    ui.label(event['symbol']).classes('text-sm text-blue-400')
                                if event.get('details'):
                                    ui.label(event['details']).classes('text-xs text-slate-400')

            ui.timer(15, update_history)
            update_history()


@ui.page('/reports')
def reports_page():
    """Reports page - Generate and view markdown reports"""

    ui.dark_mode(True)
    apply_modern_styles()

    # Header
    with ui.header().classes('header-gradient'):
        with ui.row().classes('w-full items-center px-4'):
            ui.icon('description', size='lg').classes('text-green-400')
            ui.label('Rapports').classes('text-xl font-bold text-white')
            ui.space()
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat').classes('text-white modern-btn')

    with ui.column().classes('w-full p-6 gap-4'):
        # Generate report button
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('📝 Générer un Rapport').classes('text-lg font-bold text-white mb-2')
            ui.label(
                "Génère un rapport Markdown (.md) avec les résultats du dernier scan, "
                "le raisonnement du bot, et les alertes détectées."
            ).classes('text-slate-400 text-sm mb-4')

            report_status = ui.label('').classes('text-sm mb-2')

            def on_generate_report():
                report_status.text = "⏳ Génération en cours..."
                filepath = generate_report()
                if filepath:
                    report_status.text = f"✅ Rapport généré: {filepath}"
                    report_status.classes('text-green-400', remove='text-red-400')
                else:
                    report_status.text = "❌ Erreur lors de la génération"
                    report_status.classes('text-red-400', remove='text-green-400')

            ui.button('Générer Rapport', icon='create', on_click=on_generate_report).classes('bg-green-600')

        # List existing reports
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('📚 Rapports Existants').classes('text-lg font-bold text-white mb-4')

            reports_container = ui.column().classes('gap-2')

            def list_reports():
                reports_container.clear()
                reports_dir = Path('data/reports')

                with reports_container:
                    if not reports_dir.exists():
                        ui.label('Aucun rapport. Générez-en un!').classes('text-slate-400 italic')
                        return

                    report_files = sorted(reports_dir.glob('*.md'), reverse=True)[:10]

                    if not report_files:
                        ui.label('Aucun rapport. Générez-en un!').classes('text-slate-400 italic')
                        return

                    for report_file in report_files:
                        with ui.row().classes('w-full justify-between items-center p-2 bg-slate-700 rounded'):
                            ui.label(report_file.name).classes('text-white font-mono text-sm')
                            size_kb = report_file.stat().st_size / 1024
                            ui.label(f'{size_kb:.1f} KB').classes('text-slate-400 text-xs')

            list_reports()
            ui.button('Rafraîchir', icon='refresh', on_click=list_reports).props('flat').classes('text-blue-400 mt-2')

        # View latest report
        with ui.card().classes('w-full glass-card p-6'):
            ui.label('📄 Dernier Rapport').classes('text-lg font-bold text-white mb-4')

            report_content = ui.markdown('*Chargement...*').classes('text-white')

            def load_latest_report():
                latest = Path('data/reports/latest_report.md')
                if latest.exists():
                    content = latest.read_text(encoding='utf-8')
                    report_content.set_content(content[:5000])  # Limit to 5000 chars
                else:
                    report_content.set_content('*Aucun rapport disponible. Générez-en un!*')

            load_latest_report()
            ui.button('Recharger', icon='refresh', on_click=load_latest_report).props('flat').classes('text-blue-400')


# =============================================================================
# MAIN
# =============================================================================

if __name__ in {"__main__", "__mp_main__"}:
    add_log("Dashboard TradingBot V4.1 demarre")

    # V4.5: Load persisted journeys at startup
    try:
        loaded = load_persisted_journeys()
        if loaded:
            add_log(f"V4.5: {loaded} journeys restaures depuis le stockage persistant")
    except Exception as e:
        add_log(f"Erreur chargement journeys: {e}")

    ui.run(
        title='TradingBot V4.1',
        port=8080,
        reload=False,
        show=True
    )
