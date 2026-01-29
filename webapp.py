"""
TradingBot V5 - Dashboard Professionnel avec 4 Piliers
Interface web responsive avec Chain of Thought detaillee et visualisation du flux de decision.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta, time
from typing import Optional, Dict, List, Literal, Any
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))

from nicegui import ui, app
from dotenv import load_dotenv

load_dotenv(override=True)

# =============================================================================
# STYLES - Professional Dark Theme with Visual Richness
# =============================================================================

STYLES = """
<style>
/* ===== FINARY DESIGN SYSTEM ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* Finary Color Palette - Dark Mode */
    --bg-primary: #0A0F1C;
    --bg-secondary: #111827;
    --bg-surface: #111827;
    --bg-card: #1e293b;
    --bg-elevated: #1F2937;
    --bg-card-hover: #334155;

    /* Text Colors */
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --text-muted: #6B7280;

    /* Accent Colors */
    --profit: #10B981;
    --profit-glow: rgba(16, 185, 129, 0.3);
    --loss: #EF4444;
    --loss-glow: rgba(239, 68, 68, 0.3);
    --accent: #8B5CF6;
    --accent-secondary: #6366f1;
    --accent-glow: rgba(139, 92, 246, 0.3);
    --warning: #F59E0B;
    --info: #3B82F6;
    --cyan: #22D3EE;
    --pink: #EC4899;

    /* Borders */
    --border: #374151;
    --border-subtle: #374151;
    --border-accent: #4B5563;
    --border-light: rgba(148, 163, 184, 0.2);

    /* Radius */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-full: 9999px;

    /* Fonts */
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

body {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
    background-attachment: fixed !important;
    color: var(--text-primary);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
}

/* Cards with Glass Effect */
.card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.card:hover {
    border-color: var(--accent);
    box-shadow: 0 8px 30px var(--accent-glow);
    transform: translateY(-2px);
}

.card-flat {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

/* Stat Cards */
.stat-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent-secondary));
}

.stat-card.profit::before { background: linear-gradient(90deg, var(--profit), #059669); }
.stat-card.loss::before { background: linear-gradient(90deg, var(--loss), #dc2626); }

.stat-value {
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1.2;
    background: linear-gradient(135deg, #fff 0%, #e2e8f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

/* Live Feed - Training */
.live-feed-signal {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.375rem 0.5rem;
    border-radius: 6px;
    transition: background 0.15s ease;
    border-left: 3px solid transparent;
}

.live-feed-signal:hover {
    background: rgba(100, 116, 139, 0.15);
}

.live-feed-signal.winner {
    border-left-color: var(--profit);
    background: rgba(16, 185, 129, 0.05);
}

.live-feed-signal.loser {
    border-left-color: var(--loss);
    background: rgba(239, 68, 68, 0.05);
}

.pillar-indicator {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
}

.pillar-indicator.tech { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
.pillar-indicator.fund { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
.pillar-indicator.sent { background: rgba(236, 72, 153, 0.2); color: #f472b6; }
.pillar-indicator.news { background: rgba(34, 211, 238, 0.2); color: #22d3ee; }

/* Colors */
.profit { color: var(--profit) !important; }
.loss { color: var(--loss) !important; }
.accent { color: var(--accent) !important; }
.muted { color: var(--text-muted) !important; }
.warning { color: var(--warning) !important; }
.info { color: var(--info) !important; }

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary)) !important;
    color: white !important;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    min-height: 44px;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 4px 15px var(--accent-glow);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px var(--accent-glow);
}

.btn-success {
    background: linear-gradient(135deg, var(--profit), #059669) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--profit-glow);
}

.btn-danger {
    background: linear-gradient(135deg, var(--loss), #dc2626) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--loss-glow);
}

/* Finary-style Buttons */
.btn-finary-primary {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.875rem;
    transition: all 0.2s ease;
    border: none;
}

.btn-finary-primary:hover {
    background: #7C3AED !important;
    transform: translateY(-1px);
}

.btn-finary-secondary {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.875rem;
    transition: all 0.2s ease;
}

.btn-finary-secondary:hover {
    border-color: var(--border-accent);
}

.btn-finary-success {
    background: var(--profit) !important;
    color: white !important;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
}

/* Navigation */
.nav-header {
    background: linear-gradient(180deg, rgba(10, 15, 26, 0.95) 0%, rgba(10, 15, 26, 0.8) 100%);
    border-bottom: 1px solid var(--border-light);
    backdrop-filter: blur(10px);
}

.nav-item {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    color: var(--text-secondary);
    transition: all 0.2s ease;
    cursor: pointer;
}

.nav-item:hover, .nav-item.active {
    background: rgba(139, 92, 246, 0.2);
    color: var(--text-primary);
}

/* Alert Cards */
.alert-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border-left: 4px solid var(--accent);
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.alert-card:hover {
    transform: translateX(4px);
}

.alert-card.buy { border-left-color: var(--profit); }
.alert-card.watch { border-left-color: var(--warning); }
.alert-card.reject { border-left-color: var(--loss); }

/* Status Indicator */
.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.5rem;
}

.status-dot.active {
    background: var(--profit);
    box-shadow: 0 0 10px var(--profit-glow);
    animation: pulse 2s infinite;
}
.status-dot.inactive {
    background: var(--loss);
    box-shadow: 0 0 10px var(--loss-glow);
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px var(--profit-glow); }
    50% { opacity: 0.6; box-shadow: 0 0 20px var(--profit-glow); }
}

/* Journey Flow */
.journey-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}

.journey-card:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 20px var(--accent-glow);
}

.flow-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px 14px;
    background: rgba(71, 85, 105, 0.4);
    border-radius: 10px;
    min-width: 85px;
    transition: all 0.2s ease;
}

.flow-step.active {
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
    box-shadow: 0 4px 15px var(--accent-glow);
}

.flow-step.passed {
    background: rgba(16, 185, 129, 0.25);
    border: 1px solid var(--profit);
}

.flow-step.failed {
    background: rgba(239, 68, 68, 0.25);
    border: 1px solid var(--loss);
}

.flow-step.clickable {
    cursor: pointer;
}

.flow-step.clickable:hover {
    transform: scale(1.05);
}

.flow-arrow {
    color: var(--accent);
    font-size: 20px;
    opacity: 0.7;
}

.flow-icon { font-size: 18px; }
.flow-label { font-size: 11px; color: var(--text-secondary); margin-top: 4px; }
.flow-value { font-size: 12px; color: white; font-weight: 600; margin-top: 2px; }

/* Pillar Badge */
.pillar-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}

.pillar-badge.tech { background: rgba(59, 130, 246, 0.3); color: #60a5fa; }
.pillar-badge.fund { background: rgba(139, 92, 246, 0.3); color: #a78bfa; }
.pillar-badge.sent { background: rgba(236, 72, 153, 0.3); color: #f472b6; }
.pillar-badge.news { background: rgba(34, 211, 238, 0.3); color: #22d3ee; }

/* Reasoning Box */
.reasoning-box {
    background: rgba(0, 0, 0, 0.3);
    border-left: 3px solid var(--accent);
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    margin-top: 12px;
}

.reasoning-title {
    font-size: 12px;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 6px;
}

.reasoning-text {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Chain of Thought */
.thought-item {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid var(--accent);
    transition: all 0.2s ease;
}

.thought-item:hover {
    background: rgba(30, 41, 59, 0.8);
}

.thought-item.discovery { border-left-color: var(--info); }
.thought-item.analysis { border-left-color: var(--accent); }
.thought-item.signal { border-left-color: var(--profit); }
.thought-item.error { border-left-color: var(--loss); }
.thought-item.warning { border-left-color: var(--warning); }

.thought-time {
    font-size: 10px;
    color: var(--text-muted);
    font-family: monospace;
}

.thought-category {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(139, 92, 246, 0.2);
    color: var(--accent);
    text-transform: uppercase;
    font-weight: 600;
}

/* Learning Stats */
.learning-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px;
    border-radius: 8px;
    transition: background 0.2s ease;
}

.learning-step:hover {
    background: rgba(139, 92, 246, 0.1);
}

.learning-check {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
}

.learning-check.done {
    background: var(--profit);
    color: white;
}

.learning-check.pending {
    background: rgba(100, 116, 139, 0.3);
    color: var(--text-muted);
    border: 2px dashed var(--text-muted);
}

/* Progress Bar */
.progress-bar {
    height: 6px;
    background: rgba(100, 116, 139, 0.3);
    border-radius: 3px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--profit));
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* Table */
.q-table {
    background: transparent !important;
}

.q-table th {
    color: var(--text-secondary) !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.05em;
}

.q-table td {
    color: var(--text-primary) !important;
}

/* ===== QUASAR FORM COMPONENTS (Dark Mode) ===== */

/* Select Dropdown - Menu déroulant */
.q-select {
    background: var(--bg-elevated) !important;
    border-radius: 10px !important;
}
.q-select .q-field__control {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}
.q-select .q-field__native,
.q-select .q-field__input {
    color: var(--text-primary) !important;
}
.q-select .q-field__label {
    color: var(--text-secondary) !important;
}
.q-select .q-field__marginal {
    color: var(--text-secondary) !important;
}
/* Menu dropdown options */
.q-menu {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.q-item {
    color: var(--text-primary) !important;
}
.q-item:hover {
    background: var(--bg-elevated) !important;
}
.q-item--active {
    background: rgba(139, 92, 246, 0.2) !important;
    color: var(--accent) !important;
}

/* Slider - Curseur */
.q-slider {
    color: var(--accent) !important;
}
.q-slider__track-container {
    background: var(--bg-elevated) !important;
}
.q-slider__track {
    background: var(--accent) !important;
}
.q-slider__thumb {
    background: var(--accent) !important;
    border: 2px solid white !important;
}
.q-slider__focus-ring {
    background: var(--accent-glow) !important;
}

/* Checkbox - Case à cocher */
.q-checkbox {
    color: var(--text-primary) !important;
}
.q-checkbox__inner {
    color: var(--accent) !important;
}
.q-checkbox__bg {
    border-color: var(--border-accent) !important;
}
.q-checkbox__inner--truthy .q-checkbox__bg {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
.q-checkbox__label {
    color: var(--text-primary) !important;
}

/* Expansion Panel - Panneau déployable */
.q-expansion-item {
    background: var(--bg-elevated) !important;
    border-radius: 10px !important;
    margin-top: 8px;
}
.q-expansion-item__container {
    border-radius: 10px !important;
}
.q-expansion-item .q-item {
    background: transparent !important;
}
.q-expansion-item .q-item__label {
    color: var(--text-primary) !important;
    font-weight: 500;
}
.q-expansion-item .q-item__section--side {
    color: var(--text-secondary) !important;
}
.q-expansion-item__content {
    background: rgba(0, 0, 0, 0.2) !important;
    padding: 16px !important;
}

/* Input - Champ de saisie */
.q-field--outlined .q-field__control {
    background: var(--bg-elevated) !important;
}
.q-field--outlined .q-field__control:before {
    border-color: var(--border) !important;
}
.q-field--outlined .q-field__control:hover:before {
    border-color: var(--accent) !important;
}
.q-field--outlined.q-field--focused .q-field__control:after {
    border-color: var(--accent) !important;
}
.q-field__native,
.q-field__input {
    color: var(--text-primary) !important;
}
.q-field__label {
    color: var(--text-secondary) !important;
}

/* Number Input */
.q-field--filled .q-field__control {
    background: var(--bg-elevated) !important;
}

/* Linear Progress */
.q-linear-progress {
    background: var(--bg-elevated) !important;
    border-radius: 4px;
}
.q-linear-progress__track {
    background: var(--bg-elevated) !important;
}
.q-linear-progress__model {
    background: linear-gradient(90deg, var(--accent), var(--accent-secondary)) !important;
}

/* Responsive */
@media (max-width: 768px) {
    .stat-value { font-size: 1.75rem; }
    .hide-mobile { display: none !important; }
    .full-mobile { width: 100% !important; }
    .flow-step { min-width: 65px; padding: 8px 10px; }
    .flow-icon { font-size: 16px; }
    .flow-label { font-size: 9px; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* Section Title */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}

.section-title-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.section-title-icon.purple { background: rgba(139, 92, 246, 0.2); }
.section-title-icon.green { background: rgba(16, 185, 129, 0.2); }
.section-title-icon.blue { background: rgba(59, 130, 246, 0.2); }
.section-title-icon.cyan { background: rgba(34, 211, 238, 0.2); }

/* ===== FINARY COMPONENTS ===== */

/* Score Ring - Arc de cercle élégant */
.score-ring {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.score-ring svg {
    transform: rotate(-90deg);
}

.score-ring-bg {
    fill: none;
    stroke: var(--bg-elevated);
    stroke-width: 8;
}

.score-ring-progress {
    fill: none;
    stroke-width: 8;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.8s ease-out;
}

.score-ring-progress.buy { stroke: var(--profit); }
.score-ring-progress.watch { stroke: var(--warning); }
.score-ring-progress.reject { stroke: var(--loss); }

.score-ring-value {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}

.score-ring-number {
    font-family: var(--font-mono);
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text-primary);
}

.score-ring-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Pillar Cards Grid */
.pillar-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

@media (max-width: 640px) {
    .pillar-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.pillar-card-mini {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px;
    text-align: center;
    transition: all 0.2s ease;
}

.pillar-card-mini:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
}

.pillar-card-mini .pillar-title {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.pillar-card-mini .pillar-score {
    font-family: var(--font-mono);
    font-size: 1.25rem;
    font-weight: 700;
}

.pillar-card-mini .pillar-max {
    font-size: 0.75rem;
    color: var(--text-muted);
}

.pillar-bar-mini {
    height: 4px;
    background: var(--bg-elevated);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
}

.pillar-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease-out;
}

.pillar-tech .pillar-bar-fill { background: var(--info); }
.pillar-fund .pillar-bar-fill { background: var(--accent); }
.pillar-sent .pillar-bar-fill { background: var(--pink); }
.pillar-news .pillar-bar-fill { background: var(--cyan); }

/* Signal Badges - Style Finary pill */
.signal-badge-finary {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.signal-badge-finary.buy {
    background: rgba(16, 185, 129, 0.15);
    color: var(--profit);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.signal-badge-finary.watch {
    background: rgba(245, 158, 11, 0.15);
    color: var(--warning);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.signal-badge-finary.reject {
    background: rgba(239, 68, 68, 0.15);
    color: var(--loss);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.signal-badge-finary.pending {
    background: rgba(100, 116, 139, 0.15);
    color: var(--text-muted);
    border: 1px solid rgba(100, 116, 139, 0.3);
}

/* Stock Card Finary */
.stock-card-finary {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stock-card-finary:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.stock-card-finary.glow-green {
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
}

.stock-card-finary.glow-yellow {
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.1);
    border-color: rgba(245, 158, 11, 0.3);
}

.stock-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
}

.stock-card-symbol {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}

.stock-card-status {
    font-size: 0.7rem;
    font-weight: 500;
}

.stock-card-pillars {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}

.stock-pillar-mini {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

.stock-pillar-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
}

.stock-pillar-bar-bg {
    width: 100%;
    height: 4px;
    background: var(--bg-elevated);
    border-radius: 2px;
    overflow: hidden;
}

.stock-pillar-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}

.stock-pillar-value {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
}

.stock-card-signal {
    display: flex;
    justify-content: center;
    margin: 12px 0;
}

.stock-card-key-factor {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-align: center;
    padding: 8px 0;
    border-top: 1px solid var(--border-subtle);
}

.stock-card-hint {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 8px;
}

/* Loading Overlay Inline */
.loading-overlay-inline {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
}

/* Threshold Legend */
.threshold-legend {
    display: flex;
    justify-content: center;
    gap: 24px;
    padding: 12px 20px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: 16px;
}

.threshold-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
}

.threshold-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.threshold-dot.buy { background: var(--profit); }
.threshold-dot.watch { background: var(--warning); }
.threshold-dot.reject { background: var(--loss); }

/* Buy Explanation Card */
.buy-explanation {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: var(--radius-lg);
    padding: 20px;
    margin-top: 16px;
}

.buy-explanation-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--profit);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.buy-factor {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(16, 185, 129, 0.1);
}

.buy-factor:last-child {
    border-bottom: none;
}

.buy-factor-check {
    color: var(--profit);
    font-weight: bold;
}

.buy-factor-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* Risk Section */
.risk-section {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-top: 16px;
}

.risk-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--loss);
    margin-bottom: 10px;
}

.risk-item {
    font-size: 0.8rem;
    color: var(--text-secondary);
    padding: 4px 0;
}

/* Position Suggestion */
.position-suggestion {
    background: var(--bg-elevated);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-top: 16px;
}

.position-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    text-align: center;
}

.position-item label {
    font-size: 0.65rem;
    text-transform: uppercase;
    color: var(--text-muted);
    display: block;
    margin-bottom: 4px;
}

.position-item .value {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 600;
}

.position-item .value.profit { color: var(--profit); }
.position-item .value.loss { color: var(--loss); }

/* Discovery Timeline */
.discovery-timeline {
    position: relative;
    padding-left: 28px;
    margin-top: 16px;
}

.timeline-line {
    position: absolute;
    left: 8px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: linear-gradient(180deg, var(--accent), var(--info));
    border-radius: 1px;
}

.timeline-item {
    position: relative;
    padding-bottom: 20px;
}

.timeline-item:last-child {
    padding-bottom: 0;
}

.timeline-dot {
    position: absolute;
    left: -24px;
    top: 4px;
    width: 12px;
    height: 12px;
    background: var(--accent);
    border-radius: 50%;
    border: 2px solid var(--bg-primary);
}

.timeline-time {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
}

.timeline-title {
    font-weight: 600;
    font-size: 0.85rem;
    margin: 2px 0;
}

.timeline-detail {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* Market Status Bar */
.market-status-bar {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 10px 20px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
}

.market-status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
}

.market-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}

.market-dot.open { background: var(--profit); }
.market-dot.closed { background: var(--loss); }
.market-dot.premarket { background: var(--warning); }

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Loading Overlay */
.loading-overlay-finary {
    position: absolute;
    inset: 0;
    background: rgba(10, 15, 26, 0.9);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: inherit;
    gap: 16px;
    z-index: 50;
}

.loading-spinner-finary {
    width: 48px;
    height: 48px;
    border: 3px solid var(--bg-elevated);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin-finary 1s linear infinite;
}

@keyframes spin-finary {
    to { transform: rotate(360deg); }
}

/* Mono text class */
.text-mono {
    font-family: var(--font-mono);
}

/* Portfolio Page Styles */
.portfolio-stat-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    transition: all 0.2s ease;
}

.portfolio-stat-card:hover {
    border-color: var(--border-accent);
}

.portfolio-stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.portfolio-stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 4px 0;
}

.portfolio-stat-change {
    font-size: 0.85rem;
    font-weight: 500;
}

.portfolio-stat-subtitle {
    font-size: 0.7rem;
    color: var(--text-muted);
}

/* Trade History Item */
.trade-history-item {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    transition: all 0.2s ease;
}

.trade-history-item:hover {
    border-color: var(--border-accent);
}

/* Mini Stats */
.stat-mini {
    background: var(--bg-elevated);
    border-radius: var(--radius-md);
    padding: 12px;
}

.stat-mini-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-mini-value {
    font-family: var(--font-mono);
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 4px;
}

/* Learning Page Styles */
.learning-metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
    text-align: center;
}

.learning-metric-value {
    font-family: var(--font-mono);
    font-size: 2.5rem;
    font-weight: 700;
    margin: 8px 0;
}

.learning-metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
}

.weight-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.weight-bar-label {
    width: 100px;
    font-size: 0.85rem;
}

.weight-bar-track {
    flex: 1;
    height: 8px;
    background: var(--bg-elevated);
    border-radius: 4px;
    overflow: hidden;
}

.weight-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.lesson-item {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: var(--bg-elevated);
    border-radius: var(--radius-md);
    margin-bottom: 8px;
}

.lesson-date {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
}

.lesson-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.missed-opportunity {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: var(--radius-md);
    margin-bottom: 8px;
}
</style>
"""

# =============================================================================
# NOTIFICATION HELPER
# =============================================================================

NotifyType = Literal['positive', 'negative', 'warning', 'info', 'ongoing']

def safe_notify(message: str, type: NotifyType = 'info', **kwargs):
    """Safe notification that works in background tasks."""
    try:
        ui.notify(message, type=type, **kwargs)
    except Exception:
        level = {'positive': 'info', 'negative': 'error', 'warning': 'warning'}.get(type, 'info')
        getattr(logger, level)(f"[Notification] {message}")

# =============================================================================
# IMPORTS - Backend modules
# =============================================================================

try:
    from src.intelligence.analysis_store import get_analysis_store, AnalysisStore
    from src.intelligence.narrative_generator import NarrativeGenerator, get_narrative_generator
    from src.agents.reasoning_graph import ReasoningGraph, get_graph_store, NodeType
except ImportError as e:
    logger.warning(f"Optional import failed: {e}")

# =============================================================================
# APPLICATION STATE
# =============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.bot_running = False
        self.bot_status = "Arrete"
        self.bot_mode = "OFF"
        self.last_scan = None
        self.last_discovery = None
        self.alerts: List[Dict] = []
        self.trades: List[Dict] = []
        self.capital = float(os.getenv('TRADING_CAPITAL', os.getenv('CAPITAL', 1500)))
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.logs: List[str] = []
        self.agent = None
        self._task = None
        self.watchlist: List[Dict] = []
        self.focus_symbols: List[str] = []
        self.symbol_journeys: Dict[str, Dict] = {}
        self.narrative_reports: Dict[str, str] = {}
        self.capital_history: List[Dict] = []
        self.chain_of_thought: List[Dict] = []
        self.learning_stats: Dict[str, Any] = {
            'total_vectors': 0,
            'total_closed': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0
        }
        self.dialog_open = False  # Flag to prevent refresh when dialog is open

state = AppState()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def add_log(message: str):
    """Add log entry"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.logs.insert(0, f"[{timestamp}] {message}")
    state.logs = state.logs[:100]
    logger.info(message)

def add_thought(content: str, category: str = "analysis", details: str = ""):
    """Add thought to chain of thought with rich details"""
    state.chain_of_thought.insert(0, {
        "timestamp": datetime.now().isoformat(),
        "time_display": datetime.now().strftime("%H:%M:%S"),
        "content": content,
        "category": category,
        "details": details
    })
    state.chain_of_thought = state.chain_of_thought[:50]

def get_current_market_mode() -> str:
    """Get current market mode based on time"""
    now_et = datetime.now(ET)
    current_time = now_et.time()
    weekday = now_et.weekday()

    if weekday >= 5:
        return "WEEKEND"
    elif current_time < time(9, 0):
        return "PRE_MARKET"
    elif current_time < MARKET_OPEN:
        return "PRE_MARKET"
    elif current_time < MARKET_CLOSE:
        return "MARKET"
    else:
        return "AFTER_HOURS"

def format_pnl(value: float) -> str:
    """Format P&L with sign"""
    if value >= 0:
        return f"+{value:.2f}"
    return f"{value:.2f}"

def get_pnl_class(value: float) -> str:
    """Get CSS class for P&L value"""
    return "profit" if value >= 0 else "loss"

def convert_pillar_score_to_display(raw_score: float) -> float:
    """Convert raw pillar score (0-100) to display format (0-25)"""
    return round(raw_score / 4, 1)

def update_learning_stats():
    """Update learning statistics from graph store AND shadow tracker"""
    try:
        # Graph store stats (adaptive learning vectors)
        graph_store = get_graph_store()
        vectors = graph_store.get_all_vectors_with_outcomes()

        total = len(vectors)
        profits = [v for v in vectors if v[1] == 'profit']
        losses = [v for v in vectors if v[1] == 'loss']
        closed = len(profits) + len(losses)

        state.learning_stats = {
            'total_vectors': total,
            'total_closed': closed,
            'wins': len(profits),
            'losses': len(losses),
            'win_rate': (len(profits) / closed * 100) if closed > 0 else 0.0,
            # Shadow tracker stats (will be updated below)
            'shadow_total': 0,
            'shadow_verified': 0,
            'shadow_winners': 0,
            'shadow_losers': 0,
            'shadow_pending': 0,
            'shadow_win_rate': 0.0,
            'shadow_profit_factor': 0.0,
            'shadow_weights': {'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25}
        }

        # Shadow tracker stats (autonomous paper trading)
        try:
            from src.intelligence.shadow_tracker import get_shadow_tracker
            import asyncio

            async def get_shadow_stats():
                tracker = await get_shadow_tracker()
                return tracker.get_statistics()

            # Run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            shadow_stats = loop.run_until_complete(get_shadow_stats())
            loop.close()

            state.learning_stats.update({
                'shadow_total': shadow_stats.get('total_signals', 0),
                'shadow_verified': shadow_stats.get('verified', 0),
                'shadow_winners': shadow_stats.get('winners', 0),
                'shadow_losers': shadow_stats.get('losers', 0),
                'shadow_pending': shadow_stats.get('pending', 0),
                'shadow_win_rate': shadow_stats.get('win_rate', 0.0),
                'shadow_profit_factor': shadow_stats.get('profit_factor', 0.0),
                'shadow_weights': shadow_stats.get('current_weights', {}),
                'shadow_avg_winner': shadow_stats.get('avg_winner_pnl', 0.0),
                'shadow_avg_loser': shadow_stats.get('avg_loser_pnl', 0.0)
            })
        except Exception as shadow_e:
            logger.debug(f"Shadow tracker stats not available: {shadow_e}")

    except Exception as e:
        logger.debug(f"Could not update learning stats: {e}")

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
        add_log("Agent initialise avec succes")
        add_thought("Agent MarketAgent initialise", "startup",
                   f"Capital: {state.capital} EUR | IBKR Port: {config.ibkr_port}")
        return True
    except Exception as e:
        add_log(f"Erreur initialisation: {e}")
        add_thought(f"Erreur initialisation: {str(e)[:100]}", "error")
        return False

async def start_bot():
    """Start the trading bot"""
    if state.bot_running:
        return

    state.bot_status = "Demarrage..."
    add_log("Demarrage du bot...")
    add_thought("Demarrage du systeme de trading", "startup",
               "Initialisation de l'agent et des connexions API...")

    if state.agent is None:
        success = await initialize_agent()
        if not success:
            state.bot_status = "Erreur"
            safe_notify("Erreur initialisation", type='negative')
            return

    state.bot_running = True
    state.bot_status = "Actif"
    add_log("Bot demarre avec succes")

    # Run initial discovery
    try:
        add_thought("Phase Discovery en cours...", "discovery",
                   "Scan des sources sociales: Reddit, StockTwits, X/Twitter via Grok API")
        result = await state.agent.run_discovery_phase()
        state.last_discovery = datetime.now()
        watchlist = result.get('watchlist', [])
        if watchlist:
            state.focus_symbols = watchlist[:50]
            sources_str = ", ".join(result.get('sources', ['social', 'momentum']))
            add_thought(f"Discovery complete: {len(watchlist)} symboles identifies", "discovery",
                       f"Sources: {sources_str} | Top symbols: {', '.join(watchlist[:5])}")

            # Update journeys
            for symbol in watchlist[:20]:
                if symbol not in state.symbol_journeys:
                    state.symbol_journeys[symbol] = {
                        'current_status': 'discovered',
                        'current_score': 0,
                        'last_update': datetime.now().isoformat(),
                        'journey': [{'step': 'discovery', 'time': datetime.now().isoformat(),
                                    'data': {'sources': result.get('sources', [])}}]
                    }

            # Run immediate analysis after discovery (don't wait 15-30 min)
            add_thought("Lancement de l'analyse 4 Piliers...", "analysis",
                       f"Analyse de {len(state.focus_symbols)} symboles decouverts")
            try:
                analysis_result = await state.agent.run_trading_scan(execute_trades=False)
                state.last_scan = datetime.now()
                signals = analysis_result.get('signals_found', 0)
                analyzed = analysis_result.get('analyzed_symbols', [])
                scoring_details = analysis_result.get('scoring_details', {})

                # Update journeys with analysis results
                for sym in analyzed[:30]:
                    analysis_data = scoring_details.get(sym, {})
                    if analysis_data:
                        _update_journey(sym, analysis_data)

                add_thought(f"Analyse initiale terminee: {signals} signal(s) sur {len(analyzed)} symboles",
                           "signal" if signals > 0 else "analysis",
                           f"Symboles analyses avec succes: {', '.join(analyzed[:5])}...")
            except Exception as e2:
                add_thought(f"Erreur analyse initiale: {str(e2)[:80]}", "error")

    except Exception as e:
        add_thought(f"Erreur Discovery: {str(e)[:100]}", "error")

    # Start main loop
    state._task = asyncio.create_task(run_bot_loop())
    safe_notify("Bot demarre", type='positive')

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
    add_log("Bot arrete")
    add_thought("Systeme arrete", "startup")
    safe_notify("Bot arrete", type='info')

async def run_bot_loop():
    """Main bot loop"""
    add_log("Boucle principale demarree")

    while state.bot_running:
        try:
            mode = get_current_market_mode()
            state.bot_mode = mode

            # Scan interval based on mode
            interval = 900 if mode == "MARKET" else 1800  # 15 min or 30 min

            await asyncio.sleep(interval)

            if not state.bot_running:
                break

            # Run scan
            add_thought(f"Scan automatique ({mode})", "analysis",
                       f"Intervalle: {interval//60} min | Execute trades: {mode == 'MARKET'}")
            execute = (mode == "MARKET")
            result = await state.agent.run_trading_scan(execute_trades=execute)
            state.last_scan = datetime.now()

            # Process results
            signals = result.get('signals_found', 0)
            analyzed = result.get('analyzed_symbols', [])

            if signals > 0:
                add_thought(f"{signals} signal(s) de trading detecte(s)", "signal",
                           f"Symboles analyses: {len(analyzed)} | Actions: {result.get('actions_taken', 0)}")

            # Update journeys with analysis results (key is 'scoring_details' from orchestrator)
            scoring_details = result.get('scoring_details', {})
            for symbol in analyzed[:20]:
                analysis_data = scoring_details.get(symbol, {})
                if analysis_data:
                    _update_journey(symbol, analysis_data)

            # Update alerts
            if result.get('alerts'):
                for alert in result['alerts'][:10]:
                    state.alerts.insert(0, {
                        'time': datetime.now().isoformat(),
                        **alert
                    })
                state.alerts = state.alerts[:100]

        except asyncio.CancelledError:
            break
        except Exception as e:
            add_log(f"Erreur boucle: {e}")
            add_thought(f"Erreur dans la boucle principale: {str(e)[:80]}", "error")
            await asyncio.sleep(60)

    add_log("Boucle principale terminee")

def _update_journey(symbol: str, analysis_data: Dict):
    """Update symbol journey with analysis data"""
    if symbol not in state.symbol_journeys:
        state.symbol_journeys[symbol] = {
            'current_status': 'discovered',
            'current_score': 0,
            'last_update': datetime.now().isoformat(),
            'journey': []
        }

    journey = state.symbol_journeys[symbol]

    # Use total_score from orchestrator if available, otherwise calculate
    if 'total_score' in analysis_data:
        total = analysis_data['total_score']
    else:
        # Fallback: calculate from pillar scores (0-100 scale each)
        t = convert_pillar_score_to_display(analysis_data.get('technical', 0))
        f = convert_pillar_score_to_display(analysis_data.get('fundamental', 0))
        s = convert_pillar_score_to_display(analysis_data.get('sentiment', 0))
        n = convert_pillar_score_to_display(analysis_data.get('news', 0))
        total = t + f + s + n

    journey['current_score'] = round(total, 1)
    journey['current_status'] = 'analyzed'
    journey['last_update'] = datetime.now().isoformat()
    journey['journey'].append({
        'step': 'analysis',
        'time': datetime.now().isoformat(),
        'data': analysis_data,
        'reasoning': analysis_data.get('reasoning_summary', analysis_data.get('reasoning', ''))
    })

    # Update status based on score
    if total >= 55:
        journey['current_status'] = 'signal'
    elif total >= 40:
        journey['current_status'] = 'watch'

async def run_manual_scan():
    """Run a manual scan"""
    if state.agent is None:
        success = await initialize_agent()
        if not success:
            safe_notify("Agent non initialise", type='negative')
            return

    add_log("Scan manuel lance...")
    add_thought("Scan manuel declenche", "analysis",
               "Analyse complete des symboles de la watchlist...")
    safe_notify("Scan en cours...", type='info')

    try:
        result = await state.agent.run_trading_scan(execute_trades=False)
        state.last_scan = datetime.now()
        signals = result.get('signals_found', 0)
        analyzed = result.get('analyzed_symbols', [])

        # Update journeys (key is 'scoring_details' from orchestrator)
        scoring_details = result.get('scoring_details', {})
        for symbol in analyzed[:20]:
            analysis_data = scoring_details.get(symbol, {})
            if analysis_data:
                _update_journey(symbol, analysis_data)

        # Add alerts
        if result.get('alerts'):
            for alert in result['alerts'][:10]:
                state.alerts.insert(0, {
                    'time': datetime.now().isoformat(),
                    **alert
                })

        # Build top scores string
        top_scores_list = []
        for s in analyzed[:3]:
            score = state.symbol_journeys.get(s, {}).get('current_score', 0)
            top_scores_list.append(f"{s}({score})")
        top_scores_str = ', '.join(top_scores_list) if top_scores_list else 'N/A'
        add_thought(f"Scan termine: {signals} signal(s) sur {len(analyzed)} symboles",
                   "signal" if signals > 0 else "analysis",
                   f"Top scores: {top_scores_str}")
        safe_notify(f"{signals} signal(s) trouve(s)", type='positive')
    except Exception as e:
        add_log(f"Erreur scan: {e}")
        add_thought(f"Erreur scan manuel: {str(e)[:80]}", "error")
        safe_notify(f"Erreur: {e}", type='negative')

# =============================================================================
# REUSABLE COMPONENTS
# =============================================================================

def stat_card(title: str, value: str, subtitle: str = "", icon: str = "", value_class: str = "", card_class: str = ""):
    """Create a rich stat card component"""
    with ui.element('div').classes(f'stat-card {card_class}'):
        with ui.row().classes('items-center gap-4'):
            if icon:
                with ui.element('div').classes('w-12 h-12 rounded-xl bg-slate-700/50 flex items-center justify-center'):
                    ui.icon(icon).classes('text-2xl text-slate-300')
            with ui.column().classes('gap-0'):
                ui.label(value).classes(f'stat-value {value_class}')
                ui.label(title).classes('stat-label')
                if subtitle:
                    ui.label(subtitle).classes('text-xs text-slate-500 mt-1')

def alert_card(alert: Dict):
    """Create an alert card component"""
    signal = alert.get('signal', 'WATCH')
    signal_class = 'buy' if 'BUY' in signal else ('reject' if 'REJECT' in signal else 'watch')

    with ui.element('div').classes(f'alert-card {signal_class}'):
        with ui.row().classes('justify-between items-center'):
            with ui.column().classes('gap-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.label(alert.get('symbol', 'N/A')).classes('font-bold text-lg')
                    signal_icon = '✅' if 'STRONG' in signal else ('📈' if 'BUY' in signal else ('👀' if 'WATCH' in signal else '❌'))
                    ui.label(signal_icon)
                ui.label(signal).classes('text-sm ' + ('profit' if 'BUY' in signal else ('warning' if 'WATCH' in signal else 'loss')))
            with ui.column().classes('text-right gap-1'):
                score = alert.get('score', alert.get('confidence_score', 0))
                score_color = 'profit' if score >= 55 else ('warning' if score >= 40 else 'loss')
                ui.label(f"{score}/100").classes(f'font-bold text-xl {score_color}')
                time_str = alert.get('time', '')[:10]
                ui.label(time_str).classes('text-xs muted')


# =============================================================================
# FINARY-STYLE COMPONENTS
# =============================================================================

def score_ring(score: float, size: int = 100):
    """Create a Finary-style score ring with animated arc"""
    # Determine color based on score
    if score >= 55:
        color_class = 'buy'
        signal = 'ACHETER'
    elif score >= 40:
        color_class = 'watch'
        signal = 'SUIVRE'
    else:
        color_class = 'reject'
        signal = 'PASSER'

    # SVG calculations
    radius = 42
    circumference = 2 * 3.14159 * radius
    progress = (score / 100) * circumference
    offset = circumference - progress

    with ui.element('div').classes('score-ring').style(f'width: {size}px; height: {size}px'):
        ui.html(f'''
            <svg width="{size}" height="{size}" viewBox="0 0 100 100">
                <circle class="score-ring-bg" cx="50" cy="50" r="{radius}"/>
                <circle class="score-ring-progress {color_class}" cx="50" cy="50" r="{radius}"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"/>
            </svg>
        ''', sanitize=False)
        with ui.element('div').classes('score-ring-value'):
            ui.label(f'{int(score)}').classes('score-ring-number')
            ui.label('/100').classes('score-ring-label')


def signal_badge(score: float):
    """Create a Finary-style signal badge"""
    if score >= 55:
        badge_class = 'buy'
        icon = '●'
        text = 'ACHETER'
    elif score >= 40:
        badge_class = 'watch'
        icon = '○'
        text = 'SUIVRE'
    elif score > 0:
        badge_class = 'reject'
        icon = '✕'
        text = 'PASSER'
    else:
        badge_class = 'pending'
        icon = '◌'
        text = 'EN ATTENTE'

    with ui.element('span').classes(f'signal-badge-finary {badge_class}'):
        ui.label(f'{icon} {text}')


def pillar_cards(analysis_data: Dict):
    """Create 4 mini pillar cards in a grid"""
    pillars = [
        ('TECH', 'technical', 'pillar-tech', '🔧'),
        ('FUND', 'fundamental', 'pillar-fund', '📊'),
        ('SENT', 'sentiment', 'pillar-sent', '💬'),
        ('NEWS', 'news', 'pillar-news', '📰'),
    ]

    with ui.element('div').classes('pillar-grid'):
        for title, key, pillar_class, icon in pillars:
            raw_score = analysis_data.get(key, 0)
            display_score = convert_pillar_score_to_display(raw_score)
            percentage = (display_score / 25) * 100

            with ui.element('div').classes(f'pillar-card-mini {pillar_class}'):
                ui.label(f'{icon} {title}').classes('pillar-title')
                with ui.row().classes('items-baseline justify-center gap-1'):
                    color = 'profit' if display_score >= 15 else ('warning' if display_score >= 10 else 'loss')
                    ui.label(f'{display_score:.0f}').classes(f'pillar-score {color}')
                    ui.label('/25').classes('pillar-max')
                with ui.element('div').classes('pillar-bar-mini'):
                    ui.element('div').classes('pillar-bar-fill').style(f'width: {percentage}%')


def threshold_legend():
    """Create threshold legend explaining score meanings"""
    with ui.element('div').classes('threshold-legend'):
        with ui.element('div').classes('threshold-item'):
            ui.element('span').classes('threshold-dot buy')
            ui.label('≥55: Signal ACHETER')
        with ui.element('div').classes('threshold-item'):
            ui.element('span').classes('threshold-dot watch')
            ui.label('40-55: À SUIVRE')
        with ui.element('div').classes('threshold-item'):
            ui.element('span').classes('threshold-dot reject')
            ui.label('<40: PASSER')


def format_key_factor(factor: str) -> str:
    """Format technical factors to human-readable text"""
    replacements = {
        'ema_bullish': 'EMAs alignées en tendance haussière',
        'ema_bearish': 'EMAs alignées en tendance baissière',
        'rsi_breakout': 'Cassure de la trendline RSI confirmée',
        'volume_surge': 'Volume en forte hausse vs moyenne',
        'volume_spike': 'Pic de volume détecté',
        'macd_bullish': 'MACD en croisement haussier',
        'support_test': 'Test de support EMA réussi',
        'momentum_positive': 'Momentum positif',
        'trend_up': 'Tendance haussière confirmée',
    }
    # Try to match and replace
    factor_lower = factor.lower().replace(' ', '_')
    for key, value in replacements.items():
        if key in factor_lower:
            return value
    return factor


def buy_explanation_card(analysis_data: Dict, score: float):
    """Create a detailed explanation card for BUY signals (score >= 50)"""
    if score < 50:
        return

    key_factors = analysis_data.get('key_factors', [])
    risk_factors = analysis_data.get('risk_factors', [])
    pillar_details = analysis_data.get('pillar_details', {})

    with ui.element('div').classes('buy-explanation'):
        # Title
        with ui.element('div').classes('buy-explanation-title'):
            ui.label('🎯')
            ui.label('POURQUOI CETTE OPPORTUNITÉ?')

        # Key factors
        if key_factors:
            for factor in key_factors[:5]:
                with ui.element('div').classes('buy-factor'):
                    ui.label('✓').classes('buy-factor-check')
                    ui.label(format_key_factor(factor)).classes('buy-factor-text')
        else:
            # Generate from pillar details
            t_detail = pillar_details.get('technical', {})
            f_detail = pillar_details.get('fundamental', {})
            s_detail = pillar_details.get('sentiment', {})

            if t_detail.get('reasoning'):
                with ui.element('div').classes('buy-factor'):
                    ui.label('✓').classes('buy-factor-check')
                    ui.label(t_detail.get('reasoning', '')[:100]).classes('buy-factor-text')

    # Risk section
    if risk_factors:
        with ui.element('div').classes('risk-section'):
            ui.label('⚠️ RISQUES IDENTIFIÉS').classes('risk-title')
            for risk in risk_factors[:3]:
                ui.label(f'• {risk}').classes('risk-item')


def discovery_timeline(journey: Dict):
    """Create a visual timeline showing how the symbol was discovered"""
    steps = journey.get('journey', [])
    if not steps:
        return

    with ui.element('div').classes('discovery-timeline'):
        ui.element('div').classes('timeline-line')

        for step in steps[:5]:
            step_name = step.get('step', 'unknown')
            step_time = step.get('time', '')
            step_data = step.get('data', {})

            time_display = step_time[11:16] if len(step_time) > 16 else 'N/A'

            with ui.element('div').classes('timeline-item'):
                ui.element('div').classes('timeline-dot')
                ui.label(time_display).classes('timeline-time')

                if step_name == 'discovery':
                    sources = step_data.get('sources', [])
                    ui.label('Détecté via scan social').classes('timeline-title')
                    if sources:
                        ui.label(f'Sources: {", ".join(sources)}').classes('timeline-detail')
                elif step_name == 'analysis':
                    ui.label('Analyse 4 Piliers').classes('timeline-title')
                    t = convert_pillar_score_to_display(step_data.get('technical', 0))
                    f = convert_pillar_score_to_display(step_data.get('fundamental', 0))
                    s = convert_pillar_score_to_display(step_data.get('sentiment', 0))
                    n = convert_pillar_score_to_display(step_data.get('news', 0))
                    total = t + f + s + n
                    ui.label(f'Score: {total:.0f}/100 (T{t:.0f} F{f:.0f} S{s:.0f} N{n:.0f})').classes('timeline-detail')
                else:
                    ui.label(step_name.upper()).classes('timeline-title')


def market_status_bar():
    """Create a Finary-style market status bar"""
    mode = get_current_market_mode()

    mode_labels = {
        'MARKET': ('US Markets Ouverts', 'open'),
        'PRE_MARKET': ('Pré-Market', 'premarket'),
        'AFTER_HOURS': ('After-Hours', 'premarket'),
        'WEEKEND': ('Marchés Fermés', 'closed'),
    }

    label, dot_class = mode_labels.get(mode, ('Inconnu', 'closed'))

    with ui.element('div').classes('market-status-bar'):
        with ui.element('div').classes('market-status-indicator'):
            ui.element('span').classes(f'market-dot {dot_class}')
            ui.label(label)
        ui.label('|').classes('text-slate-600')
        ui.label(f'Mode: {mode}').classes('text-slate-400')


def show_journey_dialog(symbol: str, journey: Dict):
    """Show detailed journey dialog for a symbol - FINARY STYLE"""
    status = journey.get('current_status', 'discovered')
    score = journey.get('current_score', 0)
    steps = journey.get('journey', [])
    last_update = journey.get('last_update', '')

    # Get analysis data
    analysis_step = next((s for s in steps if s.get('step') == 'analysis'), None)
    analysis_data = analysis_step.get('data', {}) if analysis_step else {}
    discovery_step = next((s for s in steps if s.get('step') == 'discovery'), None)

    # Prevent auto-refresh from closing dialog
    state.dialog_open = True

    def close_dialog():
        state.dialog_open = False
        dialog.close()

    with ui.dialog() as dialog, ui.card().classes('dialog-finary w-full max-w-4xl'):
        # Header - Finary Style
        with ui.element('div').classes('dialog-header'):
            with ui.row().classes('justify-between items-start w-full'):
                with ui.element('div').classes('text-center flex-1'):
                    ui.label(symbol).classes('text-3xl font-bold mb-1')
                    # Status badge
                    status_colors = {
                        'discovered': 'bg-blue-500/20 text-blue-400',
                        'analyzing': 'bg-yellow-500/20 text-yellow-400',
                        'analyzed': 'bg-yellow-500/20 text-yellow-400',
                        'watch': 'bg-orange-500/20 text-orange-400',
                        'signal': 'bg-green-500/20 text-green-400',
                        'bought': 'bg-purple-500/20 text-purple-400',
                        'rejected': 'bg-red-500/20 text-red-400'
                    }
                    status_class = status_colors.get(status, 'bg-slate-500/20 text-slate-400')
                    ui.label(status.upper()).classes(f'text-xs px-3 py-1 rounded-full {status_class}')
                ui.button(icon='close', on_click=close_dialog).props('flat round dense').classes('text-slate-400')

        # Content
        with ui.element('div').classes('dialog-body max-h-[70vh] overflow-auto'):
            # Score Ring + Signal Badge - Prominent Display
            with ui.element('div').classes('flex justify-center items-center gap-8 mb-8'):
                score_ring(score, 140)
                with ui.element('div').classes('text-center'):
                    signal_badge(score)
                    ui.label(f'Dernière MAJ: {last_update[:16] if last_update else "N/A"}').classes('text-xs text-slate-500 mt-2')

            # 4 Pillars Mini Cards - Quick Overview
            if analysis_data:
                with ui.element('div').classes('dialog-section mb-6'):
                    ui.label('ANALYSE 4 PILIERS').classes('dialog-section-title')
                    pillar_cards(analysis_data)

            # Buy Explanation Card (for scores >= 50)
            if score >= 50 and analysis_data:
                buy_explanation_card(analysis_data, score)

            # Discovery Timeline - How was it found?
            if steps:
                with ui.element('div').classes('dialog-section mb-6'):
                    ui.label('🔍 PARCOURS DE DÉCOUVERTE').classes('dialog-section-title')
                    discovery_timeline(journey)

            # Detailed Pillar Breakdown (Collapsible)
            if analysis_data:
                with ui.expansion('📊 Détails des 4 Piliers', icon='analytics').classes('w-full mb-4 bg-slate-800/50 rounded-xl'):
                    pillar_details = analysis_data.get('pillar_details', {})

                    # Technical
                    t_score = convert_pillar_score_to_display(analysis_data.get('technical', 0))
                    t_detail = pillar_details.get('technical', {})
                    with ui.element('div').classes('mb-4 p-4 rounded-xl border-l-4 border-blue-500 bg-blue-500/10'):
                        with ui.row().classes('justify-between items-center mb-2'):
                            ui.label('🔧 Technical').classes('font-bold')
                            t_color = 'profit' if t_score >= 15 else ('warning' if t_score >= 10 else 'loss')
                            ui.label(f'{t_score:.1f}/25').classes(f'font-bold {t_color}')
                        ui.label(t_detail.get('reasoning', 'Analyse des indicateurs: EMA, RSI, MACD, Volume')).classes('text-sm text-slate-300')
                        if t_detail.get('factors'):
                            with ui.row().classes('gap-2 mt-2 flex-wrap'):
                                for factor in t_detail.get('factors', [])[:5]:
                                    f_text = factor.get('name', str(factor)) if isinstance(factor, dict) else str(factor)
                                    ui.label(f_text).classes('text-xs px-2 py-1 bg-slate-700 rounded')

                    # Fundamental
                    f_score = convert_pillar_score_to_display(analysis_data.get('fundamental', 0))
                    f_detail = pillar_details.get('fundamental', {})
                    with ui.element('div').classes('mb-4 p-4 rounded-xl border-l-4 border-purple-500 bg-purple-500/10'):
                        with ui.row().classes('justify-between items-center mb-2'):
                            ui.label('📊 Fundamental').classes('font-bold')
                            f_color = 'profit' if f_score >= 15 else ('warning' if f_score >= 10 else 'loss')
                            ui.label(f'{f_score:.1f}/25').classes(f'font-bold {f_color}')
                        ui.label(f_detail.get('reasoning', 'Analyse fondamentale: P/E, ROE, marges, croissance')).classes('text-sm text-slate-300')

                    # Sentiment
                    s_score = convert_pillar_score_to_display(analysis_data.get('sentiment', 0))
                    s_detail = pillar_details.get('sentiment', {})
                    with ui.element('div').classes('mb-4 p-4 rounded-xl border-l-4 border-pink-500 bg-pink-500/10'):
                        with ui.row().classes('justify-between items-center mb-2'):
                            ui.label('💬 Sentiment').classes('font-bold')
                            s_color = 'profit' if s_score >= 15 else ('warning' if s_score >= 10 else 'loss')
                            ui.label(f'{s_score:.1f}/25').classes(f'font-bold {s_color}')
                        ui.label(s_detail.get('reasoning', 'Sentiment social: X/Twitter via Grok, Reddit')).classes('text-sm text-slate-300')

                    # News
                    n_score = convert_pillar_score_to_display(analysis_data.get('news', 0))
                    n_detail = pillar_details.get('news', {})
                    with ui.element('div').classes('p-4 rounded-xl border-l-4 border-cyan-500 bg-cyan-500/10'):
                        with ui.row().classes('justify-between items-center mb-2'):
                            ui.label('📰 News').classes('font-bold')
                            n_color = 'profit' if n_score >= 15 else ('warning' if n_score >= 10 else 'loss')
                            ui.label(f'{n_score:.1f}/25').classes(f'font-bold {n_color}')
                        ui.label(n_detail.get('reasoning', 'Analyse des actualités via Gemini/OpenRouter')).classes('text-sm text-slate-300')

            else:
                # No analysis data yet
                with ui.element('div').classes('dialog-section text-center'):
                    with ui.element('div').classes('loading-overlay-inline'):
                        ui.spinner('dots', size='lg', color='purple')
                    ui.label('Analyse en attente').classes('font-bold text-lg mt-4 mb-2')
                    ui.label('Ce symbole a été découvert mais n\'a pas encore été analysé.').classes('text-sm text-slate-400')
                    ui.label('Le bot analysera automatiquement ou utilisez "Scan Manuel".').classes('text-sm text-slate-500 mt-1')

            # Full Reasoning (Collapsible)
            reasoning_text = ""
            for step in steps:
                if step.get('reasoning'):
                    reasoning_text += step.get('reasoning') + "\n\n"
            if reasoning_text:
                with ui.expansion('💭 Raisonnement Complet', icon='psychology').classes('w-full bg-slate-800/50 rounded-xl'):
                    ui.label(reasoning_text).classes('text-sm text-slate-300 whitespace-pre-wrap')

        # Footer - Finary Style
        with ui.element('div').classes('p-4 border-t border-slate-700/50 flex justify-between items-center'):
            # Threshold reminder
            with ui.row().classes('gap-4 text-xs text-slate-500'):
                ui.label('≥55 = Acheter')
                ui.label('40-55 = Suivre')
                ui.label('<40 = Passer')
            ui.button('Fermer', on_click=close_dialog).classes('btn-finary-primary')

    dialog.open()


def journey_card(symbol: str, journey: Dict):
    """Create a symbol journey card - FINARY STOCK CARD STYLE"""
    status = journey.get('current_status', 'discovered')
    score = journey.get('current_score', 0)
    steps = journey.get('journey', [])

    # Get analysis data
    analysis_step = next((s for s in steps if s.get('step') == 'analysis'), None)
    analysis_data = analysis_step.get('data', {}) if analysis_step else {}

    # Determine card glow based on score
    glow_class = ''
    if score >= 55:
        glow_class = 'glow-green'
    elif score >= 40:
        glow_class = 'glow-yellow'

    def open_dialog():
        show_journey_dialog(symbol, journey)

    # Finary-style stock card
    with ui.element('div').classes(f'stock-card-finary {glow_class}').on('click', open_dialog):
        # Header: Symbol + Score Ring
        with ui.element('div').classes('stock-card-header'):
            with ui.element('div'):
                ui.label(symbol).classes('stock-card-symbol')
                # Status indicator
                status_labels = {
                    'discovered': ('Découvert', 'text-blue-400'),
                    'analyzing': ('Analyse...', 'text-yellow-400'),
                    'analyzed': ('Analysé', 'text-yellow-400'),
                    'watch': ('À suivre', 'text-orange-400'),
                    'signal': ('Signal!', 'text-green-400'),
                    'bought': ('En position', 'text-purple-400'),
                    'rejected': ('Rejeté', 'text-red-400')
                }
                label, color = status_labels.get(status, ('Inconnu', 'text-slate-400'))
                ui.label(label).classes(f'stock-card-status {color}')

            # Mini Score Ring
            if score > 0:
                score_ring(score, 60)
            else:
                with ui.element('div').classes('w-[60px] h-[60px] flex items-center justify-center'):
                    ui.spinner('dots', size='sm', color='slate')

        # Mini Pillar Bars (horizontal compact)
        if analysis_data:
            with ui.element('div').classes('stock-card-pillars'):
                pillars = [
                    ('T', 'technical', '#3B82F6'),
                    ('F', 'fundamental', '#8B5CF6'),
                    ('S', 'sentiment', '#EC4899'),
                    ('N', 'news', '#22D3EE'),
                ]
                for label, key, color in pillars:
                    raw = analysis_data.get(key, 0)
                    display = convert_pillar_score_to_display(raw)
                    pct = (display / 25) * 100

                    with ui.element('div').classes('stock-pillar-mini'):
                        ui.label(label).classes('stock-pillar-label')
                        with ui.element('div').classes('stock-pillar-bar-bg'):
                            ui.element('div').classes('stock-pillar-bar-fill').style(f'width: {pct}%; background: {color}')
                        ui.label(f'{display:.0f}').classes('stock-pillar-value')

        # Signal Badge (prominent)
        with ui.element('div').classes('stock-card-signal'):
            signal_badge(score)

        # Key factor hint (if available)
        key_factors = analysis_data.get('key_factors', [])
        if key_factors:
            ui.label(f'🎯 {format_key_factor(key_factors[0])}').classes('stock-card-key-factor')

        # Click hint
        ui.label('Cliquer pour détails →').classes('stock-card-hint')

        # Short reasoning preview
        if steps:
            last_step = steps[-1]
            reasoning = last_step.get('reasoning', '')
            if reasoning:
                with ui.element('div').classes('reasoning-box'):
                    ui.label('💭 Apercu:').classes('reasoning-title')
                    short_reasoning = reasoning[:150] + '...' if len(reasoning) > 150 else reasoning
                    ui.label(short_reasoning).classes('reasoning-text')

def show_thought_dialog(thought: Dict):
    """Show detailed thought dialog"""
    category = thought.get('category', 'analysis')
    content = thought.get('content', '')
    details = thought.get('details', '')
    time_display = thought.get('time_display', '')
    timestamp = thought.get('timestamp', '')

    icon_map = {
        'startup': '🤖',
        'discovery': '🔍',
        'analysis': '📊',
        'signal': '✅',
        'error': '❌',
        'warning': '⚠️'
    }
    icon = icon_map.get(category, '💭')

    # Category descriptions
    category_desc = {
        'startup': 'Initialisation du systeme et des connexions',
        'discovery': 'Phase de decouverte des symboles via sources sociales et momentum',
        'analysis': 'Analyse technique et fondamentale des symboles',
        'signal': 'Detection de signaux de trading (BUY/WATCH/REJECT)',
        'error': 'Erreur survenue lors de l\'execution',
        'warning': 'Avertissement ou situation inhabituelle'
    }

    # Prevent auto-refresh from closing dialog
    state.dialog_open = True

    def close_dialog():
        state.dialog_open = False
        dialog.close()

    with ui.dialog() as dialog, ui.card().classes('w-full max-w-2xl bg-slate-800 p-0'):
        # Header
        with ui.element('div').classes('p-6 border-b border-slate-700'):
            with ui.row().classes('justify-between items-center'):
                with ui.row().classes('items-center gap-3'):
                    ui.label(icon).classes('text-3xl')
                    ui.label('Detail de la Pensee').classes('text-xl font-bold')
                ui.button(icon='close', on_click=close_dialog).props('flat round')

        # Content
        with ui.element('div').classes('p-6'):
            # Category badge
            with ui.element('div').classes('mb-4'):
                ui.element('span').classes('thought-category text-lg').text = category.upper()
                ui.label(time_display).classes('ml-4 text-sm muted')

            # Main content
            with ui.element('div').classes('mb-6 p-4 bg-slate-700/50 rounded-xl'):
                ui.label('📝 Contenu').classes('font-bold text-lg mb-2')
                ui.label(content).classes('text-lg')

            # Details
            if details:
                with ui.element('div').classes('mb-6 p-4 bg-slate-700/50 rounded-xl'):
                    ui.label('📋 Details').classes('font-bold text-lg mb-2')
                    ui.label(details).classes('text-sm text-slate-300')

            # Category explanation
            with ui.element('div').classes('mb-6 p-4 bg-slate-700/50 rounded-xl'):
                ui.label('ℹ️ Type de Pensee').classes('font-bold text-lg mb-2')
                ui.label(category_desc.get(category, 'Pensee generale du systeme')).classes('text-sm text-slate-300')

            # What this means
            with ui.element('div').classes('p-4 border-l-4 border-purple-500 bg-purple-500/10 rounded-r-xl'):
                ui.label('💡 Explication').classes('font-bold mb-2')
                if category == 'discovery':
                    ui.label("Le systeme scanne les sources sociales (Reddit, X/Twitter via Grok) et analyse le momentum pour identifier les symboles potentiellement interessants. Ces symboles sont ensuite ajoutes a la watchlist pour une analyse approfondie.").classes('text-sm text-slate-300')
                elif category == 'analysis':
                    ui.label("L'analyse utilise 4 piliers: Technical (indicateurs), Fundamental (ratios financiers), Sentiment (reseaux sociaux), News (actualites). Chaque pilier contribue au score final sur 100.").classes('text-sm text-slate-300')
                elif category == 'signal':
                    ui.label("Un signal est genere quand le score total depasse certains seuils: >=55 = BUY, 40-55 = WATCH, <40 = REJECT. Les signaux BUY peuvent declencher des ordres si le mode automatique est actif.").classes('text-sm text-slate-300')
                elif category == 'startup':
                    ui.label("Phase d'initialisation: connexion aux APIs, chargement de la configuration, preparation de l'agent de trading.").classes('text-sm text-slate-300')
                elif category == 'error':
                    ui.label("Une erreur s'est produite. Verifiez les logs pour plus de details. Les erreurs courantes: API timeout, donnees manquantes, probleme de connexion.").classes('text-sm text-slate-300')
                else:
                    ui.label("Information generale sur l'etat du systeme ou une action en cours.").classes('text-sm text-slate-300')

            # Timestamp
            if timestamp:
                ui.label(f'Timestamp complet: {timestamp}').classes('text-xs muted mt-4')

        # Footer
        with ui.element('div').classes('p-4 border-t border-slate-700 flex justify-end'):
            ui.button('Fermer', on_click=close_dialog).classes('btn-primary')

    dialog.open()


def thought_item(thought: Dict):
    """Create a chain of thought item - CLICKABLE"""
    category = thought.get('category', 'analysis')
    content = thought.get('content', '')
    details = thought.get('details', '')
    time_display = thought.get('time_display', '')

    icon_map = {
        'startup': '🤖',
        'discovery': '🔍',
        'analysis': '📊',
        'signal': '✅',
        'error': '❌',
        'warning': '⚠️'
    }
    icon = icon_map.get(category, '💭')

    # Make clickable
    with ui.element('div').classes(f'thought-item {category} cursor-pointer hover:bg-slate-700/80').on('click', lambda t=thought: show_thought_dialog(t)):
        with ui.row().classes('justify-between items-start mb-2'):
            with ui.row().classes('items-center gap-2'):
                ui.label(icon).classes('text-lg')
                ui.label(content).classes('font-semibold')
            with ui.row().classes('items-center gap-2'):
                ui.element('span').classes('thought-category').text = category
                ui.label(time_display).classes('thought-time')
        if details:
            ui.label(details).classes('text-sm text-slate-400 mt-1')
        # Click hint
        ui.label('Cliquer pour plus de details').classes('text-xs text-slate-600 mt-2 italic')

def nav_header():
    """Create navigation header"""
    with ui.header().classes('nav-header px-4 py-3'):
        with ui.row().classes('w-full items-center justify-between'):
            # Logo
            with ui.row().classes('items-center gap-3'):
                ui.icon('show_chart').classes('text-3xl accent')
                ui.label('TradingBot').classes('text-xl font-bold hide-mobile')
                ui.label('V5').classes('text-sm text-slate-500 hide-mobile')

            # Navigation
            with ui.row().classes('gap-1'):
                nav_items = [
                    ('/', 'Dashboard', 'dashboard'),
                    ('/watchlist', 'Watchlist', 'visibility'),
                    ('/portfolio', 'Portfolio', 'account_balance'),
                    ('/reasoning', 'Reasoning', 'psychology'),
                    ('/learning', 'Learning', 'school'),
                    ('/training', 'Training', 'fitness_center'),
                    ('/alerts', 'Alertes', 'notifications'),
                    ('/settings', 'Config', 'settings'),
                ]
                for path, label, icon in nav_items:
                    with ui.button(on_click=lambda p=path: ui.navigate.to(p)).props('flat').classes('nav-item'):
                        ui.icon(icon).classes('text-lg')
                        ui.label(label).classes('ml-1 hide-mobile')

# =============================================================================
# PAGES
# =============================================================================

@ui.page('/')
def dashboard_page():
    """Main dashboard page - FINARY STYLE"""
    ui.add_head_html(STYLES)
    nav_header()

    # Market Status Bar - Finary Style
    market_status_bar()

    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-6'):
        # Header with bot controls
        with ui.row().classes('w-full items-center justify-between flex-wrap gap-4'):
            with ui.column().classes('gap-1'):
                ui.label('Dashboard').classes('text-3xl font-bold')
                mode_label = ui.label(f'Mode: {state.bot_mode}').classes('text-sm muted')

            with ui.row().classes('gap-3'):
                start_btn = ui.button('Demarrer', icon='play_arrow', on_click=start_bot).classes('btn-success')
                stop_btn = ui.button('Arreter', icon='stop', on_click=stop_bot).classes('btn-danger')
                ui.button('Scan Manuel', icon='radar', on_click=run_manual_scan).classes('btn-primary')

        # Stats grid - responsive
        with ui.row().classes('w-full gap-4 flex-wrap'):
            with ui.element('div').classes('flex-1 min-w-[220px]'):
                stat_card('Capital', f"{state.capital:,.0f} EUR",
                         f"Disponible pour trading", 'account_balance_wallet')

            with ui.element('div').classes('flex-1 min-w-[220px]'):
                pnl_class = 'profit' if state.daily_pnl >= 0 else 'loss'
                stat_card('P&L Jour', format_pnl(state.daily_pnl) + ' EUR',
                         'Performance journaliere', 'trending_up', pnl_class, pnl_class)

            with ui.element('div').classes('flex-1 min-w-[220px]'):
                status_card = ui.element('div').classes('stat-card')
                with status_card:
                    with ui.row().classes('items-center gap-3'):
                        status_dot = ui.element('span').classes(f'status-dot {"active" if state.bot_running else "inactive"}')
                        with ui.column().classes('gap-0'):
                            status_label = ui.label(state.bot_status).classes('stat-value text-xl')
                            ui.label('STATUT BOT').classes('stat-label')

        # Two columns layout
        with ui.row().classes('w-full gap-6 flex-wrap'):
            # Recent Alerts + Chain of Thought
            with ui.column().classes('flex-1 min-w-[350px] gap-4'):
                # Recent Alerts
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('section-title'):
                        ui.element('div').classes('section-title-icon green').text = '📊'
                        ui.label('Dernieres Alertes').classes('font-bold text-lg')
                        ui.badge(str(len(state.alerts))).classes('bg-emerald-500 ml-auto')

                    alerts_container = ui.column().classes('gap-2 max-h-80 overflow-auto')

                    def update_alerts():
                        alerts_container.clear()
                        with alerts_container:
                            if not state.alerts:
                                ui.label('Aucune alerte - Lancez un scan').classes('muted italic py-4 text-center')
                            else:
                                for alert in state.alerts[:6]:
                                    alert_card(alert)

                    update_alerts()

                # Chain of Thought Summary
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('section-title'):
                        ui.element('div').classes('section-title-icon purple').text = '💭'
                        ui.label('Chain of Thought').classes('font-bold text-lg')

                    thoughts_container = ui.column().classes('gap-2 max-h-64 overflow-auto')

                    def update_thoughts():
                        thoughts_container.clear()
                        with thoughts_container:
                            if not state.chain_of_thought:
                                ui.label('En attente des premieres analyses...').classes('muted italic py-4 text-center')
                            else:
                                for t in state.chain_of_thought[:5]:
                                    thought_item(t)

                    update_thoughts()

            # Symbol Journeys - Finary Style
            with ui.column().classes('flex-1 min-w-[400px] gap-4'):
                # Threshold Legend - explains scores
                threshold_legend()

                with ui.element('div').classes('card'):
                    with ui.element('div').classes('section-title'):
                        ui.element('div').classes('section-title-icon blue').text = '🔄'
                        ui.label('Opportunités Détectées').classes('font-bold text-lg')
                        ui.badge(str(len(state.symbol_journeys))).classes('bg-blue-500 ml-auto')

                    journeys_container = ui.column().classes('gap-3 max-h-[500px] overflow-auto')

                    def update_journeys():
                        journeys_container.clear()
                        with journeys_container:
                            if not state.symbol_journeys:
                                with ui.column().classes('items-center py-8'):
                                    ui.label('🤖').classes('text-5xl mb-4')
                                    ui.label('En attente des premieres analyses...').classes('muted')
                                    ui.label('Lancez le bot pour voir le flux de decision').classes('text-xs muted mt-2')
                            else:
                                # Sort by most recent
                                sorted_journeys = sorted(
                                    state.symbol_journeys.items(),
                                    key=lambda x: x[1].get('last_update', ''),
                                    reverse=True
                                )[:8]
                                for symbol, journey in sorted_journeys:
                                    journey_card(symbol, journey)

                    update_journeys()

        # Logs (collapsible)
        with ui.expansion('📋 Logs Systeme', icon='article').classes('w-full bg-slate-800/50 rounded-xl'):
            logs_container = ui.column().classes('gap-1 max-h-48 overflow-auto font-mono text-sm p-2')

            def update_logs():
                logs_container.clear()
                with logs_container:
                    for log in state.logs[:20]:
                        ui.label(log).classes('text-slate-400')

            update_logs()

        # Update timer
        def update_dashboard():
            # Skip updates if a dialog is open to prevent it from closing
            if state.dialog_open:
                return

            mode_label.text = f'Mode: {get_current_market_mode()}'
            status_label.text = state.bot_status
            status_dot._classes = [c for c in status_dot._classes if 'active' not in c and 'inactive' not in c]
            status_dot._classes.append('status-dot')
            status_dot._classes.append('active' if state.bot_running else 'inactive')
            update_alerts()
            update_thoughts()
            update_journeys()
            update_logs()

        ui.timer(5, update_dashboard)


@ui.page('/watchlist')
def watchlist_page():
    """Watchlist page"""
    ui.add_head_html(STYLES)
    nav_header()

    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-4'):
        with ui.row().classes('justify-between items-center'):
            ui.label('Watchlist').classes('text-2xl font-bold')
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')

        # Add symbol input
        with ui.element('div').classes('card'):
            with ui.row().classes('gap-3 items-center'):
                symbol_input = ui.input('Ajouter symbole').classes('flex-1')

                def add_symbol():
                    symbol = symbol_input.value.upper().strip()
                    if symbol and symbol not in [w.get('symbol') for w in state.watchlist]:
                        state.watchlist.append({'symbol': symbol, 'added': datetime.now().isoformat()})
                        symbol_input.value = ''
                        safe_notify(f'{symbol} ajoute', type='positive')
                        update_watchlist()

                ui.button('Ajouter', icon='add', on_click=add_symbol).classes('btn-primary')

        # Watchlist
        watchlist_container = ui.column().classes('w-full gap-3')

        def update_watchlist():
            watchlist_container.clear()
            with watchlist_container:
                if not state.watchlist:
                    with ui.element('div').classes('card text-center py-8'):
                        ui.label('📋').classes('text-4xl mb-2')
                        ui.label('Watchlist vide').classes('muted')
                else:
                    for item in state.watchlist[:50]:
                        symbol = item.get('symbol', 'N/A')
                        journey = state.symbol_journeys.get(symbol, {})
                        score = journey.get('current_score', 0)
                        status = journey.get('current_status', 'pending')

                        with ui.element('div').classes('card'):
                            with ui.row().classes('justify-between items-center'):
                                with ui.row().classes('items-center gap-4'):
                                    ui.label(symbol).classes('font-bold text-xl')
                                    if score > 0:
                                        score_color = 'profit' if score >= 55 else ('warning' if score >= 40 else 'loss')
                                        ui.label(f'{score}/100').classes(f'font-semibold {score_color}')
                                    ui.label(status.upper()).classes('text-xs muted px-2 py-1 bg-slate-700 rounded')

                                def remove_symbol(s=symbol):
                                    state.watchlist = [w for w in state.watchlist if w.get('symbol') != s]
                                    update_watchlist()
                                    safe_notify(f'{s} retire', type='info')

                                ui.button(icon='delete', on_click=remove_symbol).props('flat round').classes('text-red-400')

        update_watchlist()


@ui.page('/portfolio')
def portfolio_page():
    """Portfolio page - FINARY STYLE PREMIUM"""
    ui.add_head_html(STYLES)
    nav_header()

    # Calculate portfolio metrics
    total_value = state.capital
    total_pnl_pct = (state.total_pnl / state.capital * 100) if state.capital > 0 else 0
    win_rate = 0
    profit_factor = 0
    max_drawdown = 0

    # Calculate from trades if available
    if state.trades:
        winners = [t for t in state.trades if t.get('pnl', 0) > 0]
        losers = [t for t in state.trades if t.get('pnl', 0) < 0]
        win_rate = (len(winners) / len(state.trades) * 100) if state.trades else 0
        total_wins = sum(t.get('pnl', 0) for t in winners)
        total_losses = abs(sum(t.get('pnl', 0) for t in losers))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else total_wins

    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-6'):
        # Header
        with ui.row().classes('justify-between items-center w-full'):
            with ui.column().classes('gap-1'):
                ui.label('💼 Portfolio').classes('text-3xl font-bold')
                ui.label('Suivi des positions et performance').classes('text-sm text-slate-400')
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')

        # Top Stats Row - Finary Style
        with ui.row().classes('w-full gap-4 flex-wrap'):
            # Total Value Card
            with ui.element('div').classes('flex-1 min-w-[200px] portfolio-stat-card'):
                ui.label('Valeur Totale').classes('portfolio-stat-label')
                ui.label(f'{total_value:,.0f} €').classes('portfolio-stat-value font-mono')
                pnl_color = 'profit' if state.total_pnl >= 0 else 'loss'
                ui.label(f'{format_pnl(state.total_pnl)} € ({total_pnl_pct:+.1f}%)').classes(f'portfolio-stat-change {pnl_color}')

            # P&L Jour
            with ui.element('div').classes('flex-1 min-w-[200px] portfolio-stat-card'):
                ui.label('P&L Aujourd\'hui').classes('portfolio-stat-label')
                day_color = 'profit' if state.daily_pnl >= 0 else 'loss'
                ui.label(f'{format_pnl(state.daily_pnl)} €').classes(f'portfolio-stat-value font-mono {day_color}')
                ui.label('Performance journalière').classes('portfolio-stat-subtitle')

            # Win Rate
            with ui.element('div').classes('flex-1 min-w-[200px] portfolio-stat-card'):
                ui.label('Win Rate').classes('portfolio-stat-label')
                wr_color = 'profit' if win_rate >= 50 else 'loss'
                ui.label(f'{win_rate:.0f}%').classes(f'portfolio-stat-value font-mono {wr_color}')
                ui.label(f'{len(state.trades)} trades').classes('portfolio-stat-subtitle')

            # Profit Factor
            with ui.element('div').classes('flex-1 min-w-[200px] portfolio-stat-card'):
                ui.label('Profit Factor').classes('portfolio-stat-label')
                pf_color = 'profit' if profit_factor >= 1 else 'loss'
                ui.label(f'{profit_factor:.2f}').classes(f'portfolio-stat-value font-mono {pf_color}')
                ui.label('Gains / Pertes').classes('portfolio-stat-subtitle')

        # Two columns layout
        with ui.row().classes('w-full gap-6 flex-wrap'):
            # Left column - Positions
            with ui.column().classes('flex-1 min-w-[400px] gap-4'):
                # Open Positions
                with ui.element('div').classes('card'):
                    with ui.row().classes('justify-between items-center mb-4'):
                        ui.label('📈 Positions Ouvertes').classes('font-bold text-lg')
                        ui.badge('0').classes('bg-purple-500')

                    with ui.column().classes('gap-3'):
                        # Placeholder for open positions
                        with ui.element('div').classes('p-6 text-center bg-slate-800/50 rounded-xl'):
                            ui.label('💤').classes('text-4xl mb-2')
                            ui.label('Aucune position ouverte').classes('text-slate-400')
                            ui.label('Les positions IBKR apparaîtront ici').classes('text-xs text-slate-500 mt-1')

                # Pending Signals
                with ui.element('div').classes('card'):
                    with ui.row().classes('justify-between items-center mb-4'):
                        ui.label('🎯 Signaux en Attente').classes('font-bold text-lg')
                        buy_signals = [j for j in state.symbol_journeys.values() if j.get('current_score', 0) >= 55]
                        ui.badge(str(len(buy_signals))).classes('bg-green-500')

                    with ui.column().classes('gap-2 max-h-64 overflow-auto'):
                        if not buy_signals:
                            ui.label('Aucun signal BUY actif').classes('text-slate-400 italic text-center py-4')
                        else:
                            for symbol, journey in state.symbol_journeys.items():
                                score = journey.get('current_score', 0)
                                if score >= 55:
                                    with ui.element('div').classes('flex justify-between items-center p-3 bg-green-900/20 rounded-lg border border-green-800/30'):
                                        with ui.column().classes('gap-0'):
                                            ui.label(symbol).classes('font-bold')
                                            ui.label('Signal ACHETER').classes('text-xs text-green-400')
                                        with ui.row().classes('items-center gap-2'):
                                            score_ring(score, 40)

            # Right column - History & Stats
            with ui.column().classes('flex-1 min-w-[400px] gap-4'):
                # Trade History
                with ui.element('div').classes('card'):
                    with ui.row().classes('justify-between items-center mb-4'):
                        ui.label('📜 Historique des Trades').classes('font-bold text-lg')
                        ui.badge(str(len(state.trades))).classes('bg-slate-500')

                    if not state.trades:
                        with ui.column().classes('items-center py-8'):
                            ui.label('📊').classes('text-4xl mb-2')
                            ui.label('Aucun trade enregistré').classes('muted')
                            ui.label('Les trades apparaîtront après exécution via IBKR').classes('text-xs text-slate-500 mt-2')
                    else:
                        with ui.column().classes('gap-2 max-h-80 overflow-auto'):
                            for trade in state.trades[:10]:
                                pnl = trade.get('pnl', 0)
                                pnl_pct = trade.get('pnl_pct', 0)
                                exit_type = trade.get('exit_type', 'unknown')

                                pnl_color = 'profit' if pnl > 0 else 'loss'
                                exit_icon = '✅' if exit_type == 'take_profit' else ('🛑' if exit_type == 'stop_loss' else '📤')

                                with ui.element('div').classes('trade-history-item'):
                                    with ui.row().classes('justify-between items-center'):
                                        with ui.column().classes('gap-0'):
                                            ui.label(trade.get('symbol', 'N/A')).classes('font-bold')
                                            ui.label(trade.get('date', '')[:10]).classes('text-xs text-slate-500')
                                        with ui.column().classes('items-end gap-0'):
                                            ui.label(f'{format_pnl(pnl)} €').classes(f'font-mono {pnl_color}')
                                            ui.label(f'{exit_icon} {exit_type}').classes('text-xs text-slate-400')

                # Stats détaillées
                with ui.element('div').classes('card'):
                    ui.label('📊 Statistiques Détaillées').classes('font-bold text-lg mb-4')

                    with ui.element('div').classes('grid grid-cols-2 gap-4'):
                        # Left stats
                        with ui.column().classes('gap-3'):
                            with ui.element('div').classes('stat-mini'):
                                ui.label('Meilleur Trade').classes('stat-mini-label')
                                best = max([t.get('pnl', 0) for t in state.trades], default=0)
                                ui.label(f'+{best:.0f} €').classes('stat-mini-value profit')

                            with ui.element('div').classes('stat-mini'):
                                ui.label('Pire Trade').classes('stat-mini-label')
                                worst = min([t.get('pnl', 0) for t in state.trades], default=0)
                                ui.label(f'{worst:.0f} €').classes('stat-mini-value loss')

                        # Right stats
                        with ui.column().classes('gap-3'):
                            with ui.element('div').classes('stat-mini'):
                                ui.label('Durée Moy.').classes('stat-mini-label')
                                avg_days = sum(t.get('hold_days', 0) for t in state.trades) / len(state.trades) if state.trades else 0
                                ui.label(f'{avg_days:.1f} jours').classes('stat-mini-value')

                            with ui.element('div').classes('stat-mini'):
                                ui.label('Max Drawdown').classes('stat-mini-label')
                                ui.label(f'{max_drawdown:.1f}%').classes('stat-mini-value loss')


@ui.page('/reasoning')
def reasoning_page():
    """Reasoning page - 4 Pillars analysis with detailed visualization"""
    ui.add_head_html(STYLES)
    nav_header()

    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-6'):
        with ui.row().classes('justify-between items-center'):
            ui.label('Reasoning - Analyse 4 Piliers').classes('text-2xl font-bold')
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')

        # Chain of Thought - Detailed
        with ui.element('div').classes('card'):
            with ui.element('div').classes('section-title'):
                ui.element('div').classes('section-title-icon purple').text = '💭'
                ui.label('Chain of Thought - Flux de Raisonnement').classes('font-bold text-lg')
                ui.badge(str(len(state.chain_of_thought))).classes('bg-purple-500 ml-auto')

            with ui.row().classes('gap-2 mb-4 flex-wrap'):
                ui.label('Categories:').classes('text-sm muted')
                ui.element('span').classes('thought-category').text = 'DISCOVERY'
                ui.element('span').classes('thought-category').style('background: rgba(16, 185, 129, 0.2); color: #10b981;').text = 'SIGNAL'
                ui.element('span').classes('thought-category').style('background: rgba(239, 68, 68, 0.2); color: #ef4444;').text = 'ERROR'

            thoughts_container = ui.column().classes('gap-3 max-h-96 overflow-auto')

            def update_thoughts():
                thoughts_container.clear()
                with thoughts_container:
                    if not state.chain_of_thought:
                        with ui.column().classes('items-center py-8'):
                            ui.label('🤔').classes('text-5xl mb-4')
                            ui.label('Aucune pensee enregistree').classes('muted text-lg')
                            ui.label('Lancez le bot ou un scan manuel pour voir le raisonnement').classes('text-sm muted mt-2')
                    else:
                        for t in state.chain_of_thought[:25]:
                            thought_item(t)

            update_thoughts()
            # Manual refresh only - NO auto-refresh timer (fixes report closing bug)
            ui.button('Rafraichir', icon='refresh', on_click=update_thoughts).props('flat').classes('mt-4')

        # Symbol Journeys with Flow
        with ui.element('div').classes('card'):
            with ui.element('div').classes('section-title'):
                ui.element('div').classes('section-title-icon blue').text = '🔄'
                ui.label('Parcours des Symboles - Flux de Decision').classes('font-bold text-lg')
                ui.badge(str(len(state.symbol_journeys))).classes('bg-blue-500 ml-auto')

            # Legend
            with ui.row().classes('gap-4 mb-4 flex-wrap text-sm'):
                ui.label('Legende:').classes('muted')
                with ui.row().classes('items-center gap-1'):
                    ui.element('div').classes('w-3 h-3 rounded bg-blue-500')
                    ui.label('Decouvert').classes('text-xs')
                with ui.row().classes('items-center gap-1'):
                    ui.element('div').classes('w-3 h-3 rounded bg-yellow-500')
                    ui.label('Analyse').classes('text-xs')
                with ui.row().classes('items-center gap-1'):
                    ui.element('div').classes('w-3 h-3 rounded bg-green-500')
                    ui.label('Signal').classes('text-xs')
                with ui.row().classes('items-center gap-1'):
                    ui.element('div').classes('w-3 h-3 rounded bg-purple-500')
                    ui.label('Position').classes('text-xs')

            journeys_container = ui.column().classes('gap-4 max-h-[600px] overflow-auto')

            def update_journeys():
                journeys_container.clear()
                with journeys_container:
                    if not state.symbol_journeys:
                        with ui.column().classes('items-center py-12'):
                            ui.label('🤖').classes('text-6xl mb-4')
                            ui.label('En attente des premieres analyses...').classes('muted text-lg')
                            ui.label('Lancez le bot pour voir le flux de decision en temps reel').classes('text-sm muted mt-2')
                    else:
                        sorted_journeys = sorted(
                            state.symbol_journeys.items(),
                            key=lambda x: x[1].get('last_update', ''),
                            reverse=True
                        )[:12]
                        for symbol, journey in sorted_journeys:
                            journey_card(symbol, journey)

            update_journeys()
            ui.button('Rafraichir', icon='refresh', on_click=update_journeys).props('flat').classes('mt-4')

        # Narrative Reports - NO AUTO-REFRESH (fixes bug)
        with ui.element('div').classes('card'):
            with ui.element('div').classes('section-title'):
                ui.element('div').classes('section-title-icon green').text = '📝'
                ui.label('Rapports Narratifs').classes('font-bold text-lg')
                ui.button('Rafraichir', icon='refresh', on_click=lambda: update_narratives()).props('flat').classes('ml-auto')

            narratives_container = ui.column().classes('gap-3 max-h-96 overflow-auto')

            def update_narratives():
                narratives_container.clear()
                with narratives_container:
                    if not state.narrative_reports:
                        with ui.column().classes('items-center py-8'):
                            ui.label('📄').classes('text-4xl mb-2')
                            ui.label('Aucun rapport disponible').classes('muted')
                            ui.label('Lancez une analyse pour generer des rapports').classes('text-xs muted mt-2')
                    else:
                        for symbol, report in state.narrative_reports.items():
                            with ui.expansion(f'📈 {symbol}', icon='analytics').classes('w-full bg-slate-700/50 rounded-xl'):
                                ui.markdown(report).classes('text-sm p-4')

            update_narratives()
            # NO AUTO-REFRESH TIMER HERE - this was the bug causing reports to close

        # Learning Stats
        with ui.element('div').classes('card'):
            with ui.element('div').classes('section-title'):
                ui.element('div').classes('section-title-icon cyan').text = '🎓'
                ui.label('Systeme d\'Apprentissage').classes('font-bold text-lg')

            learning_container = ui.column().classes('gap-4')

            def update_learning():
                update_learning_stats()
                learning_container.clear()
                with learning_container:
                    stats = state.learning_stats

                    # Learning steps
                    with ui.element('div').classes('card-flat'):
                        ui.label('📖 Cycle d\'Apprentissage').classes('font-semibold mb-4')

                        steps = [
                            ('1️⃣', 'Analyse', 'Chaque analyse genere un vecteur 80D', stats['total_vectors'] > 0),
                            ('2️⃣', 'Trade', 'Position ouverte si score >= 55', any(j.get('current_status') == 'bought' for j in state.symbol_journeys.values())),
                            ('3️⃣', 'Cloture', 'Trade ferme, resultat enregistre', stats['total_closed'] > 0),
                            ('4️⃣', 'Feedback', 'Vecteur marque avec outcome', stats['total_closed'] > 0),
                            ('5️⃣', 'Optimisation', 'Patterns gagnants identifies', stats['total_closed'] >= 5),
                        ]

                        for icon, name, desc, completed in steps:
                            with ui.element('div').classes('learning-step'):
                                ui.element('div').classes(f'learning-check {"done" if completed else "pending"}').text = '✓' if completed else '○'
                                with ui.column().classes('gap-0'):
                                    ui.label(f'{icon} {name}').classes(f'font-semibold {"" if completed else "muted"}')
                                    ui.label(desc).classes('text-xs muted')

                    # Stats cards
                    with ui.row().classes('gap-4 flex-wrap'):
                        with ui.element('div').classes('card-flat flex-1 min-w-[120px] text-center'):
                            ui.label('📊').classes('text-2xl')
                            ui.label(str(len(state.symbol_journeys))).classes('text-2xl font-bold info')
                            ui.label('Analyses').classes('text-xs muted')

                        with ui.element('div').classes('card-flat flex-1 min-w-[120px] text-center'):
                            ui.label('🧮').classes('text-2xl')
                            ui.label(str(stats['total_vectors'])).classes('text-2xl font-bold accent')
                            ui.label('Vecteurs').classes('text-xs muted')

                        with ui.element('div').classes('card-flat flex-1 min-w-[120px] text-center'):
                            ui.label('📈').classes('text-2xl')
                            ui.label(str(stats['total_closed'])).classes('text-2xl font-bold')
                            ui.label(f'{stats["wins"]}W / {stats["losses"]}L').classes('text-xs muted')

                        with ui.element('div').classes('card-flat flex-1 min-w-[120px] text-center'):
                            ui.label('🏆').classes('text-2xl')
                            wr = stats['win_rate']
                            wr_color = 'profit' if wr > 50 else ('warning' if wr > 0 else 'muted')
                            ui.label(f'{wr:.0f}%' if wr > 0 else '—').classes(f'text-2xl font-bold {wr_color}')

                    # Shadow Tracking Stats (Paper Trading Autonome)
                    if stats.get('shadow_total', 0) > 0:
                        ui.separator().classes('my-4')
                        ui.label('🔮 Paper Trading Autonome (Shadow Tracker)').classes('font-semibold mb-3')

                        with ui.row().classes('gap-3 flex-wrap'):
                            with ui.element('div').classes('card-flat flex-1 min-w-[100px] text-center'):
                                ui.label('📥').classes('text-xl')
                                ui.label(str(stats['shadow_total'])).classes('text-xl font-bold')
                                ui.label('Tracked').classes('text-xs muted')

                            with ui.element('div').classes('card-flat flex-1 min-w-[100px] text-center'):
                                ui.label('✅').classes('text-xl')
                                ui.label(str(stats['shadow_verified'])).classes('text-xl font-bold profit')
                                ui.label('Verifies').classes('text-xs muted')

                            with ui.element('div').classes('card-flat flex-1 min-w-[100px] text-center'):
                                ui.label('⏳').classes('text-xl')
                                ui.label(str(stats['shadow_pending'])).classes('text-xl font-bold warning')
                                ui.label('En cours').classes('text-xs muted')

                            with ui.element('div').classes('card-flat flex-1 min-w-[100px] text-center'):
                                s_wr = stats['shadow_win_rate']
                                wr_col = 'profit' if s_wr >= 50 else ('warning' if s_wr > 30 else 'loss')
                                ui.label('📊').classes('text-xl')
                                ui.label(f'{s_wr:.0f}%' if s_wr > 0 else '—').classes(f'text-xl font-bold {wr_col}')
                                ui.label('Win Rate').classes('text-xs muted')

                            with ui.element('div').classes('card-flat flex-1 min-w-[100px] text-center'):
                                pf = stats['shadow_profit_factor']
                                pf_col = 'profit' if pf >= 1.5 else ('warning' if pf >= 1 else 'loss')
                                ui.label('💰').classes('text-xl')
                                ui.label(f'{pf:.2f}' if pf > 0 else '—').classes(f'text-xl font-bold {pf_col}')
                                ui.label('Profit Factor').classes('text-xs muted')

                        # Poids optimises
                        weights = stats.get('shadow_weights', {})
                        if weights:
                            ui.label('Poids Optimises:').classes('text-xs muted mt-2')
                            with ui.row().classes('gap-2'):
                                for pillar, weight in weights.items():
                                    ui.badge(f'{pillar[:4]}: {weight:.0%}').classes('text-xs')

                    # Status
                    with ui.element('div').classes('card-flat mt-2'):
                        if stats['total_closed'] == 0:
                            ui.label('⏳ En attente du premier trade cloture').classes('warning')
                            ui.label('Le bot analyse mais ne peut pas encore apprendre sans historique').classes('text-xs muted mt-1')
                        elif stats['total_closed'] < 5:
                            ui.label(f'📥 Collecte: {stats["total_closed"]}/5 trades').classes('warning')
                            ui.label('5 trades minimum pour identifier des patterns').classes('text-xs muted mt-1')
                        else:
                            ui.label('✅ Apprentissage actif').classes('profit')
                            insight = '💡 Strategie profitable' if stats['wins'] > stats['losses'] else '⚠️ Ajustement necessaire'
                            ui.label(insight).classes('text-xs mt-1')

            update_learning()
            ui.timer(30, update_learning)


@ui.page('/learning')
def learning_page():
    """Learning Dashboard - FINARY STYLE with auto-learning visualization"""
    ui.add_head_html(STYLES)
    nav_header()

    # Calculate learning metrics
    update_learning_stats()
    stats = state.learning_stats

    # Calculate additional metrics
    lessons_learned = []
    missed_opportunities = []
    weight_evolution = {
        'technical': 25,
        'fundamental': 25,
        'sentiment': 25,
        'news': 25
    }

    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-6'):
        # Header
        with ui.row().classes('justify-between items-center w-full'):
            with ui.column().classes('gap-1'):
                ui.label('🧠 Apprentissage & Optimisation').classes('text-3xl font-bold')
                ui.label('Auto-correction basée sur les résultats des trades').classes('text-sm text-slate-400')
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')

        # Top Metrics - Finary Style
        with ui.row().classes('w-full gap-4 flex-wrap'):
            # Total Vectors
            with ui.element('div').classes('flex-1 min-w-[180px] learning-metric-card'):
                ui.label('🧮').classes('text-3xl')
                ui.label(str(stats['total_vectors'])).classes('learning-metric-value text-purple-400')
                ui.label('Vecteurs stockés').classes('learning-metric-label')
                ui.label('80 dimensions chacun').classes('text-xs text-slate-500 mt-1')

            # Win Rate
            with ui.element('div').classes('flex-1 min-w-[180px] learning-metric-card'):
                wr = stats['win_rate']
                wr_color = 'profit' if wr >= 50 else 'loss'
                ui.label('🏆').classes('text-3xl')
                ui.label(f'{wr:.0f}%' if wr > 0 else '—').classes(f'learning-metric-value {wr_color}')
                ui.label('Win Rate').classes('learning-metric-label')
                ui.label(f'{stats["wins"]}W / {stats["losses"]}L').classes('text-xs text-slate-500 mt-1')

            # Total Closed
            with ui.element('div').classes('flex-1 min-w-[180px] learning-metric-card'):
                ui.label('📊').classes('text-3xl')
                ui.label(str(stats['total_closed'])).classes('learning-metric-value')
                ui.label('Trades clôturés').classes('learning-metric-label')
                ui.label('Feedback disponible').classes('text-xs text-slate-500 mt-1')

            # Corrections Applied
            with ui.element('div').classes('flex-1 min-w-[180px] learning-metric-card'):
                ui.label('🔧').classes('text-3xl')
                corrections = len(lessons_learned)
                ui.label(str(corrections)).classes('learning-metric-value text-blue-400')
                ui.label('Corrections').classes('learning-metric-label')
                ui.label('Appliquées').classes('text-xs text-slate-500 mt-1')

        # Two columns layout
        with ui.row().classes('w-full gap-6 flex-wrap'):
            # Left column - Weight Evolution
            with ui.column().classes('flex-1 min-w-[400px] gap-4'):
                # Weight Evolution Card
                with ui.element('div').classes('card'):
                    ui.label('⚖️ Évolution des Poids (4 Piliers)').classes('font-bold text-lg mb-4')
                    ui.label('Les poids s\'ajustent automatiquement selon la performance des trades').classes('text-xs text-slate-400 mb-4')

                    pillars_config = [
                        ('Technical', 'technical', '#3B82F6', '🔧'),
                        ('Fundamental', 'fundamental', '#8B5CF6', '📊'),
                        ('Sentiment', 'sentiment', '#EC4899', '💬'),
                        ('News', 'news', '#22D3EE', '📰'),
                    ]

                    for label, key, color, icon in pillars_config:
                        weight = weight_evolution.get(key, 25)
                        delta = 0  # Will be calculated from historical data

                        with ui.element('div').classes('weight-bar'):
                            ui.label(f'{icon} {label}').classes('weight-bar-label')
                            with ui.element('div').classes('weight-bar-track'):
                                ui.element('div').classes('weight-bar-fill').style(f'width: {weight}%; background: {color}')
                            with ui.row().classes('items-center gap-2'):
                                ui.label(f'{weight}%').classes('font-mono text-sm font-bold')
                                if delta != 0:
                                    delta_color = 'profit' if delta > 0 else 'loss'
                                    ui.label(f'{delta:+.0f}%').classes(f'text-xs {delta_color}')

                    with ui.element('div').classes('mt-4 p-3 bg-slate-800/50 rounded-lg'):
                        ui.label('📌 Actuellement: Poids égaux (25% chacun)').classes('text-sm text-slate-300')
                        ui.label('Les poids s\'ajusteront après accumulation de 10+ trades clôturés').classes('text-xs text-slate-500 mt-1')

                # Learning Cycle Status
                with ui.element('div').classes('card'):
                    ui.label('🔄 Cycle d\'Apprentissage').classes('font-bold text-lg mb-4')

                    steps = [
                        ('1️⃣', 'Analyse', 'Génère un vecteur 80D par symbole', stats['total_vectors'] > 0),
                        ('2️⃣', 'Signal', 'Recommandation si score ≥ 55', len([j for j in state.symbol_journeys.values() if j.get('current_score', 0) >= 55]) > 0),
                        ('3️⃣', 'Exécution', 'Ordre passé via IBKR', any(j.get('current_status') == 'bought' for j in state.symbol_journeys.values())),
                        ('4️⃣', 'Outcome', 'Trade clôturé, P&L calculé', stats['total_closed'] > 0),
                        ('5️⃣', 'Feedback', 'Vecteur + outcome = training data', stats['total_closed'] > 0),
                        ('6️⃣', 'Optimisation', 'Poids ajustés (min 5 trades)', stats['total_closed'] >= 5),
                    ]

                    for icon, name, desc, completed in steps:
                        completed_class = 'border-green-500 bg-green-900/20' if completed else 'border-slate-600 bg-slate-800/50'
                        check_class = 'text-green-400' if completed else 'text-slate-500'

                        with ui.element('div').classes(f'flex items-start gap-3 p-3 rounded-lg border {completed_class} mb-2'):
                            ui.label('✓' if completed else '○').classes(f'text-lg {check_class}')
                            with ui.column().classes('gap-0'):
                                ui.label(f'{icon} {name}').classes('font-semibold')
                                ui.label(desc).classes('text-xs text-slate-400')

            # Right column - Lessons & Missed Opportunities
            with ui.column().classes('flex-1 min-w-[400px] gap-4'):
                # Lessons Learned
                with ui.element('div').classes('card'):
                    ui.label('📚 Leçons Apprises').classes('font-bold text-lg mb-4')

                    if not lessons_learned:
                        with ui.element('div').classes('p-6 text-center bg-slate-800/50 rounded-xl'):
                            ui.label('📖').classes('text-4xl mb-2')
                            ui.label('Aucune leçon enregistrée').classes('text-slate-400')
                            ui.label('Les leçons apparaîtront après l\'analyse des trades clôturés').classes('text-xs text-slate-500 mt-1')
                    else:
                        with ui.column().classes('gap-2 max-h-64 overflow-auto'):
                            for lesson in lessons_learned[:5]:
                                with ui.element('div').classes('lesson-item'):
                                    ui.label(lesson.get('date', '')).classes('lesson-date')
                                    ui.label(lesson.get('text', '')).classes('lesson-text')

                # Missed Opportunities
                with ui.element('div').classes('card'):
                    ui.label('💡 Opportunités Manquées').classes('font-bold text-lg mb-4')
                    ui.label('Valeurs qui ont performé mais que le bot a sous-évaluées').classes('text-xs text-slate-400 mb-4')

                    # Find watch signals that performed well (simulated)
                    watch_signals = [
                        (s, j) for s, j in state.symbol_journeys.items()
                        if 40 <= j.get('current_score', 0) < 55
                    ]

                    if not watch_signals:
                        with ui.element('div').classes('p-4 text-center bg-slate-800/50 rounded-xl'):
                            ui.label('Aucune opportunité manquée détectée').classes('text-slate-400')
                    else:
                        for symbol, journey in watch_signals[:3]:
                            score = journey.get('current_score', 0)
                            with ui.element('div').classes('missed-opportunity'):
                                with ui.column().classes('gap-0'):
                                    ui.label(symbol).classes('font-bold')
                                    ui.label(f'Score: {score}/100 - Seuil: 55').classes('text-xs text-yellow-400')
                                ui.label('À surveiller').classes('text-xs text-slate-400')

                # Performance Insights
                with ui.element('div').classes('card'):
                    ui.label('📈 Insights Performance').classes('font-bold text-lg mb-4')

                    if stats['total_closed'] < 3:
                        with ui.element('div').classes('p-4 bg-slate-800/50 rounded-xl'):
                            ui.label('⏳ Accumulation de données en cours...').classes('text-slate-300')
                            ui.label(f'{stats["total_closed"]}/3 trades minimum pour générer des insights').classes('text-xs text-slate-500 mt-1')
                    else:
                        # Generate insights based on stats
                        insights = []
                        if stats['win_rate'] >= 60:
                            insights.append(('✅', 'Stratégie profitable', 'Win Rate > 60%'))
                        elif stats['win_rate'] >= 50:
                            insights.append(('⚡', 'Performance équilibrée', 'Win Rate ~50%'))
                        else:
                            insights.append(('⚠️', 'Ajustement nécessaire', 'Win Rate < 50%'))

                        for icon, title, desc in insights:
                            with ui.element('div').classes('flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg mb-2'):
                                ui.label(icon).classes('text-2xl')
                                with ui.column().classes('gap-0'):
                                    ui.label(title).classes('font-semibold')
                                    ui.label(desc).classes('text-xs text-slate-400')

        # Action Buttons
        with ui.row().classes('w-full justify-center gap-4 mt-4'):
            ui.button('Forcer Re-calibration', icon='refresh').classes('btn-finary-secondary').props('disable')
            ui.button('Exporter Rapport', icon='download').classes('btn-finary-primary').props('disable')


# =============================================================================
# TRAINING PAGE - Model Pre-training Interface
# =============================================================================

@ui.page('/training')
def training_page():
    """Training Administration - Configure and run model pre-training"""
    ui.add_head_html(STYLES)
    nav_header()

    # Training state
    training_state = {
        'running': False,
        'progress': 0,
        'status': 'idle',
        'current_symbol': '',
        'signals_found': 0,
        'results': None
    }

    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-6'):
        # Header
        with ui.element('div').classes('text-center mb-6'):
            ui.label('🎓 Entraînement du Modèle').classes('text-3xl font-bold')
            ui.label('Pré-entraîner le modèle sur données historiques pour optimiser les poids').classes('text-slate-400')

        # Two columns layout
        with ui.row().classes('w-full gap-6'):
            # Left: Configuration
            with ui.column().classes('flex-1 gap-4'):
                with ui.element('div').classes('card'):
                    ui.label('⚙️ Configuration').classes('text-xl font-bold mb-4')

                    # Period selection
                    ui.label('📅 Période historique').classes('font-semibold mt-2 text-white')
                    period_select = ui.select(
                        options={
                            3: '3 mois (rapide)',
                            6: '6 mois (recommandé)',
                            12: '1 an (complet)',
                            24: '2 ans (approfondi)',
                            60: '5 ans (extensif)',
                            120: '10 ans (maximum)'
                        },
                        value=6,
                        label='Mois de données'
                    ).classes('w-full').props('dark dense options-dense')

                    # Symbols count (0 = tous les symboles disponibles)
                    ui.label('📊 Nombre de symboles').classes('font-semibold mt-4 text-white')
                    symbols_slider = ui.slider(min=0, max=5000, step=50, value=0).classes('w-full').props('color=purple label-always')
                    symbols_label = ui.label('Tous (5000+)').classes('text-sm text-center text-white')

                    def update_symbols_label(e):
                        val = int(e.args) if e.args else 0
                        if val == 0:
                            symbols_label.set_text('Tous (illimité)')
                        else:
                            symbols_label.set_text(f'{val} symboles')
                    symbols_slider.on('update:model-value', update_symbols_label)

                    # Universe selection
                    ui.label('🌍 Univers de stocks').classes('font-semibold mt-4 text-white')
                    universe_select = ui.select(
                        options={
                            'us_top': 'US Top (NASDAQ + SP500)',
                            'us_all': 'US Complet (NYSE + NASDAQ + AMEX)',
                            'europe': 'Europe (CAC40 + DAX + FTSE)',
                            'global': 'Global (US + Europe)'
                        },
                        value='us_top',
                        label='Marché'
                    ).classes('w-full').props('dark dense options-dense')

                    # Parameters the system will optimize automatically
                    with ui.expansion('🔬 Paramètres à optimiser (automatique)', icon='auto_fix_high').classes('w-full mt-4'):
                        ui.label('Le système testera automatiquement toutes les combinaisons pour trouver les valeurs optimales :').classes('text-xs text-slate-400 mb-3')

                        with ui.element('div').classes('grid grid-cols-2 gap-3'):
                            # Threshold range
                            with ui.element('div').classes('card-flat p-3'):
                                ui.label('🎯 Seuil de score').classes('text-xs text-slate-400')
                                ui.label('45 → 70').classes('text-lg font-bold text-purple-400')
                                ui.label('par pas de 5').classes('text-xs text-slate-500')

                            # Take Profit range
                            with ui.element('div').classes('card-flat p-3'):
                                ui.label('📈 Take Profit').classes('text-xs text-slate-400')
                                ui.label('+5% → +25%').classes('text-lg font-bold text-green-400')
                                ui.label('par pas de 5%').classes('text-xs text-slate-500')

                            # Stop Loss range
                            with ui.element('div').classes('card-flat p-3'):
                                ui.label('📉 Stop Loss').classes('text-xs text-slate-400')
                                ui.label('-3% → -10%').classes('text-lg font-bold text-red-400')
                                ui.label('par pas de 1%').classes('text-xs text-slate-500')

                            # Trailing Stop
                            with ui.element('div').classes('card-flat p-3'):
                                ui.label('🔄 Trailing Stop').classes('text-xs text-slate-400')
                                ui.label('Oui / Non').classes('text-lg font-bold text-cyan-400')
                                ui.label('+ 3%, 5%, 7%').classes('text-xs text-slate-500')

                        ui.label('💡 Le système évaluera ~500 combinaisons pour maximiser le profit factor et le win rate.').classes('text-xs text-slate-400 mt-3')

                    # Estimation
                    with ui.element('div').classes('card-flat mt-4 p-3'):
                        ui.label('📊 Estimation').classes('font-semibold text-white')
                        estimation_text = ui.label('').classes('text-sm text-slate-400')

                        # Nombre de symboles par univers
                        universe_sizes = {
                            'us_top': 200,      # SP500 + NASDAQ top
                            'us_all': 5500,     # NYSE + NASDAQ complets
                            'europe': 100,      # Europe
                            'global': 3500      # US + Europe
                        }

                        def update_estimation():
                            months = period_select.value
                            symbols = int(symbols_slider.value) if symbols_slider.value else 0
                            universe = universe_select.value

                            # Si 0 symboles, utiliser la taille de l'univers
                            if symbols == 0:
                                symbols = universe_sizes.get(universe, 200)
                                symbols_text = f"tous ({symbols})"
                            else:
                                symbols_text = str(symbols)

                            # Estimation: ~2 secondes par symbole par mois
                            total_ops = symbols * (months / 6)
                            minutes = total_ops / 30  # ~30 symboles/minute
                            if minutes < 1:
                                estimation_text.set_text(f'~{int(minutes*60)} secondes pour {symbols_text} symboles sur {months} mois')
                            elif minutes < 60:
                                estimation_text.set_text(f'~{int(minutes)} minutes pour {symbols_text} symboles sur {months} mois')
                            else:
                                hours = minutes / 60
                                estimation_text.set_text(f'~{hours:.1f} heures pour {symbols_text} symboles sur {months} mois')

                        # Initial update
                        update_estimation()

                        period_select.on('update:model-value', lambda e: update_estimation())
                        symbols_slider.on('update:model-value', lambda e: update_estimation())
                        universe_select.on('update:model-value', lambda e: update_estimation())

            # Right: Status & Results
            with ui.column().classes('flex-1 gap-4'):
                # Progress card
                progress_card = ui.element('div').classes('card')
                with progress_card:
                    ui.label('📈 Progression').classes('text-xl font-bold mb-4')

                    status_label = ui.label('En attente...').classes('text-lg')
                    current_symbol_label = ui.label('').classes('text-sm text-slate-400')

                    progress_bar = ui.linear_progress(value=0, show_value=False).classes('mt-4')
                    progress_text = ui.label('0%').classes('text-center text-sm mt-1')

                    # Stats en temps reel
                    with ui.row().classes('gap-4 mt-4 justify-center flex-wrap'):
                        signals_stat = ui.element('div').classes('text-center min-w-[70px]')
                        with signals_stat:
                            signals_count_label = ui.label('0').classes('text-2xl font-bold info')
                            ui.label('Signaux').classes('text-xs text-slate-400')

                        wins_stat = ui.element('div').classes('text-center min-w-[70px]')
                        with wins_stat:
                            wins_label = ui.label('0').classes('text-2xl font-bold profit')
                            ui.label('Gagnants').classes('text-xs text-slate-400')

                        losses_stat = ui.element('div').classes('text-center min-w-[70px]')
                        with losses_stat:
                            losses_label = ui.label('0').classes('text-2xl font-bold loss')
                            ui.label('Perdants').classes('text-xs text-slate-400')

                        winrate_stat = ui.element('div').classes('text-center min-w-[70px]')
                        with winrate_stat:
                            winrate_label = ui.label('-').classes('text-2xl font-bold')
                            ui.label('Win Rate').classes('text-xs text-slate-400')

                # Live Feed - Derniers signaux
                live_feed_card = ui.element('div').classes('card mt-4')
                with live_feed_card:
                    ui.label('📡 Flux en Direct').classes('text-lg font-bold mb-2')
                    # Légende des piliers
                    with ui.row().classes('gap-3 mb-3 flex-wrap'):
                        with ui.row().classes('items-center gap-1'):
                            with ui.element('div').classes('pillar-indicator tech'):
                                ui.label('🔧').classes('text-xs')
                            ui.label('Tech').classes('text-xs text-slate-400')
                        with ui.row().classes('items-center gap-1'):
                            with ui.element('div').classes('pillar-indicator fund'):
                                ui.label('📊').classes('text-xs')
                            ui.label('Fund').classes('text-xs text-slate-400')
                        with ui.row().classes('items-center gap-1'):
                            with ui.element('div').classes('pillar-indicator sent'):
                                ui.label('💬').classes('text-xs')
                            ui.label('Sent').classes('text-xs text-slate-400')
                        with ui.row().classes('items-center gap-1'):
                            with ui.element('div').classes('pillar-indicator news'):
                                ui.label('📰').classes('text-xs')
                            ui.label('News').classes('text-xs text-slate-400')
                    # Container pour les signaux
                    live_feed_container = ui.column().classes('gap-1 max-h-[280px] overflow-y-auto')
                    with live_feed_container:
                        ui.label('En attente du démarrage...').classes('text-sm text-slate-400 italic')

                # Results card (hidden initially)
                results_card = ui.element('div').classes('card hidden')
                results_container = ui.column().classes('gap-3')

        # Action buttons
        with ui.row().classes('w-full justify-center gap-4 mt-6'):
            async def start_training():
                """Launch the pre-training process"""
                if training_state['running']:
                    return

                training_state['running'] = True
                training_state['progress'] = 0
                training_state['signals_found'] = 0
                status_label.set_text('Initialisation...')
                progress_bar.set_value(0)
                results_card.classes(remove='hidden', add='hidden')

                # Reset live feed et stats
                live_feed_container.clear()
                signals_count_label.set_text('0')
                wins_label.set_text('0')
                losses_label.set_text('0')
                winrate_label.set_text('-')
                winrate_label.classes(remove='profit loss warning')

                # Etat partage pour les mises a jour (modifie par callback, lu par timer)
                live_state = {
                    'current_symbol': '',
                    'progress_pct': 0,
                    'symbols_done': 0,
                    'symbols_total': 0,
                    'signals_total': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'last_signals': [],  # Liste des derniers signaux
                    'updated': False      # Flag pour savoir si update necessaire
                }

                def on_progress(event_type: str, data: dict):
                    """Callback pour mise a jour de l'etat partage (appele depuis le backtest)"""
                    try:
                        if event_type == 'symbol':
                            live_state['current_symbol'] = data['symbol']
                            live_state['symbols_done'] = data['current']
                            live_state['symbols_total'] = data['total']
                            live_state['progress_pct'] = data['progress_pct']
                            live_state['updated'] = True

                        elif event_type == 'signal':
                            stats = data.get('stats', {})
                            live_state['signals_total'] = stats.get('total', 0)
                            live_state['wins'] = stats.get('wins', 0)
                            live_state['losses'] = stats.get('losses', 0)
                            live_state['win_rate'] = stats.get('win_rate', 0)

                            # Ajouter le signal a la liste
                            pillar_icons = {'TECH': '🔧', 'FUND': '📊', 'SENT': '💬', 'NEWS': '📰'}
                            pillar_colors = {'TECH': 'text-blue-400', 'FUND': 'text-purple-400',
                                           'SENT': 'text-pink-400', 'NEWS': 'text-cyan-400'}
                            dominant = data.get('dominant', 'TECH')

                            signal_entry = {
                                'symbol': data['symbol'],
                                'date': data['date'],
                                'score': data['score'],
                                'pnl': data.get('pnl', 0),
                                'is_winner': data.get('is_winner', False),
                                'dominant': dominant,
                                'icon': pillar_icons.get(dominant, '📊'),
                                'color': pillar_colors.get(dominant, 'text-slate-400')
                            }
                            live_state['last_signals'].insert(0, signal_entry)
                            if len(live_state['last_signals']) > 15:
                                live_state['last_signals'].pop()

                            live_state['updated'] = True

                    except Exception as e:
                        logger.debug(f"Progress callback error: {e}")

                # Timer pour mettre a jour l'UI toutes les 2 secondes
                async def update_ui():
                    """Met a jour l'UI si l'etat a change"""
                    if not live_state['updated']:
                        return

                    live_state['updated'] = False

                    # Mise a jour progression
                    if live_state['symbols_total'] > 0:
                        pct = 0.05 + (live_state['progress_pct'] / 100 * 0.50)
                        current_symbol_label.set_text(
                            f"📊 {live_state['current_symbol']} ({live_state['symbols_done']}/{live_state['symbols_total']})"
                        )
                        progress_bar.set_value(pct)
                        progress_text.set_text(f"{int(pct * 100)}%")

                    # Mise a jour stats
                    signals_count_label.set_text(str(live_state['signals_total']))
                    wins_label.set_text(str(live_state['wins']))
                    losses_label.set_text(str(live_state['losses']))

                    wr = live_state['win_rate']
                    winrate_label.set_text(f"{wr:.0f}%")
                    winrate_label.classes(remove='profit loss warning')
                    if wr >= 55:
                        winrate_label.classes(add='profit')
                    elif wr >= 45:
                        winrate_label.classes(add='warning')
                    elif wr > 0:
                        winrate_label.classes(add='loss')

                    # Rebuild le flux des signaux
                    if live_state['last_signals']:
                        live_feed_container.clear()
                        with live_feed_container:
                            for sig in live_state['last_signals']:
                                result_icon = '✅' if sig['is_winner'] else '❌'
                                pnl_color = 'profit' if sig['pnl'] >= 0 else 'loss'
                                pnl_str = f"+{sig['pnl']:.1f}%" if sig['pnl'] >= 0 else f"{sig['pnl']:.1f}%"
                                signal_class = 'winner' if sig['is_winner'] else 'loser'
                                pillar_class = sig['dominant'].lower()

                                with ui.row().classes(f'live-feed-signal {signal_class} w-full'):
                                    with ui.element('div').classes(f'pillar-indicator {pillar_class}'):
                                        ui.label(sig['icon']).classes('text-xs')
                                    ui.label(sig['symbol']).classes('font-mono font-bold text-sm min-w-[55px]')
                                    ui.label(f"{sig['score']:.0f}").classes('text-xs text-slate-400 min-w-[25px] text-center')
                                    ui.label(sig['dominant']).classes(f"text-xs {sig['color']} min-w-[35px]")
                                    ui.label(sig['date']).classes('text-xs text-slate-500 flex-1')
                                    ui.label(result_icon).classes('text-sm')
                                    ui.label(pnl_str).classes(f'font-mono text-sm font-semibold {pnl_color} min-w-[55px] text-right')

                # Creer le timer (toutes les 2 secondes)
                ui_timer = ui.timer(2.0, update_ui)

                try:
                    from scripts.pretrain_model import ModelPretrainer
                    import asyncio
                    from concurrent.futures import ThreadPoolExecutor

                    months = period_select.value
                    symbols = int(symbols_slider.value)
                    universe = universe_select.value

                    pretrainer = ModelPretrainer(
                        months_back=months,
                        max_symbols=symbols,  # 0 = pas de limite
                        universe=universe,
                        on_progress=on_progress  # Callback temps reel
                    )

                    # Phase 1: Get universe
                    status_label.set_text('Chargement de l\'univers...')
                    await asyncio.sleep(0.1)
                    stock_universe = pretrainer.get_stock_universe()
                    num_symbols = len(stock_universe)
                    status_label.set_text(f'{num_symbols} symboles chargés')
                    progress_bar.set_value(0.05)
                    progress_text.set_text('5%')

                    # Phase 2: Run backtest with live updates
                    status_label.set_text('Backtest en cours...')
                    current_symbol_label.set_text('Démarrage du backtest...')

                    # Run backtest dans un thread pour ne pas bloquer l'event loop
                    def run_backtest_sync():
                        """Wrapper synchrone pour le backtest"""
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(pretrainer.run_backtest())
                        finally:
                            new_loop.close()

                    # Creer un executor qui reste ouvert pour toutes les operations
                    executor = ThreadPoolExecutor(max_workers=1)
                    loop = asyncio.get_event_loop()

                    try:
                        # Phase 2: Backtest
                        await loop.run_in_executor(executor, run_backtest_sync)

                        training_state['signals_found'] = len(pretrainer.signals)
                        progress_bar.set_value(0.6)
                        progress_text.set_text('60%')
                        current_symbol_label.set_text(f'✅ Backtest terminé - {len(pretrainer.signals)} signaux analysés')
                        await asyncio.sleep(0.1)  # Laisser l'UI se mettre a jour

                        # Phase 3: Optimize (dans un thread)
                        status_label.set_text('🔧 Optimisation des poids...')
                        current_symbol_label.set_text('Grid search sur Tech/Fund/Sent/News...')
                        await asyncio.sleep(0.1)
                        optimal_weights = await loop.run_in_executor(executor, pretrainer.optimize_weights)
                        progress_bar.set_value(0.8)
                        progress_text.set_text('80%')
                        await asyncio.sleep(0.1)

                        # Phase 4: Find threshold (dans un thread)
                        status_label.set_text('🎯 Recherche du seuil optimal...')
                        current_symbol_label.set_text('Test des seuils 40, 45, 50, 55, 60...')
                        await asyncio.sleep(0.1)
                        threshold, final_stats = await loop.run_in_executor(executor, pretrainer.find_optimal_threshold)
                        progress_bar.set_value(0.85)
                        progress_text.set_text('85%')
                        await asyncio.sleep(0.1)

                        # Phase 5: Optimize exit parameters (dans un thread)
                        status_label.set_text('💰 Optimisation TP/SL/Trailing...')
                        current_symbol_label.set_text('Grid search sur les paramètres de sortie...')
                        await asyncio.sleep(0.1)
                        exit_params = await loop.run_in_executor(executor, lambda: pretrainer.optimize_exit_params(threshold))
                        progress_bar.set_value(0.95)
                        progress_text.set_text('95%')

                    finally:
                        executor.shutdown(wait=False)

                    # Phase 6: Save
                    status_label.set_text('💾 Sauvegarde des résultats...')
                    current_symbol_label.set_text('Écriture dans data/shadow_tracking/')
                    pretrainer.save_results(final_stats, threshold)
                    progress_bar.set_value(1.0)
                    progress_text.set_text('100%')

                    # Show results
                    status_label.set_text('✅ Entraînement terminé!')
                    wr_final = final_stats.get('win_rate', 0)
                    pf_final = final_stats.get('profit_factor', 0)
                    current_symbol_label.set_text(f'🏆 Win Rate: {wr_final:.1f}% | Profit Factor: {pf_final:.2f} | Seuil: {threshold}')
                    training_state['results'] = {
                        'weights': optimal_weights,
                        'threshold': threshold,
                        'exit_params': exit_params,
                        'stats': final_stats,
                        'signals': len(pretrainer.signals)
                    }

                    # Display results
                    results_card.classes(remove='hidden')
                    results_container.clear()
                    with results_container:
                        ui.label('🎯 Résultats de l\'Entraînement').classes('text-xl font-bold')

                        # Stats
                        with ui.row().classes('gap-4 flex-wrap mt-4'):
                            with ui.element('div').classes('card-flat text-center flex-1 min-w-[100px]'):
                                ui.label('📊').classes('text-2xl')
                                ui.label(str(final_stats['total_signals'])).classes('text-xl font-bold')
                                ui.label('Signaux testés').classes('text-xs text-slate-400')

                            with ui.element('div').classes('card-flat text-center flex-1 min-w-[100px]'):
                                wr = final_stats['win_rate']
                                wr_col = 'profit' if wr >= 55 else ('warning' if wr >= 45 else 'loss')
                                ui.label('🏆').classes('text-2xl')
                                ui.label(f'{wr:.1f}%').classes(f'text-xl font-bold {wr_col}')
                                ui.label('Win Rate').classes('text-xs text-slate-400')

                            with ui.element('div').classes('card-flat text-center flex-1 min-w-[100px]'):
                                pf = final_stats['profit_factor']
                                pf_col = 'profit' if pf >= 1.5 else ('warning' if pf >= 1 else 'loss')
                                ui.label('💰').classes('text-2xl')
                                ui.label(f'{pf:.2f}').classes(f'text-xl font-bold {pf_col}')
                                ui.label('Profit Factor').classes('text-xs text-slate-400')

                            with ui.element('div').classes('card-flat text-center flex-1 min-w-[100px]'):
                                ui.label('🎯').classes('text-2xl')
                                ui.label(str(threshold)).classes('text-xl font-bold accent')
                                ui.label('Seuil Optimal').classes('text-xs text-slate-400')

                        # Optimized weights
                        ui.label('📐 Poids Optimisés').classes('font-semibold mt-4')
                        with ui.row().classes('gap-3 flex-wrap'):
                            colors = {'technical': 'bg-blue-500', 'fundamental': 'bg-purple-500',
                                     'sentiment': 'bg-pink-500', 'news': 'bg-cyan-500'}
                            for pillar, weight in optimal_weights.items():
                                with ui.element('div').classes(f'card-flat p-3 text-center min-w-[80px]'):
                                    ui.label(pillar.capitalize()[:4]).classes('text-xs text-slate-400')
                                    ui.label(f'{weight:.0%}').classes('text-lg font-bold')
                                    ui.linear_progress(value=weight).classes(f'mt-1 {colors.get(pillar, "")}')

                        # Optimized Exit Strategy
                        ui.label('🎯 Stratégie de Sortie Optimisée').classes('font-semibold mt-4 text-white')
                        with ui.row().classes('gap-3 flex-wrap'):
                            with ui.element('div').classes('card-flat p-3 text-center min-w-[80px]'):
                                ui.label('📈 Take Profit').classes('text-xs text-slate-400')
                                ui.label(f"+{exit_params['take_profit_pct']*100:.0f}%").classes('text-lg font-bold text-green-400')

                            with ui.element('div').classes('card-flat p-3 text-center min-w-[80px]'):
                                ui.label('📉 Stop Loss').classes('text-xs text-slate-400')
                                ui.label(f"-{exit_params['stop_loss_pct']*100:.0f}%").classes('text-lg font-bold text-red-400')

                            with ui.element('div').classes('card-flat p-3 text-center min-w-[80px]'):
                                ui.label('🔄 Trailing').classes('text-xs text-slate-400')
                                if exit_params.get('use_trailing'):
                                    ui.label(f"{exit_params['trailing_stop_pct']*100:.0f}%").classes('text-lg font-bold text-cyan-400')
                                else:
                                    ui.label('Non').classes('text-lg font-bold text-slate-500')

                        # Expected performance
                        ui.label('📈 Performance Attendue').classes('font-semibold mt-4 text-white')
                        with ui.element('div').classes('card-flat p-3'):
                            ui.label(f"• Gain moyen gagnant: +{final_stats['avg_winner']:.1f}%").classes('text-sm profit')
                            ui.label(f"• Perte moyenne perdant: {final_stats['avg_loser']:.1f}%").classes('text-sm loss')
                            ui.label(f"• Espérance par trade: {final_stats['expectancy']:.2f}%").classes('text-sm text-white')

                        # Apply button
                        ui.button('Appliquer ces Paramètres', icon='check',
                                 on_click=lambda: ui.notify('Paramètres appliqués!', type='positive')
                                 ).classes('btn-finary-primary mt-4')

                    add_log(f"Entraînement terminé: WR={final_stats['win_rate']:.1f}%, PF={final_stats['profit_factor']:.2f}")

                except Exception as e:
                    status_label.set_text(f'❌ Erreur: {str(e)[:50]}')
                    current_symbol_label.set_text('Entraînement interrompu')
                    logger.error(f"Training error: {e}")
                    import traceback
                    traceback.print_exc()

                finally:
                    training_state['running'] = False
                    # Arreter le timer de mise a jour
                    try:
                        ui_timer.deactivate()
                    except:
                        pass

            start_btn = ui.button('🚀 Lancer l\'Entraînement', icon='play_arrow',
                                 on_click=start_training).classes('btn-finary-primary text-lg px-8 py-3')

            ui.button('Retour', icon='arrow_back',
                     on_click=lambda: ui.navigate.to('/learning')).classes('btn-finary-secondary')

        # Info section
        with ui.element('div').classes('card mt-6'):
            ui.label('ℹ️ Comment fonctionne l\'entraînement?').classes('font-bold mb-2')
            with ui.element('div').classes('text-sm text-slate-400'):
                ui.markdown('''
**1. Simulation Historique**
- Pour chaque symbole et chaque semaine de la période
- Calcule les 4 scores (Technical, Fundamental, Sentiment, News)
- Génère des signaux BUY si score >= seuil

**2. Vérification des Outcomes**
- Pour chaque signal, vérifie le prix à T+5, T+10, T+20 jours
- Enregistre le gain max et le drawdown max pour chaque trade
- Stocke les données pour optimisation

**3. Optimisation des Poids des 4 Piliers**
- Teste ~125 combinaisons de poids (grid search)
- Trouve la combinaison qui maximise Win Rate × Robustesse
- Poids peuvent aller de 15% à 35% par pilier

**4. Recherche du Seuil Optimal**
- Teste les seuils de 45 à 70 par pas de 5

**5. Optimisation des Paramètres de Sortie** 🆕
- Teste ~200 combinaisons de TP (5-25%), SL (3-10%), Trailing (0-7%)
- Trouve la stratégie qui maximise le Profit Factor
- Sauvegarde tous les paramètres optimisés dans `data/shadow_tracking/`
''')


@ui.page('/alerts')
def alerts_page():
    """Alerts page with pagination"""
    ui.add_head_html(STYLES)
    nav_header()

    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-4'):
        with ui.row().classes('justify-between items-center'):
            ui.label('Alertes').classes('text-2xl font-bold')
            with ui.row().classes('gap-2'):
                ui.badge(f'{len(state.alerts)} alertes').classes('bg-indigo-500')
                ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')

        # Filter
        with ui.element('div').classes('card'):
            with ui.row().classes('gap-4 items-center flex-wrap'):
                filter_input = ui.input('Filtrer par symbole').classes('flex-1')
                signal_filter = ui.select(['Tous', 'STRONG_BUY', 'BUY', 'WATCH'], value='Tous', label='Signal')

        # Alerts list
        with ui.element('div').classes('card'):
            if not state.alerts:
                with ui.column().classes('items-center py-8'):
                    ui.label('🔔').classes('text-4xl mb-2')
                    ui.label('Aucune alerte').classes('muted')
            else:
                alerts_container = ui.column().classes('gap-3 max-h-[600px] overflow-auto')
                with alerts_container:
                    for alert in state.alerts[:50]:
                        alert_card(alert)


@ui.page('/settings')
def settings_page():
    """Settings page"""
    ui.add_head_html(STYLES)
    nav_header()

    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-4'):
        with ui.row().classes('justify-between items-center'):
            ui.label('Parametres').classes('text-2xl font-bold')
            ui.button('Retour', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')

        # Capital
        with ui.element('div').classes('card'):
            ui.label('💰 Capital de Trading').classes('font-bold text-lg mb-4')
            with ui.row().classes('gap-4 items-end'):
                capital_input = ui.number('Capital (EUR)', value=state.capital, min=100, max=1000000)

                def save_capital():
                    state.capital = capital_input.value
                    safe_notify('Capital mis a jour', type='positive')

                ui.button('Sauvegarder', icon='save', on_click=save_capital).classes('btn-primary')

        # API Status
        with ui.element('div').classes('card'):
            ui.label('🔌 Connexions API').classes('font-bold text-lg mb-4')

            apis = [
                ('Telegram', bool(os.getenv('TELEGRAM_BOT_TOKEN')), 'Notifications'),
                ('Grok (xAI)', bool(os.getenv('GROK_API_KEY')), 'Sentiment X/Twitter'),
                ('Gemini', bool(os.getenv('GOOGLE_AI_API_KEY')), 'Analyse News'),
                ('OpenRouter', bool(os.getenv('OPENROUTER_API_KEY')), 'Trend Discovery'),
                ('FMP', bool(os.getenv('FMP_API_KEY')), 'Donnees financieres'),
                ('IBKR', False, 'Execution ordres'),
            ]

            for name, connected, description in apis:
                with ui.row().classes('items-center gap-3 py-2'):
                    ui.element('span').classes(f'status-dot {"active" if connected else "inactive"}')
                    with ui.column().classes('flex-1 gap-0'):
                        ui.label(name).classes('font-semibold')
                        ui.label(description).classes('text-xs muted')
                    ui.label('Connecte' if connected else 'Non configure').classes(f'text-sm {"profit" if connected else "loss"}')

        # Logs
        with ui.expansion('📋 Logs Systeme', icon='article').classes('w-full card'):
            logs_area = ui.column().classes('gap-1 max-h-64 overflow-auto font-mono text-xs')

            def update_logs():
                logs_area.clear()
                with logs_area:
                    for log in state.logs[:50]:
                        ui.label(log).classes('text-slate-400')

            update_logs()
            ui.timer(5, update_logs)


# =============================================================================
# MAIN
# =============================================================================

if __name__ in {"__main__", "__mp_main__"}:
    print("=" * 50)
    print("   TradingBot V5 - Dashboard Professionnel")
    print("=" * 50)
    print()
    print("[INFO] Demarrage sur http://localhost:8080")
    print("[INFO] Ctrl+C pour arreter")
    print("=" * 50)

    ui.run(
        title='TradingBot V5',
        port=8080,
        reload=False,
        show=False
    )
