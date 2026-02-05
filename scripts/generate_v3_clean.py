#!/usr/bin/env python3
"""
Generate honest marketing charts for MIDAS.
Version 3 Clean - No emojis, professional text-based design.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Professional color palette
BACKGROUND = '#0f0f1a'
CARD_BG = '#1a1a2e'
MIDAS_GOLD = '#FFD700'
BLUE = '#4A90D9'
PURPLE = '#9B6DFF'
GREEN = '#00D084'
RED = '#FF4757'
CYAN = '#4ECDC4'
PINK = '#FF6B9D'
TEXT_COLOR = '#E8E8E8'

OUTPUT_DIR = '/root/tradingbot-github/data/charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.color'] = TEXT_COLOR

def chart_5_pillars():
    """Visual representation of the 5 pillars scoring system."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'MIDAS SCORING SYSTEM', fontsize=32, fontweight='bold',
            ha='center', va='top', color=MIDAS_GOLD, transform=ax.transAxes)
    ax.text(0.5, 0.88, '5 Independent AI Modules Analyze Every Stock', fontsize=16,
            ha='center', va='top', color='#aaaaaa', transform=ax.transAxes)
    
    # Pillars data
    pillars = [
        ('TECHNICAL', '35%', 'Price Action\nTrends & Momentum\n20+ Indicators', BLUE),
        ('FUNDAMENTAL', '20%', 'Financials\nValuations\nEarnings Quality', GREEN),
        ('SENTIMENT', '15%', 'Social Buzz\nRetail vs Institutions\nOptions Flow', PURPLE),
        ('NEWS', '10%', 'Breaking News\nAnalyst Ratings\nSEC Filings', CYAN),
        ('ML ADAPTIVE', '20%', 'Pattern Recognition\nRegime Detection\nSelf-Improving', MIDAS_GOLD),
    ]
    
    # Draw pillars as vertical bars
    x_positions = [0.1, 0.28, 0.46, 0.64, 0.82]
    bar_heights = [0.35, 0.20, 0.15, 0.10, 0.20]  # Proportional to weights
    
    for i, ((name, weight, desc, color), x, h) in enumerate(zip(pillars, x_positions, bar_heights)):
        # Bar
        bar_bottom = 0.35
        bar_height = h * 0.8  # Scale for visual
        rect = mpatches.FancyBboxPatch((x-0.06, bar_bottom), 0.12, bar_height + 0.15,
                                        boxstyle='round,pad=0.01,rounding_size=0.01',
                                        facecolor=color, edgecolor='white', linewidth=1, alpha=0.9,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Weight at top of bar
        ax.text(x, bar_bottom + bar_height + 0.18, weight, fontsize=22, fontweight='bold',
                ha='center', va='center', color='white', transform=ax.transAxes)
        
        # Name below bar
        ax.text(x, bar_bottom - 0.04, name, fontsize=11, fontweight='bold',
                ha='center', va='top', color=color, transform=ax.transAxes)
        
        # Description below name
        ax.text(x, bar_bottom - 0.10, desc, fontsize=8, ha='center', va='top',
                color='#888888', transform=ax.transAxes, linespacing=1.2)
    
    # Arrow and final score
    ax.annotate('', xy=(0.5, 0.12), xytext=(0.5, 0.18),
                arrowprops=dict(arrowstyle='->', color=MIDAS_GOLD, lw=3),
                transform=ax.transAxes)
    
    # Score box
    score_rect = mpatches.FancyBboxPatch((0.3, 0.02), 0.4, 0.10,
                                          boxstyle='round,pad=0.01',
                                          facecolor=CARD_BG, edgecolor=MIDAS_GOLD, linewidth=3,
                                          transform=ax.transAxes)
    ax.add_patch(score_rect)
    
    ax.text(0.5, 0.07, 'FINAL SCORE: 0-100', fontsize=18, fontweight='bold',
            ha='center', va='center', color=MIDAS_GOLD, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pillars_clean.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('pillars_clean.png')


def chart_how_it_works():
    """Simple flow showing how MIDAS works."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'HOW MIDAS WORKS', fontsize=28, fontweight='bold',
            ha='center', va='top', color=MIDAS_GOLD, transform=ax.transAxes)
    
    # Steps
    steps = [
        ('1', 'SCAN', '3,400+ stocks\nanalyzed daily', BLUE),
        ('2', 'SCORE', '5 pillars generate\na 0-100 score', GREEN),
        ('3', 'SIGNAL', 'Top picks published\nevery morning', PURPLE),
        ('4', 'TRACK', 'Performance tracked\nat J+1, J+7, J+30', CYAN),
        ('5', 'LEARN', 'AI improves from\nreal results', MIDAS_GOLD),
    ]
    
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for (num, title, desc, color), x in zip(steps, x_positions):
        # Circle with number
        circle = mpatches.Circle((x, 0.55), 0.06, facecolor=color, edgecolor='white',
                                  linewidth=2, transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x, 0.55, num, fontsize=24, fontweight='bold', ha='center', va='center',
                color='white', transform=ax.transAxes)
        
        # Title and desc
        ax.text(x, 0.40, title, fontsize=14, fontweight='bold', ha='center', va='top',
                color=color, transform=ax.transAxes)
        ax.text(x, 0.32, desc, fontsize=10, ha='center', va='top',
                color='#aaaaaa', transform=ax.transAxes, linespacing=1.3)
    
    # Arrows
    for i in range(4):
        ax.annotate('', xy=(x_positions[i+1]-0.08, 0.55), xytext=(x_positions[i]+0.08, 0.55),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=2),
                    transform=ax.transAxes)
    
    # Loop back arrow
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    ax.annotate('', xy=(0.1, 0.70), xytext=(0.9, 0.70),
                arrowprops=dict(arrowstyle='->', color=MIDAS_GOLD, lw=2,
                               connectionstyle='arc3,rad=0.2'),
                transform=ax.transAxes)
    ax.text(0.5, 0.78, 'CONTINUOUS IMPROVEMENT', fontsize=10, ha='center',
            color=MIDAS_GOLD, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/how_it_works.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('how_it_works.png')


def chart_transparency():
    """Our transparency promise."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'OUR TRANSPARENCY PROMISE', fontsize=26, fontweight='bold',
            ha='center', va='top', color=MIDAS_GOLD, transform=ax.transAxes)
    ax.text(0.5, 0.84, '"Judge us by our results, not our promises"', fontsize=14,
            ha='center', va='top', color='#888888', style='italic', transform=ax.transAxes)
    
    # Promises as cards
    promises = [
        ('DAILY PICKS', 'Every morning we publish our signals.\nNo cherry-picking. No hiding losers.', GREEN),
        ('TRACKED RESULTS', 'Every pick tracked at J+1, J+7, J+30.\nPublicly verifiable performance.', BLUE),
        ('OPEN METHOD', 'We explain how MIDAS scores stocks.\nNo black box. Full transparency.', PURPLE),
        ('HONEST LIMITS', 'We share our losses too.\nNo system is perfect. We learn from mistakes.', CYAN),
    ]
    
    positions = [(0.25, 0.58), (0.75, 0.58), (0.25, 0.22), (0.75, 0.22)]
    
    for (title, desc, color), (x, y) in zip(promises, positions):
        # Card
        rect = mpatches.FancyBboxPatch((x-0.20, y-0.12), 0.40, 0.28,
                                        boxstyle='round,pad=0.02',
                                        facecolor=CARD_BG, edgecolor=color, linewidth=2,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Title
        ax.text(x, y+0.10, title, fontsize=14, fontweight='bold', ha='center', va='center',
                color=color, transform=ax.transAxes)
        
        # Description
        ax.text(x, y-0.02, desc, fontsize=10, ha='center', va='center',
                color='#aaaaaa', transform=ax.transAxes, linespacing=1.4)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/transparency.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('transparency.png')


def chart_cta():
    """Call to action card."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    # Main frame
    rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                    boxstyle='round,pad=0.02,rounding_size=0.03',
                                    facecolor=CARD_BG, edgecolor=MIDAS_GOLD, linewidth=4,
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Logo
    ax.text(0.5, 0.80, 'MIDAS', fontsize=56, fontweight='bold',
            ha='center', va='center', color=MIDAS_GOLD, transform=ax.transAxes)
    ax.text(0.5, 0.68, 'AI TRADING SIGNALS', fontsize=18,
            ha='center', va='center', color='white', transform=ax.transAxes)
    
    # Tagline
    ax.text(0.5, 0.55, 'Public Track Record\nStarting February 2026', fontsize=16,
            ha='center', va='center', color='#888888', transform=ax.transAxes,
            linespacing=1.5)
    
    # Features
    features = ['Daily Top Picks', 'Transparent Scoring', 'Tracked Performance', 'Free to Follow']
    for i, feat in enumerate(features):
        ax.text(0.5, 0.40 - i*0.06, f'+ {feat}', fontsize=13, ha='center', va='center',
                color=GREEN, transform=ax.transAxes)
    
    # CTA button
    cta_rect = mpatches.FancyBboxPatch((0.2, 0.10), 0.6, 0.08,
                                        boxstyle='round,pad=0.01',
                                        facecolor=MIDAS_GOLD, edgecolor='none',
                                        transform=ax.transAxes)
    ax.add_patch(cta_rect)
    ax.text(0.5, 0.14, 'SUBSCRIBE FREE', fontsize=16, fontweight='bold',
            ha='center', va='center', color=BACKGROUND, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cta_card.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('cta_card.png')


def chart_signal_example():
    """Example of a MIDAS signal - what subscribers will see."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    # Header
    ax.text(0.5, 0.96, 'EXAMPLE: MIDAS DAILY SIGNAL', fontsize=14,
            ha='center', va='top', color='#666666', transform=ax.transAxes)
    
    # Main card
    rect = mpatches.FancyBboxPatch((0.05, 0.08), 0.9, 0.85,
                                    boxstyle='round,pad=0.02',
                                    facecolor=CARD_BG, edgecolor=MIDAS_GOLD, linewidth=2,
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Ticker and score
    ax.text(0.15, 0.85, 'NVDA', fontsize=36, fontweight='bold',
            ha='left', va='center', color='white', transform=ax.transAxes)
    ax.text(0.15, 0.78, 'NVIDIA Corporation', fontsize=12,
            ha='left', va='center', color='#888888', transform=ax.transAxes)
    
    # Score circle
    circle = mpatches.Circle((0.82, 0.82), 0.08, facecolor=GREEN, edgecolor='white',
                              linewidth=2, transform=ax.transAxes)
    ax.add_patch(circle)
    ax.text(0.82, 0.82, '78', fontsize=28, fontweight='bold', ha='center', va='center',
            color='white', transform=ax.transAxes)
    ax.text(0.82, 0.72, 'BUY', fontsize=12, fontweight='bold', ha='center',
            color=GREEN, transform=ax.transAxes)
    
    # Pillar breakdown
    ax.text(0.15, 0.65, 'PILLAR BREAKDOWN', fontsize=11, fontweight='bold',
            ha='left', va='center', color=MIDAS_GOLD, transform=ax.transAxes)
    
    pillars_data = [
        ('Technical', 82, BLUE),
        ('Fundamental', 75, GREEN),
        ('Sentiment', 88, PURPLE),
        ('News', 65, CYAN),
        ('ML Adaptive', 72, MIDAS_GOLD),
    ]
    
    for i, (name, score, color) in enumerate(pillars_data):
        y = 0.58 - i * 0.07
        # Label
        ax.text(0.15, y, name, fontsize=10, ha='left', va='center',
                color='#aaaaaa', transform=ax.transAxes)
        # Bar background
        bar_bg = mpatches.FancyBboxPatch((0.38, y-0.015), 0.45, 0.03,
                                          boxstyle='round,pad=0.005',
                                          facecolor='#333344', edgecolor='none',
                                          transform=ax.transAxes)
        ax.add_patch(bar_bg)
        # Bar fill
        bar_fill = mpatches.FancyBboxPatch((0.38, y-0.015), 0.45 * score/100, 0.03,
                                            boxstyle='round,pad=0.005',
                                            facecolor=color, edgecolor='none',
                                            transform=ax.transAxes)
        ax.add_patch(bar_fill)
        # Score
        ax.text(0.88, y, str(score), fontsize=10, ha='right', va='center',
                color='white', transform=ax.transAxes)
    
    # Reasoning
    ax.text(0.15, 0.22, 'AI REASONING', fontsize=11, fontweight='bold',
            ha='left', va='center', color=MIDAS_GOLD, transform=ax.transAxes)
    
    reasoning = """Strong technical momentum with price above all major MAs.
Solid fundamentals with 25% YoY revenue growth.
Extremely positive social sentiment around AI/datacenter.
Recent analyst upgrades from Goldman and Morgan Stanley.
ML model detects bullish regime continuation pattern."""
    
    ax.text(0.15, 0.17, reasoning, fontsize=9, ha='left', va='top',
            color='#aaaaaa', transform=ax.transAxes, linespacing=1.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/signal_example.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('signal_example.png')


if __name__ == '__main__':
    print('Generating MIDAS v3 clean charts...')
    chart_5_pillars()
    chart_how_it_works()
    chart_transparency()
    chart_cta()
    chart_signal_example()
    print('\nAll done!')
