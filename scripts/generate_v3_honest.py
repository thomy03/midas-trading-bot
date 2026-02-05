#!/usr/bin/env python3
"""
Generate honest marketing charts for MIDAS.
Version 3 - No fake backtest, focus on methodology and transparency.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Professional color palette
BACKGROUND = '#0f0f1a'
CARD_BG = '#1a1a2e'
MIDAS_GOLD = '#FFD700'
MIDAS_ORANGE = '#FF8C00'
BLUE = '#4A90D9'
PURPLE = '#9B6DFF'
GREEN = '#00D084'
RED = '#FF4757'
CYAN = '#4ECDC4'
GRID_COLOR = '#2a2a4a'
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
    ax.text(0.5, 0.95, 'MIDAS: 5 Intelligent Pillars', fontsize=28, fontweight='bold',
            ha='center', va='top', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.89, 'Each stock is analyzed through 5 independent AI modules', fontsize=14,
            ha='center', va='top', color='#888888', transform=ax.transAxes, style='italic')
    
    # Pillars data
    pillars = [
        ('📊', 'Technical', '35%', 'Price patterns, trends,\nmomentum indicators', BLUE),
        ('💰', 'Fundamental', '20%', 'Financials, valuations,\nearnings quality', GREEN),
        ('🐦', 'Sentiment', '15%', 'Social media buzz,\nretail vs institutional', PURPLE),
        ('📰', 'News', '10%', 'Breaking news,\nanalyst ratings', CYAN),
        ('🧠', 'ML Adaptive', '20%', 'Pattern recognition,\nself-improving model', MIDAS_GOLD),
    ]
    
    # Draw pillars as cards
    y_pos = 0.72
    x_positions = [0.1, 0.28, 0.46, 0.64, 0.82]
    
    for i, (emoji, name, weight, desc, color) in enumerate(pillars):
        x = x_positions[i]
        
        # Card background
        rect = mpatches.FancyBboxPatch((x-0.07, y_pos-0.28), 0.14, 0.35,
                                        boxstyle='round,pad=0.02,rounding_size=0.02',
                                        facecolor=CARD_BG, edgecolor=color, linewidth=3,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Emoji
        ax.text(x, y_pos+0.02, emoji, fontsize=36, ha='center', va='center', transform=ax.transAxes)
        
        # Name
        ax.text(x, y_pos-0.08, name, fontsize=14, fontweight='bold',
                ha='center', va='center', color='white', transform=ax.transAxes)
        
        # Weight
        ax.text(x, y_pos-0.15, weight, fontsize=20, fontweight='bold',
                ha='center', va='center', color=color, transform=ax.transAxes)
        
        # Description
        ax.text(x, y_pos-0.24, desc, fontsize=9, ha='center', va='center',
                color='#aaaaaa', transform=ax.transAxes, linespacing=1.3)
    
    # Arrow pointing down to score
    ax.annotate('', xy=(0.5, 0.32), xytext=(0.5, 0.40),
                arrowprops=dict(arrowstyle='->', color=MIDAS_GOLD, lw=3),
                transform=ax.transAxes)
    
    # Final score box
    score_rect = mpatches.FancyBboxPatch((0.3, 0.12), 0.4, 0.18,
                                          boxstyle='round,pad=0.02,rounding_size=0.02',
                                          facecolor=CARD_BG, edgecolor=MIDAS_GOLD, linewidth=4,
                                          transform=ax.transAxes)
    ax.add_patch(score_rect)
    
    ax.text(0.5, 0.24, 'MIDAS SCORE', fontsize=12, ha='center', va='center',
            color='#888888', transform=ax.transAxes)
    ax.text(0.5, 0.17, '0-100', fontsize=36, fontweight='bold', ha='center', va='center',
            color=MIDAS_GOLD, transform=ax.transAxes)
    
    # Legend at bottom
    ax.text(0.5, 0.04, '> 60 = BUY signal  |  40-60 = HOLD  |  < 40 = AVOID',
            fontsize=12, ha='center', va='center', color='#666666', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pillars_v3.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('✅ pillars_v3.png')


def chart_adaptive_learning():
    """Show how the system improves over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Self-Improving AI: How MIDAS Learns', fontsize=26, fontweight='bold',
            ha='center', va='top', color='white', transform=ax.transAxes)
    
    # Flow diagram
    steps = [
        ('1️⃣', 'ANALYZE', 'Scan 3,400+ stocks\ndaily with 5 pillars', 0.12),
        ('2️⃣', 'SIGNAL', 'Generate BUY/SELL\nrecommendations', 0.32),
        ('3️⃣', 'TRACK', 'Monitor actual\nmarket outcomes', 0.52),
        ('4️⃣', 'LEARN', 'Adjust weights for\naccurate indicators', 0.72),
        ('5️⃣', 'IMPROVE', 'Better predictions\nover time', 0.92),
    ]
    
    for emoji, title, desc, x in steps:
        # Circle
        circle = mpatches.Circle((x, 0.6), 0.06, facecolor=CARD_BG, edgecolor=MIDAS_GOLD,
                                  linewidth=2, transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x, 0.6, emoji, fontsize=24, ha='center', va='center', transform=ax.transAxes)
        
        # Title and desc below
        ax.text(x, 0.48, title, fontsize=14, fontweight='bold', ha='center', va='top',
                color=MIDAS_GOLD, transform=ax.transAxes)
        ax.text(x, 0.42, desc, fontsize=10, ha='center', va='top',
                color='#aaaaaa', transform=ax.transAxes, linespacing=1.4)
    
    # Arrows between steps
    for i in range(4):
        ax.annotate('', xy=(steps[i+1][3]-0.08, 0.6), xytext=(steps[i][3]+0.08, 0.6),
                    arrowprops=dict(arrowstyle='->', color='#444466', lw=2),
                    transform=ax.transAxes)
    
    # Loop arrow from step 5 back to step 1
    ax.annotate('', xy=(0.12, 0.72), xytext=(0.92, 0.72),
                arrowprops=dict(arrowstyle='->', color=MIDAS_GOLD, lw=2,
                               connectionstyle='arc3,rad=0.3'),
                transform=ax.transAxes)
    ax.text(0.5, 0.82, 'Continuous Learning Loop', fontsize=11, ha='center',
            color=MIDAS_GOLD, style='italic', transform=ax.transAxes)
    
    # Key benefit box
    benefit_rect = mpatches.FancyBboxPatch((0.2, 0.08), 0.6, 0.18,
                                            boxstyle='round,pad=0.02',
                                            facecolor=CARD_BG, edgecolor=GREEN, linewidth=2,
                                            transform=ax.transAxes)
    ax.add_patch(benefit_rect)
    
    ax.text(0.5, 0.20, '💡 Unlike static algorithms, MIDAS gets smarter every day', 
            fontsize=14, fontweight='bold', ha='center', va='center',
            color='white', transform=ax.transAxes)
    ax.text(0.5, 0.12, 'Indicators that work gain weight • Bad signals get filtered out',
            fontsize=11, ha='center', va='center', color='#888888', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/learning_v3.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('✅ learning_v3.png')


def chart_transparency_commitment():
    """Our commitment to transparency."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.92, '🔍 Our Transparency Commitment', fontsize=26, fontweight='bold',
            ha='center', va='top', color='white', transform=ax.transAxes)
    
    # Commitments
    commitments = [
        ('📊', 'Daily Public Picks', 'Every morning we publish our top signals.\nNo cherry-picking, no hiding losers.'),
        ('📈', 'Tracked Performance', 'Every pick is tracked at J+1, J+7, J+30.\nReal results, publicly verifiable.'),
        ('🔓', 'Open Methodology', 'We explain exactly how MIDAS scores stocks.\nNo black box magic.'),
        ('⚠️', 'Honest About Limits', 'We share losses too. No system is perfect.\nWe improve from mistakes.'),
    ]
    
    y_positions = [0.72, 0.52, 0.32, 0.12]
    
    for (emoji, title, desc), y in zip(commitments, y_positions):
        # Icon circle
        circle = mpatches.Circle((0.12, y+0.04), 0.05, facecolor=MIDAS_GOLD, alpha=0.2,
                                  transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(0.12, y+0.04, emoji, fontsize=28, ha='center', va='center', transform=ax.transAxes)
        
        # Title and description
        ax.text(0.22, y+0.06, title, fontsize=16, fontweight='bold', ha='left', va='center',
                color='white', transform=ax.transAxes)
        ax.text(0.22, y-0.02, desc, fontsize=11, ha='left', va='top',
                color='#aaaaaa', transform=ax.transAxes, linespacing=1.4)
    
    # Bottom message
    ax.text(0.5, 0.02, '"Judge us by our results, not our promises."',
            fontsize=14, ha='center', va='bottom', color=MIDAS_GOLD, style='italic',
            transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/transparency_v3.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('✅ transparency_v3.png')


def chart_cta_card():
    """Call to action card for social media."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis('off')
    
    # Main card
    main_rect = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                         boxstyle='round,pad=0.02,rounding_size=0.03',
                                         facecolor=CARD_BG, edgecolor=MIDAS_GOLD, linewidth=3,
                                         transform=ax.transAxes)
    ax.add_patch(main_rect)
    
    # Logo/Title
    ax.text(0.5, 0.82, '🤖 MIDAS', fontsize=48, fontweight='bold',
            ha='center', va='center', color=MIDAS_GOLD, transform=ax.transAxes)
    ax.text(0.5, 0.70, 'AI Trading Signals', fontsize=20,
            ha='center', va='center', color='white', transform=ax.transAxes)
    
    # Tagline
    ax.text(0.5, 0.55, 'Public Track Record Starting Now', fontsize=18,
            ha='center', va='center', color='#888888', style='italic', transform=ax.transAxes)
    
    # Features
    features = ['✅ Daily top picks', '✅ Transparent scoring', '✅ Tracked performance', '✅ Free to follow']
    for i, feat in enumerate(features):
        ax.text(0.5, 0.42 - i*0.07, feat, fontsize=14, ha='center', va='center',
                color='white', transform=ax.transAxes)
    
    # CTA
    cta_rect = mpatches.FancyBboxPatch((0.25, 0.12), 0.5, 0.08,
                                        boxstyle='round,pad=0.01',
                                        facecolor=MIDAS_GOLD, edgecolor='none',
                                        transform=ax.transAxes)
    ax.add_patch(cta_rect)
    ax.text(0.5, 0.16, 'Subscribe Free →', fontsize=16, fontweight='bold',
            ha='center', va='center', color=BACKGROUND, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cta_card_v3.png', dpi=200, bbox_inches='tight',
                facecolor=BACKGROUND, edgecolor='none')
    plt.close()
    print('✅ cta_card_v3.png')


if __name__ == '__main__':
    print('🎨 Generating MIDAS honest charts v3...')
    chart_5_pillars()
    chart_adaptive_learning()
    chart_transparency_commitment()
    chart_cta_card()
    print('\n✅ All v3 charts generated!')
