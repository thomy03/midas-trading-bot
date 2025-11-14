"""
Trendline Visualization

Adds trendlines and breakout markers to RSI charts using Plotly.
Integrates with the existing Visualizer class from src/utils/visualizer.py
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict

from ..core.trendline_detector import Trendline, RSITrendlineDetector
from ..core.breakout_analyzer import Breakout, TrendlineBreakoutAnalyzer
from ..config.settings import (
    TRENDLINE_COLOR,
    BREAKOUT_COLOR,
    TRENDLINE_WIDTH,
    TRENDLINE_DASH
)


class TrendlineVisualizer:
    """Adds trendline visualization to RSI charts"""

    def __init__(self):
        """Initialize visualizer"""
        self.detector = RSITrendlineDetector()
        self.analyzer = TrendlineBreakoutAnalyzer()

    def add_trendline_to_rsi_subplot(
        self,
        fig: go.Figure,
        rsi: pd.Series,
        trendline: Trendline,
        breakout: Optional[Breakout] = None,
        row: int = 2,
        col: int = 1
    ):
        """
        Add trendline and breakout markers to RSI subplot

        Args:
            fig: Plotly figure
            rsi: RSI values
            trendline: Trendline object
            breakout: Optional breakout object
            row: Subplot row number (default 2 for RSI)
            col: Subplot column number
        """
        # Create trendline data points
        trendline_x = []
        trendline_y = []

        for i in range(trendline.start_idx, trendline.end_idx + 1):
            trendline_x.append(rsi.index[i])
            trendline_y.append(self.detector.get_trendline_value(trendline, i))

        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=trendline_x,
                y=trendline_y,
                mode='lines',
                name=f'Trendline (R²={trendline.r_squared:.2f})',
                line=dict(
                    color=TRENDLINE_COLOR,
                    width=TRENDLINE_WIDTH,
                    dash=TRENDLINE_DASH
                ),
                hovertemplate=(
                    'Trendline<br>'
                    'Date: %{x}<br>'
                    'Value: %{y:.2f}<br>'
                    f'Quality: {trendline.quality_score:.1f}/100<br>'
                    f'Slope: {trendline.slope:.3f}<br>'
                    '<extra></extra>'
                ),
                showlegend=True
            ),
            row=row,
            col=col
        )

        # Add peak markers
        peak_x = [rsi.index[i] for i in trendline.peak_indices]
        peak_y = trendline.peak_values

        fig.add_trace(
            go.Scatter(
                x=peak_x,
                y=peak_y,
                mode='markers',
                name='Trendline Peaks',
                marker=dict(
                    color=TRENDLINE_COLOR,
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    'Peak<br>'
                    'Date: %{x}<br>'
                    'RSI: %{y:.2f}<br>'
                    '<extra></extra>'
                ),
                showlegend=True
            ),
            row=row,
            col=col
        )

        # Add breakout marker if present
        if breakout:
            fig.add_trace(
                go.Scatter(
                    x=[breakout.date],
                    y=[breakout.rsi_value],
                    mode='markers+text',
                    name=f'Breakout ({breakout.strength})',
                    marker=dict(
                        color=BREAKOUT_COLOR,
                        size=15,
                        symbol='star',
                        line=dict(color='white', width=2)
                    ),
                    text=[f'BREAKOUT<br>{breakout.strength}'],
                    textposition='top center',
                    textfont=dict(
                        size=10,
                        color=BREAKOUT_COLOR,
                        family='Arial Black'
                    ),
                    hovertemplate=(
                        f'<b>BREAKOUT</b><br>'
                        'Date: %{x}<br>'
                        f'RSI: {breakout.rsi_value:.2f}<br>'
                        f'Trendline: {breakout.trendline_value:.2f}<br>'
                        f'Distance: +{breakout.distance_above:.2f}<br>'
                        f'Strength: {breakout.strength}<br>'
                        f'Confirmed: {breakout.is_confirmed}<br>'
                        f'Age: {breakout.age_in_periods} periods ago<br>'
                        '<extra></extra>'
                    ),
                    showlegend=True
                ),
                row=row,
                col=col
            )

            # Add vertical line at breakout
            fig.add_vline(
                x=breakout.date.timestamp() * 1000 if hasattr(breakout.date, 'timestamp') else breakout.date,
                line=dict(
                    color=BREAKOUT_COLOR,
                    width=1,
                    dash='dot'
                ),
                annotation_text=f"Breakout",
                annotation_position="top",
                row=row,
                col=col
            )

    def create_annotated_rsi_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        lookback_periods: int = 52
    ) -> Optional[Dict]:
        """
        Create a standalone RSI chart with trendline annotations

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            timeframe: Timeframe string
            lookback_periods: Lookback for trendline detection

        Returns:
            Dictionary with fig and analysis, or None
        """
        # Analyze
        analysis = self.analyzer.analyze(df, lookback_periods)

        if not analysis or not analysis['trendline']:
            return None

        rsi = analysis['rsi']
        trendline = analysis['trendline']
        breakout = analysis.get('breakout')

        # Create figure
        fig = go.Figure()

        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi.values,
                mode='lines',
                name='RSI (14)',
                line=dict(color='#2962FF', width=2)
            )
        )

        # Add RSI EMAs if calculated
        if 'RSI_EMA9' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_EMA9'],
                    name='RSI EMA9',
                    line=dict(color='#FF6D00', width=1.5)
                )
            )

        if 'RSI_EMA14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_EMA14'],
                    name='RSI EMA14',
                    line=dict(color='#00E676', width=1.5)
                )
            )

        # Add reference lines
        for level, color, name in [(50, 'gray', '50'), (30, 'green', '30'), (70, 'red', '70')]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=color,
                annotation_text=name,
                annotation_position="right"
            )

        # Add trendline visualization (using row=None for single plot)
        self.add_trendline_to_rsi_subplot(
            fig, rsi, trendline, breakout,
            row=None, col=None
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} - RSI Trendline Analysis ({timeframe.upper()})',
            xaxis_title='Date',
            yaxis_title='RSI',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return {
            'fig': fig,
            'analysis': analysis,
            'has_breakout': analysis['has_breakout'],
            'signal': self.analyzer.get_signal(analysis)
        }

    def get_trendline_info_text(self, trendline: Trendline, breakout: Optional[Breakout] = None) -> str:
        """
        Generate text description of trendline and breakout

        Args:
            trendline: Trendline object
            breakout: Optional breakout object

        Returns:
            Formatted text description
        """
        text = f"""
Trendline Analysis:
-------------------
Quality Score: {trendline.quality_score:.1f}/100
R² Fit: {trendline.r_squared:.3f}
Slope: {trendline.slope:.3f}
Number of Peaks: {len(trendline.peak_indices)}
Date Range: {trendline.peak_dates[0].strftime('%Y-%m-%d')} to {trendline.peak_dates[-1].strftime('%Y-%m-%d')}
"""

        if breakout:
            text += f"""
Breakout Detected:
------------------
Date: {breakout.date.strftime('%Y-%m-%d')}
RSI Value: {breakout.rsi_value:.2f}
Trendline Value: {breakout.trendline_value:.2f}
Distance Above: +{breakout.distance_above:.2f} points
Strength: {breakout.strength}
Confirmed: {'Yes' if breakout.is_confirmed else 'No'}
Age: {breakout.age_in_periods} periods ago
"""

        return text
