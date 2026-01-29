"""
Interactive Chart Component with TradingView-like features

Features:
- Zoom and pan with mouse
- Crosshair cursor with price/date display
- Drawing tools (trendlines, horizontal lines, rectangles, Fibonacci)
- Multiple timeframe support
- Save/load drawings
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from pathlib import Path


class InteractiveChartBuilder:
    """Builder for interactive TradingView-like charts"""

    def __init__(self):
        self.drawings_dir = Path("data/drawings")
        self.drawings_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme (dark theme)
        self.colors = {
            'background': '#131722',
            'grid': '#1e222d',
            'text': '#d1d4dc',
            'up_candle': '#26a69a',
            'down_candle': '#ef5350',
            'ema_24': '#2962FF',
            'ema_38': '#FF6D00',
            'ema_62': '#AB47BC',
            'volume_up': 'rgba(38, 166, 154, 0.5)',
            'volume_down': 'rgba(239, 83, 80, 0.5)',
            'crosshair': '#758696',
            'support': '#00E676',
            'resistance': '#FF5252',
            'rsi': '#2196F3',
            'rsi_overbought': '#ef5350',
            'rsi_oversold': '#26a69a',
            # Confidence score colors
            'score_strong': '#00C853',   # Green for STRONG_BUY (>75)
            'score_buy': '#FF9800',       # Orange for BUY (55-75)
            'score_watch': '#9E9E9E',     # Gray for WATCH (35-55)
            'score_observe': '#616161',   # Dark gray for OBSERVE (<35)
            # Volume ratio colors
            'volume_high': 'rgba(0, 200, 83, 0.9)',    # Bright green (>1.5x)
            'volume_medium': 'rgba(255, 152, 0, 0.8)', # Orange (1.0-1.5x)
            'volume_low': 'rgba(158, 158, 158, 0.5)',  # Gray (<1.0x)
            # Risk zone
            'risk_zone': 'rgba(255, 82, 82, 0.15)'
        }

    def create_interactive_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = 'daily',
        show_volume: bool = True,
        show_rsi: bool = True,
        show_emas: bool = True,
        ema_periods: List[int] = [24, 38, 62],
        drawings: Optional[List[Dict]] = None,
        height: int = 800,
        crossovers: Optional[List[Dict]] = None,
        show_crossover_zones: bool = True
    ) -> go.Figure:
        """
        Create an interactive chart with TradingView-like features

        Args:
            crossovers: List of EMA crossover events to display as zones
            show_crossover_zones: Whether to show crossover zones on chart
        """
        # Extract price data (handle multi-level columns from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
            open_price = df['Open'].iloc[:, 0]
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
            volume = df['Volume'].iloc[:, 0]
        else:
            close = df['Close']
            open_price = df['Open']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=3 if show_volume and show_rsi else (2 if show_volume or show_rsi else 1),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2] if show_volume and show_rsi else ([0.7, 0.3] if show_volume or show_rsi else [1.0]),
            subplot_titles=[f'{symbol} - {timeframe.upper()}', 'Volume' if show_volume else 'RSI (14)', 'RSI (14)' if show_volume and show_rsi else None]
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=open_price,
                high=high,
                low=low,
                close=close,
                name='Price',
                increasing_line_color=self.colors['up_candle'],
                decreasing_line_color=self.colors['down_candle'],
                increasing_fillcolor=self.colors['up_candle'],
                decreasing_fillcolor=self.colors['down_candle']
            ),
            row=1, col=1
        )

        # Add EMAs
        if show_emas:
            ema_colors = [self.colors['ema_24'], self.colors['ema_38'], self.colors['ema_62']]
            for i, period in enumerate(ema_periods):
                ema = close.ewm(span=period, adjust=False).mean()
                color = ema_colors[i] if i < len(ema_colors) else '#FFFFFF'
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema,
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=color, width=1.5)
                    ),
                    row=1, col=1
                )

        # Volume bars
        current_row = 2
        if show_volume:
            colors = [self.colors['volume_up'] if close.iloc[i] >= open_price.iloc[i] else self.colors['volume_down']
                     for i in range(len(close))]

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=volume,
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=current_row, col=1
            )
            current_row += 1

        # RSI
        if show_rsi:
            rsi = self._calculate_rsi(close, period=14)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['rsi'], width=2)
                ),
                row=current_row, col=1
            )

            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color=self.colors['rsi_overbought'],
                         line_width=1, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=self.colors['rsi_oversold'],
                         line_width=1, row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color=self.colors['crosshair'],
                         line_width=1, row=current_row, col=1)

        # Add EMA crossover zones
        if show_crossover_zones and crossovers:
            self._add_crossover_zones(fig, crossovers, df.index, close)

        # Layout configuration - Pan mode by default, scroll wheel zoom enabled
        fig.update_layout(
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'], size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            hovermode='x unified',
            dragmode='pan',  # Default to pan mode - click and drag to move
            margin=dict(l=10, r=60, t=80, b=40),

            # Range selector buttons
            xaxis=dict(
                rangeslider=dict(visible=False),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor=self.colors['grid'],
                    activecolor=self.colors['ema_24'],
                    font=dict(color=self.colors['text']),
                    x=0,
                    xanchor='left'
                ),
                fixedrange=False  # Allow x-axis zoom
            )
        )

        # Configure y-axes
        fig.update_yaxes(
            title_text="Price",
            side='right',
            gridcolor=self.colors['grid'],
            tickformat='$,.2f',
            row=1, col=1
        )

        if show_volume:
            fig.update_yaxes(
                title_text="Volume",
                side='right',
                gridcolor=self.colors['grid'],
                tickformat='.2s',
                row=2, col=1
            )

        if show_rsi:
            rsi_row = 3 if show_volume else 2
            fig.update_yaxes(
                title_text="RSI",
                side='right',
                gridcolor=self.colors['grid'],
                range=[0, 100],
                dtick=20,
                row=rsi_row, col=1
            )

        # Configure x-axes
        fig.update_xaxes(gridcolor=self.colors['grid'])

        # Crosshair
        fig.update_xaxes(showspikes=True, spikecolor=self.colors['crosshair'],
                        spikesnap="cursor", spikemode="across", spikethickness=1)
        fig.update_yaxes(showspikes=True, spikecolor=self.colors['crosshair'],
                        spikesnap="cursor", spikemode="across", spikethickness=1)

        return fig

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_score_color(self, score: float) -> str:
        """Get color based on confidence score"""
        if score >= 75:
            return self.colors['score_strong']
        elif score >= 55:
            return self.colors['score_buy']
        elif score >= 35:
            return self.colors['score_watch']
        else:
            return self.colors['score_observe']

    def add_confidence_badge(
        self,
        fig: go.Figure,
        score: float,
        signal: str
    ) -> go.Figure:
        """
        Add confidence score badge to top-right of chart

        Args:
            fig: Plotly figure
            score: Confidence score (0-100)
            signal: Signal text (STRONG_BUY, BUY, WATCH, OBSERVE)

        Returns:
            Updated figure with badge
        """
        if score is None or signal is None:
            return fig

        color = self._get_score_color(score)

        # Format signal display (remove underscores, shorten)
        display_signal = signal.replace('_', ' ')

        fig.add_annotation(
            text=f"<b>{display_signal}</b><br><span style='font-size:18px'>{score:.0f}/100</span>",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor=color,
            bordercolor=color,
            borderwidth=2,
            borderpad=8,
            opacity=0.95,
            align="center"
        )

        return fig

    def add_volume_ratio_annotation(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        breakout_idx: int,
        volume_ratio: float,
        volume_row: int = 2
    ) -> go.Figure:
        """
        Add volume ratio annotation at breakout bar

        Args:
            fig: Plotly figure
            df: DataFrame with OHLCV data
            breakout_idx: Index of breakout bar
            volume_ratio: Volume ratio at breakout (vs 20-day avg)
            volume_row: Row number of volume subplot

        Returns:
            Updated figure with annotation
        """
        if volume_ratio is None or breakout_idx is None:
            return fig

        try:
            # Get volume at breakout
            if isinstance(df.columns, pd.MultiIndex):
                volume = df['Volume'].iloc[:, 0]
            else:
                volume = df['Volume']

            breakout_date = df.index[breakout_idx]
            breakout_volume = float(volume.iloc[breakout_idx])

            # Color based on ratio
            if volume_ratio >= 1.5:
                color = self.colors['volume_high']
                text_color = "#00C853"
            elif volume_ratio >= 1.0:
                color = self.colors['volume_medium']
                text_color = "#FF9800"
            else:
                color = self.colors['volume_low']
                text_color = "#9E9E9E"

            # Add annotation above volume bar
            fig.add_annotation(
                x=breakout_date,
                y=breakout_volume,
                text=f"<b>{volume_ratio:.1f}x</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=text_color,
                ax=0,
                ay=-25,
                font=dict(size=11, color=text_color),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=text_color,
                borderwidth=1,
                borderpad=3,
                row=volume_row, col=1
            )

        except Exception:
            pass

        return fig

    def add_risk_zone(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        entry_price: float,
        stop_loss: float,
        row: int = 1
    ) -> go.Figure:
        """
        Add risk zone rectangle between entry and stop-loss

        Args:
            fig: Plotly figure
            df: DataFrame with OHLCV data
            entry_price: Entry price (current price)
            stop_loss: Stop-loss price
            row: Row number for price chart

        Returns:
            Updated figure with risk zone
        """
        if entry_price is None or stop_loss is None:
            return fig

        if stop_loss <= 0 or entry_price <= 0:
            return fig

        try:
            # Use last 20 bars for zone width
            zone_start = df.index[-min(20, len(df))]
            zone_end = df.index[-1]

            # Add semi-transparent rectangle
            fig.add_shape(
                type="rect",
                x0=zone_start, x1=zone_end,
                y0=stop_loss, y1=entry_price,
                fillcolor=self.colors['risk_zone'],
                line=dict(width=0),
                row=row, col=1
            )

            # Calculate and display risk percentage
            risk_pct = (entry_price - stop_loss) / entry_price * 100
            mid_price = (entry_price + stop_loss) / 2
            mid_date = df.index[-10] if len(df) >= 10 else df.index[-1]

            fig.add_annotation(
                x=mid_date,
                y=mid_price,
                text=f"<b>Risque: {risk_pct:.1f}%</b>",
                showarrow=False,
                font=dict(size=10, color="#FF5252"),
                bgcolor='rgba(0,0,0,0.6)',
                borderpad=4
            )

        except Exception:
            pass

        return fig

    def add_position_info_box(
        self,
        fig: go.Figure,
        position_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Add position sizing info box to top-left of chart

        Args:
            fig: Plotly figure
            position_data: Dict with keys: shares, value, stop_loss, risk_pct

        Returns:
            Updated figure with info box
        """
        if not position_data:
            return fig

        shares = position_data.get('shares')
        value = position_data.get('value')
        stop_loss = position_data.get('stop_loss')
        risk_pct = position_data.get('risk_pct')

        # Build info text
        lines = ["<b>Position Recommandée</b>"]

        if shares is not None and shares > 0:
            lines.append(f"Actions: {int(shares)}")

        if value is not None and value > 0:
            lines.append(f"Valeur: ${value:,.0f}")

        if stop_loss is not None and stop_loss > 0:
            lines.append(f"Stop: ${stop_loss:.2f}")

        if risk_pct is not None:
            lines.append(f"Risque: {risk_pct:.1f}%")

        # Only show if we have useful info
        if len(lines) <= 1:
            return fig

        info_text = "<br>".join(lines)

        fig.add_annotation(
            text=info_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color=self.colors['text']),
            bgcolor='rgba(30, 30, 30, 0.9)',
            bordercolor='#424242',
            borderwidth=1,
            borderpad=8,
            align="left"
        )

        return fig

    def add_rsi_trendline_breakout(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        rsi_row: int = 3,
        df_with_emas: pd.DataFrame = None,
        support_zones: list = None,
        show_stop_loss: bool = True,
        forced_peak_indices: list = None
    ) -> go.Figure:
        """
        Add ALL RSI trendlines and breakout visualization to the chart

        LOGIQUE DE VALIDATION DU BREAKOUT:
        Un breakout RSI n'est VALIDE que si:
        1. EMAs croissantes: Au moins une EMA courte > EMA longue (24>38 OU 24>62 OU 38>62)
        2. Prix proche d'un support EMA: Le prix a touché une zone de support récemment
        3. Le breakout intervient tant que les EMAs restent en configuration croissante

        STOP-LOSS:
        Le stop-loss est calculé au prix LOW du bar où le RSI était au minimum
        durant la période de la première oblique affichée (celle avec le meilleur score).

        Args:
            fig: Plotly figure with RSI subplot
            df: DataFrame with OHLCV data
            rsi_row: Row number of RSI subplot (default 3 with volume)
            df_with_emas: DataFrame with EMA columns (optional, for validation)
            support_zones: List of support zones (optional, for validation)
            show_stop_loss: If True, add stop-loss line based on RSI minimum (default True)
            forced_peak_indices: If provided, use ONLY this trendline (from ELITE signal)
                                 instead of detecting trendlines dynamically

        Returns:
            Updated figure with trendline and breakout markers
        """
        try:
            from trendline_analysis.core.trendline_detector import RSITrendlineDetector, Trendline
            from trendline_analysis.core.breakout_analyzer import TrendlineBreakoutAnalyzer
            import numpy as np

            detector = RSITrendlineDetector()
            breakout_analyzer = TrendlineBreakoutAnalyzer()
            lookback = 252 if len(df) > 200 else 104

            # Calculate RSI
            rsi = detector.calculate_rsi(df)

            # If forced_peak_indices provided from ELITE signal URL, use ONLY that trendline
            if forced_peak_indices and len(forced_peak_indices) >= 3:
                # Build the forced trendline directly from the provided indices
                valid_indices = [i for i in forced_peak_indices if 0 <= i < len(rsi)]
                if len(valid_indices) >= 3:
                    selected_peaks = np.array(valid_indices)
                    selected_rsi = rsi.iloc[selected_peaks].values

                    # Linear regression
                    slope, intercept = np.polyfit(selected_peaks, selected_rsi, 1)
                    predicted_rsi = slope * selected_peaks + intercept
                    ss_res = np.sum((selected_rsi - predicted_rsi) ** 2)
                    ss_tot = np.sum((selected_rsi - np.mean(selected_rsi)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    # Create trendline object
                    # Extend to end of data: the trendline continues until the breakout
                    # (the buy signal), so stop-loss should include RSI minimum until then
                    forced_trendline = Trendline(
                        slope=slope,
                        intercept=intercept,
                        r_squared=r_squared,
                        peak_indices=valid_indices,
                        peak_dates=[df.index[i] for i in valid_indices],
                        peak_values=list(selected_rsi),  # RSI values at peak indices
                        start_idx=min(valid_indices),
                        end_idx=len(df) - 1,  # Extend to end (until breakout/buy signal)
                        quality_score=r_squared  # Use R² as quality score
                    )
                    all_trendlines = [forced_trendline]  # Only this trendline
                else:
                    # Fallback to dynamic detection if forced indices invalid
                    peaks, _ = detector.detect_peaks(rsi)
                    if len(peaks) == 0:
                        return fig
                    all_trendlines = detector.find_all_trendlines(rsi, peaks, lookback)
            else:
                # Normal mode: detect peaks and find all trendlines dynamically
                peaks, _ = detector.detect_peaks(rsi)
                if len(peaks) == 0:
                    return fig
                all_trendlines = detector.find_all_trendlines(rsi, peaks, lookback)

            if not all_trendlines:
                return fig

            # Colors for different trendlines
            # PRIMARY (breakout): Orange, bright and thick
            # SECONDARY (no breakout): Purple/Cyan, dimmer and thinner
            primary_color = '#FF9800'  # Orange for breakout trendline
            secondary_colors = ['#9C27B0', '#00BCD4', '#8BC34A']  # Other trendlines

            # Track all peaks used (to avoid duplicate markers)
            all_peak_indices = set()
            all_breakouts = []

            # FIRST PASS: Identify all trendlines with breakouts
            trendline_has_breakout = {}
            for trendline in all_trendlines:
                breakout = breakout_analyzer.detect_breakout(rsi, trendline)
                trendline_has_breakout[id(trendline)] = breakout

            # Find the primary breakout trendline (first one with breakout)
            primary_breakout_trendline = None
            for trendline in all_trendlines:
                if trendline_has_breakout[id(trendline)]:
                    primary_breakout_trendline = trendline
                    break

            # SECOND PASS: Draw each trendline with appropriate style
            secondary_idx = 0
            for idx, trendline in enumerate(all_trendlines):
                breakout = trendline_has_breakout[id(trendline)]
                is_primary = (trendline is primary_breakout_trendline)

                if is_primary:
                    # PRIMARY BREAKOUT TRENDLINE: Orange, thick, solid
                    color = primary_color
                    line_width = 3
                    line_dash = 'solid'
                    label = 'RSI Oblique BREAKOUT'
                else:
                    # SECONDARY TRENDLINE: Dimmer, thinner, dashed
                    color = secondary_colors[secondary_idx % len(secondary_colors)]
                    secondary_idx += 1
                    line_width = 1.5
                    line_dash = 'dash'
                    label = f'RSI Oblique #{idx+1}'

                # Draw RSI trendline
                # Extend trendline to the end of the data (not just to last peak)
                end_idx = min(trendline.end_idx, len(df) - 1)
                trendline_x = df.index[trendline.start_idx:end_idx + 1].tolist()
                trendline_y = [
                    detector.get_trendline_value(trendline, i)
                    for i in range(trendline.start_idx, end_idx + 1)
                ]

                fig.add_trace(
                    go.Scatter(
                        x=trendline_x,
                        y=trendline_y,
                        mode='lines',
                        name=label,
                        line=dict(color=color, width=line_width, dash=line_dash),
                        hovertemplate=f'{label}: %{{y:.1f}}<extra></extra>',
                        showlegend=is_primary  # Only show breakout trendline in legend
                    ),
                    row=rsi_row, col=1
                )

                # Add R² and peak count label on trendline
                mid_idx = (trendline.start_idx + end_idx) // 2
                if mid_idx < len(df):
                    mid_x = df.index[mid_idx]
                    mid_y = detector.get_trendline_value(trendline, mid_idx)
                    num_peaks = len(trendline.peak_indices)

                    fig.add_annotation(
                        x=mid_x,
                        y=mid_y + 3,  # Slightly above trendline
                        text=f"R²={trendline.r_squared:.2f} ({num_peaks}pts)",
                        showarrow=False,
                        font=dict(size=9, color=color),
                        bgcolor='rgba(0,0,0,0.5)',
                        borderpad=2,
                        row=rsi_row, col=1
                    )

                # Collect peaks for this trendline
                for peak_idx in trendline.peak_indices:
                    all_peak_indices.add(peak_idx)

                # Store breakout info
                if breakout:
                    all_breakouts.append((trendline, breakout, color))

            # Mark ALL peaks used (red triangles) - once for all trendlines
            all_peak_dates = [df.index[i] for i in sorted(all_peak_indices)]
            all_peak_values = [float(rsi.iloc[i]) for i in sorted(all_peak_indices)]

            fig.add_trace(
                go.Scatter(
                    x=all_peak_dates,
                    y=all_peak_values,
                    mode='markers',
                    name='RSI Peaks',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='#ef5350',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Peak RSI: %{y:.1f}<br>%{x}<extra></extra>'
                ),
                row=rsi_row, col=1
            )

            # Mark breakout points
            for trendline, breakout, color in all_breakouts:
                # VALIDATION: Vérifier si le breakout est valide
                is_valid_breakout, validation_reason = self._validate_rsi_breakout(
                    df, df_with_emas, support_zones, breakout
                )

                if is_valid_breakout:
                    # BREAKOUT VALIDE = étoile verte
                    marker_color = '#00E676'
                    marker_symbol = 'star'
                    marker_size = 15
                    label = f"★ VALID {breakout.strength}"
                    annotation_text = f"★ {breakout.strength}"
                else:
                    # BREAKOUT NON VALIDE = cercle jaune (avertissement)
                    marker_color = '#FFC107'
                    marker_symbol = 'circle'
                    marker_size = 12
                    label = f"⚠ {breakout.strength} (non validé)"
                    annotation_text = f"⚠ {validation_reason}"

                fig.add_trace(
                    go.Scatter(
                        x=[breakout.date],
                        y=[breakout.rsi_value],
                        mode='markers',
                        name=label,
                        marker=dict(
                            symbol=marker_symbol,
                            size=marker_size,
                            color=marker_color,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=f'<b>RSI BREAKOUT</b><br>' +
                                     f'RSI: {breakout.rsi_value:.1f}<br>' +
                                     f'Strength: {breakout.strength}<br>' +
                                     f'Valid: {"✅ OUI" if is_valid_breakout else "❌ NON - " + validation_reason}<br>' +
                                     f'Age: {breakout.age_in_periods} periods<br>' +
                                     f'Date: %{{x}}<extra></extra>',
                        showlegend=False
                    ),
                    row=rsi_row, col=1
                )

                # Add annotation for breakout
                fig.add_annotation(
                    x=breakout.date,
                    y=breakout.rsi_value,
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=marker_color,
                    ax=30,
                    ay=-20,
                    font=dict(size=10, color=marker_color),
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor=marker_color,
                    borderwidth=1,
                    borderpad=3,
                    row=rsi_row, col=1
                )

            # Add STOP-LOSS line based on RSI minimum of the BREAKOUT trendline
            # IMPORTANT: Use the trendline that has a breakout, not just the best quality one
            if show_stop_loss and all_trendlines:
                # Find the trendline with breakout (same logic as screener)
                if all_breakouts:
                    # Use the first trendline with breakout (most significant one)
                    breakout_trendline = all_breakouts[0][0]  # (trendline, breakout, color)
                else:
                    # Fallback to best quality trendline if no breakout
                    breakout_trendline = all_trendlines[0]

                stop_loss_price, stop_source = self._calculate_stop_loss_from_trendline(
                    df, rsi, breakout_trendline
                )

                if stop_loss_price and stop_loss_price > 0:
                    # Add stop-loss line to main price chart (row=1)
                    fig.add_hline(
                        y=stop_loss_price,
                        line_dash="dash",
                        line_color="#FF5252",
                        line_width=2,
                        annotation_text=f"STOP ${stop_loss_price:.2f}",
                        annotation_position="right",
                        annotation_font_color="#FF5252",
                        annotation_font_size=11,
                        row=1, col=1
                    )

                    # Also mark the RSI minimum point on RSI chart
                    # Find the bar where RSI was minimum (using trendline.end_idx, not last peak)
                    start_idx = min(breakout_trendline.peak_indices)
                    end_idx = min(breakout_trendline.end_idx, len(df) - 1)  # Use trendline end
                    period_rsi = rsi.iloc[start_idx:end_idx + 1]
                    rsi_min_idx = period_rsi.idxmin()
                    rsi_min_value = float(period_rsi.min())

                    fig.add_trace(
                        go.Scatter(
                            x=[rsi_min_idx],
                            y=[rsi_min_value],
                            mode='markers',
                            name='RSI Min (Stop)',
                            marker=dict(
                                symbol='x',
                                size=12,
                                color='#FF5252',
                                line=dict(color='white', width=2)
                            ),
                            hovertemplate=f'<b>STOP LOSS LEVEL</b><br>' +
                                         f'RSI min: {rsi_min_value:.1f}<br>' +
                                         f'Low price: ${stop_loss_price:.2f}<br>' +
                                         f'{stop_source}<extra></extra>',
                            showlegend=False
                        ),
                        row=rsi_row, col=1
                    )

        except ImportError:
            # trendline_analysis module not available
            pass
        except Exception as e:
            # Log but don't fail the chart
            pass

        return fig

    def _validate_rsi_breakout(
        self,
        df: pd.DataFrame,
        df_with_emas: pd.DataFrame,
        support_zones: list,
        breakout
    ) -> tuple:
        """
        Valide si un breakout RSI est un vrai signal d'achat

        Conditions requises:
        1. EMAs croissantes: Au moins une EMA courte > EMA longue
        2. Prix proche d'un support EMA (dans les dernières périodes)

        Args:
            df: DataFrame OHLCV
            df_with_emas: DataFrame avec colonnes EMA
            support_zones: Liste des zones de support détectées
            breakout: Objet Breakout

        Returns:
            Tuple (is_valid: bool, reason: str)
        """
        # Si pas de données EMA, on ne peut pas valider
        if df_with_emas is None or df_with_emas.empty:
            return False, "Pas de données EMA"

        try:
            # Obtenir les valeurs EMA au moment du breakout
            breakout_idx = breakout.index
            if breakout_idx >= len(df_with_emas):
                breakout_idx = len(df_with_emas) - 1

            # Utiliser les EMAs au moment du breakout (pas les actuelles)
            ema_24 = float(df_with_emas['EMA_24'].iloc[breakout_idx])
            ema_38 = float(df_with_emas['EMA_38'].iloc[breakout_idx])
            ema_62 = float(df_with_emas['EMA_62'].iloc[breakout_idx])

            # CONDITION 1: Au moins une EMA courte > EMA longue
            ema_rising = (ema_24 > ema_38) or (ema_24 > ema_62) or (ema_38 > ema_62)

            if not ema_rising:
                return False, "EMAs non croissantes"

            # CONDITION 2: Prix proche d'un support (ou était proche récemment)
            # Vérifier si le prix a touché une zone de support dans les N dernières périodes
            if support_zones and len(support_zones) > 0:
                # Il y a au moins une zone de support proche
                has_support = True
            else:
                # Vérifier si le prix était proche d'une EMA récemment
                lookback_for_support = min(20, breakout_idx)  # 20 périodes avant le breakout
                has_support = False

                for i in range(max(0, breakout_idx - lookback_for_support), breakout_idx + 1):
                    if i >= len(df_with_emas):
                        continue

                    close_price = float(df['Close'].iloc[i] if not isinstance(df['Close'], pd.DataFrame)
                                       else df['Close'].iloc[i, 0])
                    ema_24_i = float(df_with_emas['EMA_24'].iloc[i])
                    ema_38_i = float(df_with_emas['EMA_38'].iloc[i])
                    ema_62_i = float(df_with_emas['EMA_62'].iloc[i])

                    # Distance au support le plus proche (une des EMAs)
                    min_ema = min(ema_24_i, ema_38_i, ema_62_i)
                    distance_pct = abs((close_price - min_ema) / min_ema * 100)

                    if distance_pct <= 8.0:  # Dans la zone de tolérance
                        has_support = True
                        break

                if not has_support:
                    return False, "Prix loin des supports"

            # Toutes les conditions sont remplies
            return True, "Signal validé"

        except Exception as e:
            return False, f"Erreur: {str(e)[:20]}"

    def _calculate_stop_loss_from_trendline(
        self,
        df: pd.DataFrame,
        rsi: pd.Series,
        trendline
    ) -> tuple:
        """
        Calculate stop-loss price based on LOW at RSI minimum during trendline period

        The stop-loss is the LOW price at the bar where RSI reached its MINIMUM
        during the RSI trendline period (from first peak to end of trendline/breakout).

        IMPORTANT: The period extends from first peak to trendline.end_idx (which
        represents the breakout/buy signal), NOT just to the last peak. The trendline
        continues beyond the last peak until it is broken.

        Args:
            df: DataFrame with OHLCV data
            rsi: RSI Series
            trendline: Trendline object with peak_indices and end_idx

        Returns:
            Tuple (stop_loss_price, stop_source_description)
        """
        try:
            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                low = df['Low'].iloc[:, 0]
            else:
                low = df['Low']

            # Get the range of the RSI trendline period
            # Use trendline.end_idx (breakout/current) NOT max(peak_indices)
            # The trendline continues beyond the last peak until breakout
            peak_indices = trendline.peak_indices
            start_idx = min(peak_indices)
            end_idx = min(trendline.end_idx, len(df) - 1)  # Use trendline end, not last peak

            # Find where RSI was at its MINIMUM during the trendline period
            period_rsi = rsi.iloc[start_idx:end_idx + 1]
            rsi_min_idx = period_rsi.idxmin()
            rsi_min_value = float(period_rsi.min())

            # Get the position index (integer) for iloc access
            if hasattr(rsi_min_idx, 'strftime'):
                # It's a datetime index, need to find its position
                rsi_min_pos = df.index.get_loc(rsi_min_idx)
            else:
                rsi_min_pos = rsi_min_idx

            # Stop-loss is the LOW price at the bar where RSI was minimum
            stop_loss = float(low.iloc[rsi_min_pos])

            # Get the date for description
            if hasattr(rsi_min_idx, 'strftime'):
                stop_date = rsi_min_idx.strftime('%Y-%m-%d')
            else:
                stop_date = str(rsi_min_idx)

            return (stop_loss, f"Low at RSI min {rsi_min_value:.0f} ({stop_date})")

        except Exception as e:
            return (0, f"Error: {str(e)[:30]}")

    def _add_crossover_zones(self, fig: go.Figure, crossovers: List[Dict],
                             index: pd.DatetimeIndex, close: pd.Series):
        """
        Add EMA crossover zones to the chart as vertical highlighted regions

        Args:
            fig: Plotly figure
            crossovers: List of crossover events with 'date', 'price', 'type' keys
            index: DataFrame index (dates)
            close: Close price series
        """
        # Colors for different crossover types
        bullish_color = 'rgba(38, 166, 154, 0.15)'  # Green transparent
        bearish_color = 'rgba(239, 83, 80, 0.15)'   # Red transparent

        for cross in crossovers:
            try:
                cross_date = cross.get('date')
                cross_price = cross.get('price', 0)
                cross_type = cross.get('type', 'bullish').lower()
                fast_ema = cross.get('fast_ema', '')
                slow_ema = cross.get('slow_ema', '')

                # Skip if no date
                if cross_date is None:
                    continue

                # Determine zone color based on crossover type
                if cross_type == 'bullish' or cross_type == 'golden':
                    zone_color = bullish_color
                    marker_color = self.colors['up_candle']
                else:
                    zone_color = bearish_color
                    marker_color = self.colors['down_candle']

                # Add vertical line at crossover date
                fig.add_vline(
                    x=cross_date,
                    line_dash="dot",
                    line_color=marker_color,
                    line_width=1,
                    opacity=0.7,
                    row=1, col=1
                )

                # Add marker at crossover point
                fig.add_trace(
                    go.Scatter(
                        x=[cross_date],
                        y=[cross_price],
                        mode='markers',
                        name=f'{cross_type.upper()} {fast_ema}/{slow_ema}',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color=marker_color,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=f'<b>{cross_type.upper()} Cross</b><br>' +
                                     f'EMA {fast_ema}/{slow_ema}<br>' +
                                     f'Price: ${cross_price:.2f}<br>' +
                                     f'Date: %{{x}}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # Add annotation
                fig.add_annotation(
                    x=cross_date,
                    y=cross_price,
                    text=f"{'▲' if cross_type == 'bullish' else '▼'} {fast_ema}/{slow_ema}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=marker_color,
                    ax=0,
                    ay=-40 if cross_type == 'bullish' else 40,
                    font=dict(size=9, color=marker_color),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor=marker_color,
                    borderwidth=1,
                    borderpad=2,
                    row=1, col=1
                )

            except Exception as e:
                # Skip invalid crossovers
                continue

    def _add_drawings(self, fig: go.Figure, drawings: List[Dict], index: pd.DatetimeIndex):
        """Add saved drawings to the chart"""
        for drawing in drawings:
            drawing_type = drawing.get('type', 'line')

            if drawing_type == 'hline':
                fig.add_hline(
                    y=drawing['y'],
                    line_dash=drawing.get('dash', 'solid'),
                    line_color=drawing.get('color', self.colors['support']),
                    line_width=drawing.get('width', 2),
                    annotation_text=drawing.get('label', ''),
                    annotation_position="right",
                    row=1, col=1
                )

            elif drawing_type == 'trendline':
                fig.add_shape(
                    type="line",
                    x0=drawing['x0'],
                    y0=drawing['y0'],
                    x1=drawing['x1'],
                    y1=drawing['y1'],
                    line=dict(
                        color=drawing.get('color', self.colors['support']),
                        width=drawing.get('width', 2),
                        dash=drawing.get('dash', 'solid')
                    ),
                    row=1, col=1
                )

            elif drawing_type == 'rect':
                fig.add_shape(
                    type="rect",
                    x0=drawing['x0'],
                    y0=drawing['y0'],
                    x1=drawing['x1'],
                    y1=drawing['y1'],
                    line=dict(
                        color=drawing.get('color', self.colors['support']),
                        width=1
                    ),
                    fillcolor=drawing.get('fillcolor', 'rgba(0, 230, 118, 0.1)'),
                    row=1, col=1
                )

    def add_support_resistance_levels(
        self,
        fig: go.Figure,
        levels: List[Dict],
        current_price: float
    ) -> go.Figure:
        """Add support and resistance levels to the chart"""
        for level in levels:
            price = level['price']
            level_type = level.get('type', 'support')
            strength = level.get('strength', 50)

            if level_type == 'support':
                color = self.colors['support']
                dash = 'solid' if strength > 70 else 'dash'
            else:
                color = self.colors['resistance']
                dash = 'solid' if strength > 70 else 'dash'

            width = 1 + (strength / 50)

            fig.add_hline(
                y=price,
                line_dash=dash,
                line_color=color,
                line_width=width,
                opacity=0.7,
                annotation_text=f"${price:.2f} ({strength:.0f}%)",
                annotation_position="left",
                row=1, col=1
            )

        return fig

    def save_drawings(self, symbol: str, drawings: List[Dict]):
        """Save drawings to file"""
        file_path = self.drawings_dir / f"{symbol}_drawings.json"
        with open(file_path, 'w') as f:
            json.dump(drawings, f, default=str)

    def load_drawings(self, symbol: str) -> List[Dict]:
        """Load drawings from file"""
        file_path = self.drawings_dir / f"{symbol}_drawings.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return []

    def create_mini_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        height: int = 200
    ) -> go.Figure:
        """Create a minimal chart for overview"""
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=close,
                mode='lines',
                name=symbol,
                line=dict(color=self.colors['ema_24'], width=1.5),
                fill='tozeroy',
                fillcolor='rgba(41, 98, 255, 0.1)'
            )
        )

        fig.update_layout(
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            title=dict(
                text=symbol,
                x=0.5,
                font=dict(size=12, color=self.colors['text'])
            ),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, side='right')
        )

        return fig

    # ==================== PERFORMANCE VISUALIZATION ====================

    def create_equity_curve_chart(
        self,
        equity_df: pd.DataFrame,
        initial_capital: float = 10000,
        height: int = 500
    ) -> go.Figure:
        """
        Create equity curve chart with drawdown

        Args:
            equity_df: DataFrame from TradeTracker.get_equity_curve()
            initial_capital: Starting capital
            height: Chart height

        Returns:
            Plotly figure
        """
        if equity_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade history yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown')
        )

        # Add initial point
        dates = [equity_df['date'].iloc[0]] + equity_df['date'].tolist()
        equities = [initial_capital] + equity_df['equity'].tolist()

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equities,
                mode='lines+markers',
                name='Equity',
                line=dict(color='#2962FF', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(41, 98, 255, 0.1)',
                hovertemplate='<b>Equity:</b> $%{y:,.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add initial capital line
        fig.add_hline(
            y=initial_capital,
            line_dash='dash',
            line_color='#9E9E9E',
            annotation_text=f'Initial: ${initial_capital:,.0f}',
            annotation_position='left',
            row=1, col=1
        )

        # Color trades by profit/loss
        for i, row in equity_df.iterrows():
            color = '#26a69a' if row['pnl'] >= 0 else '#ef5350'
            fig.add_trace(
                go.Scatter(
                    x=[row['date']],
                    y=[row['equity']],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='circle'),
                    name=row['symbol'],
                    hovertemplate=f"<b>{row['symbol']}</b><br>" +
                                 f"P&L: ${row['pnl']:,.2f} ({row['pnl_pct']:.1f}%)<br>" +
                                 f"Equity: ${row['equity']:,.2f}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )

        # Drawdown area
        drawdown_dates = equity_df['date'].tolist()
        drawdown_values = [-d for d in equity_df['drawdown_pct'].tolist()]  # Negative for display

        fig.add_trace(
            go.Scatter(
                x=drawdown_dates,
                y=drawdown_values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(239, 83, 80, 0.3)',
                line=dict(color='#ef5350', width=1),
                hovertemplate='<b>Drawdown:</b> %{y:.1f}%<br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )

        # Layout
        fig.update_layout(
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=60, r=40, t=60, b=40)
        )

        fig.update_xaxes(showgrid=True, gridcolor=self.colors['grid'])
        fig.update_yaxes(showgrid=True, gridcolor=self.colors['grid'])

        # Y-axis formatting
        fig.update_yaxes(tickprefix='$', tickformat=',.0f', row=1, col=1)
        fig.update_yaxes(ticksuffix='%', row=2, col=1)

        return fig

    def create_monthly_returns_chart(
        self,
        monthly_df: pd.DataFrame,
        height: int = 400
    ) -> go.Figure:
        """
        Create monthly returns bar chart

        Args:
            monthly_df: DataFrame from TradeTracker.get_monthly_returns()
            height: Chart height

        Returns:
            Plotly figure
        """
        if monthly_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No monthly data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        # Create month labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        labels = [f"{month_names[row['month']-1]} {row['year']}" for _, row in monthly_df.iterrows()]
        colors = ['#26a69a' if pnl >= 0 else '#ef5350' for pnl in monthly_df['pnl']]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=labels,
                y=monthly_df['pnl'],
                marker_color=colors,
                text=[f"${p:,.0f}" for p in monthly_df['pnl']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                             'P&L: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            )
        )

        # Add trade count as secondary info
        for i, (_, row) in enumerate(monthly_df.iterrows()):
            fig.add_annotation(
                x=labels[i],
                y=row['pnl'] + (50 if row['pnl'] >= 0 else -50),
                text=f"{row['trades']} trades",
                showarrow=False,
                font=dict(size=9, color=self.colors['text']),
                yshift=15 if row['pnl'] >= 0 else -15
            )

        fig.update_layout(
            title='Monthly P&L',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=60, r=40, t=60, b=60),
            yaxis=dict(
                title='P&L ($)',
                tickprefix='$',
                gridcolor=self.colors['grid']
            ),
            xaxis=dict(
                tickangle=-45,
                gridcolor=self.colors['grid']
            )
        )

        # Add zero line
        fig.add_hline(y=0, line_dash='dash', line_color='#9E9E9E', line_width=1)

        return fig

    def create_symbol_performance_chart(
        self,
        symbol_df: pd.DataFrame,
        height: int = 400
    ) -> go.Figure:
        """
        Create symbol performance horizontal bar chart

        Args:
            symbol_df: DataFrame from TradeTracker.get_symbol_performance()
            height: Chart height

        Returns:
            Plotly figure
        """
        if symbol_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No symbol data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        # Sort by P&L
        symbol_df = symbol_df.sort_values('pnl', ascending=True)
        colors = ['#26a69a' if pnl >= 0 else '#ef5350' for pnl in symbol_df['pnl']]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=symbol_df['symbol'],
                x=symbol_df['pnl'],
                orientation='h',
                marker_color=colors,
                text=[f"${p:,.0f}" for p in symbol_df['pnl']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                             'P&L: $%{x:,.2f}<br>' +
                             '<extra></extra>'
            )
        )

        # Add win rate annotation
        for i, (_, row) in enumerate(symbol_df.iterrows()):
            fig.add_annotation(
                x=row['pnl'],
                y=row['symbol'],
                text=f"{row['win_rate']:.0f}% ({row['trades']} trades)",
                showarrow=False,
                font=dict(size=9, color=self.colors['text']),
                xshift=60 if row['pnl'] >= 0 else -60
            )

        fig.update_layout(
            title='Performance by Symbol',
            height=max(height, len(symbol_df) * 35 + 100),
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=80, r=100, t=60, b=40),
            xaxis=dict(
                title='P&L ($)',
                tickprefix='$',
                gridcolor=self.colors['grid']
            ),
            yaxis=dict(gridcolor=self.colors['grid'])
        )

        # Add zero line
        fig.add_vline(x=0, line_dash='dash', line_color='#9E9E9E', line_width=1)

        return fig

    def create_win_loss_pie(
        self,
        wins: int,
        losses: int,
        height: int = 300
    ) -> go.Figure:
        """
        Create win/loss pie chart

        Args:
            wins: Number of winning trades
            losses: Number of losing trades
            height: Chart height

        Returns:
            Plotly figure
        """
        if wins == 0 and losses == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        fig = go.Figure()

        fig.add_trace(
            go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                hole=0.5,
                marker=dict(colors=['#26a69a', '#ef5350']),
                textinfo='percent+value',
                textfont=dict(size=14),
                hovertemplate='<b>%{label}</b>: %{value}<br>%{percent}<extra></extra>'
            )
        )

        # Add win rate in center
        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        fig.add_annotation(
            text=f"<b>{win_rate:.1f}%</b>",
            x=0.5, y=0.5,
            font=dict(size=24, color=self.colors['text']),
            showarrow=False
        )

        fig.update_layout(
            title='Win/Loss Ratio',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.1,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=20, r=20, t=50, b=50)
        )

        return fig

    def create_trade_distribution_chart(
        self,
        trades: list,
        height: int = 350
    ) -> go.Figure:
        """
        Create trade P&L distribution histogram

        Args:
            trades: List of Trade objects
            height: Chart height

        Returns:
            Plotly figure
        """
        if not trades:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        pnls = [t.pnl for t in trades if t.pnl != 0]
        if not pnls:
            return self.create_trade_distribution_chart([], height)

        # Determine colors based on positive/negative
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=pnls,
                nbinsx=20,
                marker=dict(
                    color=['#26a69a' if p >= 0 else '#ef5350' for p in sorted(pnls)],
                    line=dict(color='white', width=1)
                ),
                hovertemplate='P&L Range: $%{x}<br>Count: %{y}<extra></extra>'
            )
        )

        # Add mean line
        mean_pnl = np.mean(pnls)
        fig.add_vline(
            x=mean_pnl,
            line_dash='dash',
            line_color='#FFD700',
            annotation_text=f'Mean: ${mean_pnl:,.0f}',
            annotation_position='top'
        )

        # Add zero line
        fig.add_vline(x=0, line_dash='solid', line_color='#9E9E9E', line_width=2)

        fig.update_layout(
            title='P&L Distribution',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=60, r=40, t=60, b=40),
            xaxis=dict(
                title='P&L ($)',
                tickprefix='$',
                gridcolor=self.colors['grid']
            ),
            yaxis=dict(
                title='Number of Trades',
                gridcolor=self.colors['grid']
            )
        )

        return fig

    # ==================== SECTOR HEATMAP VISUALIZATION ====================

    def create_sector_heatmap(
        self,
        sector_data: list,
        metric: str = '1D %',
        height: int = 500
    ) -> go.Figure:
        """
        Create sector heatmap treemap

        Args:
            sector_data: List of SectorPerformance objects
            metric: Which metric to display ('1D %', '1W %', '1M %', 'YTD %')
            height: Chart height

        Returns:
            Plotly figure with treemap
        """
        if not sector_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        # Get performance values based on metric
        metric_map = {
            '1D %': 'perf_1d',
            '1W %': 'perf_1w',
            '1M %': 'perf_1m',
            'YTD %': 'perf_ytd'
        }
        attr = metric_map.get(metric, 'perf_1d')

        labels = []
        parents = []
        values = []
        colors = []
        hover_texts = []

        # Add root
        labels.append("Market")
        parents.append("")
        values.append(0)
        colors.append(0)
        hover_texts.append("")

        for s in sector_data:
            perf = getattr(s, attr, 0)
            labels.append(s.name)
            parents.append("Market")
            values.append(abs(perf) + 1)  # Size by absolute performance
            colors.append(perf)
            hover_texts.append(
                f"<b>{s.name}</b><br>"
                f"ETF: {s.etf}<br>"
                f"1D: {s.perf_1d:+.2f}%<br>"
                f"1W: {s.perf_1w:+.2f}%<br>"
                f"1M: {s.perf_1m:+.2f}%<br>"
                f"YTD: {s.perf_ytd:+.2f}%"
            )

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                colorscale='RdYlGn',
                cmid=0,
                showscale=True,
                colorbar=dict(
                    title=metric,
                    ticksuffix='%'
                )
            ),
            textinfo='label+text',
            text=[f"{c:+.1f}%" if c != 0 else "" for c in colors],
            textfont=dict(size=14),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.update_layout(
            title=f'Sector Performance ({metric})',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            margin=dict(l=10, r=10, t=50, b=10)
        )

        return fig

    def create_sector_bar_chart(
        self,
        sector_data: list,
        metric: str = '1D %',
        height: int = 400
    ) -> go.Figure:
        """
        Create sector performance bar chart

        Args:
            sector_data: List of SectorPerformance objects
            metric: Which metric to display
            height: Chart height

        Returns:
            Plotly figure with horizontal bar chart
        """
        if not sector_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        metric_map = {
            '1D %': 'perf_1d',
            '1W %': 'perf_1w',
            '1M %': 'perf_1m',
            'YTD %': 'perf_ytd'
        }
        attr = metric_map.get(metric, 'perf_1d')

        # Sort by performance
        sorted_data = sorted(sector_data, key=lambda x: getattr(x, attr, 0))

        sectors = [s.name for s in sorted_data]
        perfs = [getattr(s, attr, 0) for s in sorted_data]
        etfs = [s.etf for s in sorted_data]
        colors = ['#26a69a' if p >= 0 else '#ef5350' for p in perfs]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=sectors,
            x=perfs,
            orientation='h',
            marker_color=colors,
            text=[f"{p:+.2f}%" for p in perfs],
            textposition='outside',
            hovertemplate='<b>%{y}</b> (%{customdata})<br>Performance: %{x:+.2f}%<extra></extra>',
            customdata=etfs
        ))

        # Add zero line
        fig.add_vline(x=0, line_dash='solid', line_color='#9E9E9E', line_width=2)

        fig.update_layout(
            title=f'Sector Performance ({metric})',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=150, r=60, t=50, b=40),
            xaxis=dict(
                title='Performance (%)',
                ticksuffix='%',
                gridcolor=self.colors['grid'],
                zeroline=True,
                zerolinecolor='#9E9E9E'
            ),
            yaxis=dict(
                gridcolor=self.colors['grid']
            )
        )

        return fig

    def create_market_breadth_gauge(
        self,
        breadth_data: dict,
        height: int = 300
    ) -> go.Figure:
        """
        Create market breadth gauge

        Args:
            breadth_data: Dict with advancing, declining counts
            height: Chart height

        Returns:
            Plotly figure with gauge
        """
        advancing = breadth_data.get('advancing', 0)
        declining = breadth_data.get('declining', 0)
        total = breadth_data.get('total', 11)
        ratio = breadth_data.get('breadth_ratio', 1.0)

        # Calculate gauge value (0-100 scale)
        gauge_value = (advancing / total) * 100 if total > 0 else 50

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_value,
            title={'text': "Market Breadth", 'font': {'size': 16, 'color': self.colors['text']}},
            delta={'reference': 50, 'relative': False, 'valueformat': '.0f'},
            number={'suffix': '%', 'font': {'color': self.colors['text']}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': self.colors['text']},
                'bar': {'color': '#2962FF'},
                'bgcolor': self.colors['grid'],
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(239, 83, 80, 0.3)'},
                    {'range': [30, 70], 'color': 'rgba(158, 158, 158, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(38, 166, 154, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': '#FFD700', 'width': 4},
                    'thickness': 0.75,
                    'value': gauge_value
                }
            }
        ))

        # Add annotations
        fig.add_annotation(
            x=0.5, y=-0.15,
            text=f"Advancing: {advancing} | Declining: {declining}",
            showarrow=False,
            font=dict(size=12, color=self.colors['text']),
            xref='paper', yref='paper'
        )

        fig.update_layout(
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            margin=dict(l=30, r=30, t=50, b=50)
        )

        return fig

    def create_sector_rotation_chart(
        self,
        sector_data: list,
        height: int = 400
    ) -> go.Figure:
        """
        Create sector rotation scatter plot (1W vs 1M performance)

        Args:
            sector_data: List of SectorPerformance objects
            height: Chart height

        Returns:
            Plotly figure with scatter plot
        """
        if not sector_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        fig = go.Figure()

        # Quadrant background
        fig.add_shape(type="rect", x0=-20, x1=0, y0=0, y1=20,
                     fillcolor="rgba(255, 193, 7, 0.1)", line_width=0)  # Improving
        fig.add_shape(type="rect", x0=0, x1=20, y0=0, y1=20,
                     fillcolor="rgba(38, 166, 154, 0.1)", line_width=0)  # Leading
        fig.add_shape(type="rect", x0=0, x1=20, y0=-20, y1=0,
                     fillcolor="rgba(255, 152, 0, 0.1)", line_width=0)  # Weakening
        fig.add_shape(type="rect", x0=-20, x1=0, y0=-20, y1=0,
                     fillcolor="rgba(239, 83, 80, 0.1)", line_width=0)  # Lagging

        for s in sector_data:
            fig.add_trace(go.Scatter(
                x=[s.perf_1m],
                y=[s.perf_1w],
                mode='markers+text',
                name=s.name,
                text=[s.name[:10]],
                textposition='top center',
                marker=dict(
                    size=20,
                    color=s.color,
                    line=dict(color='white', width=1)
                ),
                hovertemplate=f"<b>{s.name}</b><br>"
                             f"1W: {s.perf_1w:+.2f}%<br>"
                             f"1M: {s.perf_1m:+.2f}%<extra></extra>"
            ))

        # Axis lines
        fig.add_hline(y=0, line_dash='dash', line_color='#9E9E9E')
        fig.add_vline(x=0, line_dash='dash', line_color='#9E9E9E')

        # Quadrant labels
        fig.add_annotation(x=-10, y=10, text="IMPROVING", showarrow=False,
                          font=dict(size=10, color='#FFC107'), opacity=0.7)
        fig.add_annotation(x=10, y=10, text="LEADING", showarrow=False,
                          font=dict(size=10, color='#26a69a'), opacity=0.7)
        fig.add_annotation(x=10, y=-10, text="WEAKENING", showarrow=False,
                          font=dict(size=10, color='#FF9800'), opacity=0.7)
        fig.add_annotation(x=-10, y=-10, text="LAGGING", showarrow=False,
                          font=dict(size=10, color='#ef5350'), opacity=0.7)

        fig.update_layout(
            title='Sector Rotation (1W vs 1M)',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=60, r=40, t=50, b=50),
            xaxis=dict(
                title='1-Month Performance (%)',
                ticksuffix='%',
                gridcolor=self.colors['grid'],
                range=[-15, 15]
            ),
            yaxis=dict(
                title='1-Week Performance (%)',
                ticksuffix='%',
                gridcolor=self.colors['grid'],
                range=[-10, 10]
            )
        )

        return fig

    # ==================== Economic Calendar Visualization ====================

    def create_calendar_timeline(
        self,
        events: list,
        height: int = 400
    ) -> go.Figure:
        """
        Create timeline view of economic events

        Args:
            events: List of EconomicEvent objects
            height: Chart height

        Returns:
            Plotly figure with timeline
        """
        if not events:
            fig = go.Figure()
            fig.add_annotation(
                text="No upcoming events",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        # Event type colors and symbols
        type_config = {
            'earnings': {'color': '#2962FF', 'symbol': 'circle', 'y': 3},
            'fomc': {'color': '#FF6D00', 'symbol': 'diamond', 'y': 2},
            'cpi': {'color': '#00C853', 'symbol': 'square', 'y': 1},
            'jobs': {'color': '#AB47BC', 'symbol': 'triangle-up', 'y': 0},
            'gdp': {'color': '#26A69A', 'symbol': 'hexagon', 'y': -1},
            'custom': {'color': '#9E9E9E', 'symbol': 'star', 'y': -2}
        }

        fig = go.Figure()

        # Group events by type
        event_groups = {}
        for event in events:
            event_type = event.event_type.value
            if event_type not in event_groups:
                event_groups[event_type] = []
            event_groups[event_type].append(event)

        # Add traces for each event type
        for event_type, type_events in event_groups.items():
            config = type_config.get(event_type, type_config['custom'])

            dates = [e.date for e in type_events]
            y_vals = [config['y']] * len(type_events)
            texts = [e.title for e in type_events]
            symbols = [e.symbol or '' for e in type_events]

            hover_texts = []
            for e in type_events:
                hover = f"<b>{e.title}</b><br>"
                hover += f"Date: {e.date.strftime('%Y-%m-%d %H:%M')}<br>"
                hover += f"Impact: {e.impact.value.upper()}<br>"
                if e.symbol:
                    hover += f"Symbol: {e.symbol}<br>"
                if e.description:
                    hover += f"{e.description[:50]}..."
                hover_texts.append(hover)

            # Size based on impact
            sizes = []
            for e in type_events:
                if e.impact.value == 'high':
                    sizes.append(20)
                elif e.impact.value == 'medium':
                    sizes.append(15)
                else:
                    sizes.append(10)

            fig.add_trace(go.Scatter(
                x=dates,
                y=y_vals,
                mode='markers',
                name=event_type.upper(),
                marker=dict(
                    size=sizes,
                    color=config['color'],
                    symbol=config['symbol'],
                    line=dict(color='white', width=1)
                ),
                text=texts,
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts
            ))

        # Y-axis labels
        y_tickvals = [3, 2, 1, 0, -1, -2]
        y_ticktext = ['Earnings', 'FOMC', 'CPI', 'Jobs', 'GDP', 'Custom']

        fig.update_layout(
            title='Economic Calendar Timeline',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=100, r=40, t=80, b=50),
            xaxis=dict(
                title='Date',
                gridcolor=self.colors['grid'],
                type='date'
            ),
            yaxis=dict(
                tickvals=y_tickvals,
                ticktext=y_ticktext,
                gridcolor=self.colors['grid'],
                range=[-3, 4]
            )
        )

        # Add today line
        from datetime import datetime
        today = datetime.now()
        fig.add_shape(
            type="line",
            x0=today, x1=today,
            y0=-3, y1=4,
            line=dict(color='#FFC107', dash='dash', width=2)
        )
        fig.add_annotation(
            x=today, y=4,
            text="Today",
            showarrow=False,
            font=dict(color='#FFC107')
        )

        return fig

    def create_calendar_table(
        self,
        events: list,
        height: int = 400
    ) -> go.Figure:
        """
        Create table view of economic events

        Args:
            events: List of EconomicEvent objects
            height: Chart height

        Returns:
            Plotly figure with table
        """
        if not events:
            fig = go.Figure()
            fig.add_annotation(
                text="No upcoming events",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background']
            )
            return fig

        # Prepare table data
        dates = []
        event_types = []
        titles = []
        impacts = []
        symbols = []

        for e in events:
            dates.append(e.date.strftime('%Y-%m-%d'))
            event_types.append(e.event_type.value.upper())
            titles.append(e.title)
            impacts.append(e.impact.value.upper())
            symbols.append(e.symbol or '-')

        # Impact colors
        impact_colors = []
        for imp in impacts:
            if imp == 'HIGH':
                impact_colors.append('#ef5350')
            elif imp == 'MEDIUM':
                impact_colors.append('#FFC107')
            else:
                impact_colors.append('#26a69a')

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Date</b>', '<b>Type</b>', '<b>Event</b>', '<b>Impact</b>', '<b>Symbol</b>'],
                fill_color='#1E1E1E',
                align='left',
                font=dict(color='white', size=12),
                height=35
            ),
            cells=dict(
                values=[dates, event_types, titles, impacts, symbols],
                fill_color=[
                    ['#121212'] * len(dates),
                    ['#121212'] * len(dates),
                    ['#121212'] * len(dates),
                    impact_colors,
                    ['#121212'] * len(dates)
                ],
                align='left',
                font=dict(color='white', size=11),
                height=30
            )
        )])

        fig.update_layout(
            title='Upcoming Economic Events',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            margin=dict(l=10, r=10, t=50, b=10)
        )

        return fig

    def create_earnings_calendar(
        self,
        earnings_data: list,
        height: int = 400
    ) -> go.Figure:
        """
        Create earnings-specific calendar view

        Args:
            earnings_data: List of dicts with symbol and earnings_date
            height: Chart height

        Returns:
            Plotly figure with earnings timeline
        """
        if not earnings_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No upcoming earnings",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background']
            )
            return fig

        from datetime import datetime

        # Filter valid data
        valid_data = [d for d in earnings_data if d.get('earnings_date')]

        # Sort by date
        valid_data.sort(key=lambda x: x['earnings_date'])

        fig = go.Figure()

        dates = [d['earnings_date'] for d in valid_data]
        symbols = [d['symbol'] for d in valid_data]
        days_until = [d.get('days_until', 0) for d in valid_data]

        # Color by urgency
        colors = []
        for days in days_until:
            if days is None:
                colors.append('#9E9E9E')
            elif days <= 3:
                colors.append('#ef5350')  # Red - very soon
            elif days <= 7:
                colors.append('#FFC107')  # Yellow - soon
            else:
                colors.append('#26a69a')  # Green - not urgent

        hover_texts = []
        for d in valid_data:
            days = d.get('days_until', 'N/A')
            if days == 0:
                urgency = 'TODAY!'
            elif days == 1:
                urgency = 'TOMORROW'
            elif days and days <= 7:
                urgency = f'In {days} days'
            else:
                urgency = f'In {days} days' if days else 'Date unknown'

            hover_texts.append(
                f"<b>{d['symbol']}</b><br>"
                f"Date: {d['earnings_date'].strftime('%Y-%m-%d')}<br>"
                f"{urgency}"
            )

        fig.add_trace(go.Scatter(
            x=dates,
            y=list(range(len(dates))),
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=symbols,
            textposition='middle right',
            textfont=dict(size=10, color='white'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        # Today line using shape to avoid vline datetime issue
        today = datetime.now()
        y_max = len(valid_data) if valid_data else 1
        fig.add_shape(
            type="line",
            x0=today, x1=today,
            y0=-0.5, y1=y_max,
            line=dict(color='#FFC107', dash='dash', width=2)
        )
        fig.add_annotation(
            x=today, y=y_max,
            text="Today",
            showarrow=False,
            font=dict(color='#FFC107')
        )

        fig.update_layout(
            title='Upcoming Earnings Dates',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=40, r=100, t=50, b=50),
            xaxis=dict(
                title='Date',
                gridcolor=self.colors['grid'],
                type='date'
            ),
            yaxis=dict(
                showticklabels=False,
                gridcolor=self.colors['grid']
            )
        )

        return fig

    def create_week_calendar(
        self,
        week_events: dict,
        height: int = 300
    ) -> go.Figure:
        """
        Create weekly calendar view

        Args:
            week_events: Dict with day names as keys and event lists as values
            height: Chart height

        Returns:
            Plotly figure with weekly view
        """
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        fig = go.Figure()

        # Type colors
        type_colors = {
            'earnings': '#2962FF',
            'fomc': '#FF6D00',
            'cpi': '#00C853',
            'jobs': '#AB47BC',
            'gdp': '#26A69A',
            'custom': '#9E9E9E'
        }

        for i, day in enumerate(days):
            events = week_events.get(day, [])

            if events:
                for j, event in enumerate(events[:3]):  # Max 3 events per day
                    color = type_colors.get(event.event_type.value, '#9E9E9E')

                    fig.add_trace(go.Scatter(
                        x=[i],
                        y=[j],
                        mode='markers+text',
                        marker=dict(
                            size=30,
                            color=color,
                            symbol='square'
                        ),
                        text=[event.event_type.value[:3].upper()],
                        textfont=dict(size=8, color='white'),
                        hovertemplate=f"<b>{event.title}</b><br>{event.date.strftime('%H:%M')}<extra></extra>",
                        showlegend=False
                    ))
            else:
                # Empty day marker
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[0],
                    mode='markers',
                    marker=dict(
                        size=30,
                        color='rgba(158, 158, 158, 0.2)',
                        symbol='square'
                    ),
                    hovertemplate="No events<extra></extra>",
                    showlegend=False
                ))

        fig.update_layout(
            title='This Week',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            margin=dict(l=40, r=40, t=50, b=50),
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(5)),
                ticktext=days,
                gridcolor=self.colors['grid']
            ),
            yaxis=dict(
                showticklabels=False,
                range=[-0.5, 3],
                gridcolor=self.colors['grid']
            )
        )

        return fig

    def create_event_impact_chart(
        self,
        events: list,
        height: int = 300
    ) -> go.Figure:
        """
        Create chart showing events by impact level

        Args:
            events: List of EconomicEvent objects
            height: Chart height

        Returns:
            Plotly figure with impact distribution
        """
        if not events:
            fig = go.Figure()
            fig.add_annotation(
                text="No events to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=self.colors['text'])
            )
            fig.update_layout(
                height=height,
                template='plotly_dark',
                paper_bgcolor=self.colors['background']
            )
            return fig

        # Count by impact
        impact_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for event in events:
            impact_counts[event.impact.value.upper()] += 1

        fig = go.Figure(data=[go.Pie(
            labels=list(impact_counts.keys()),
            values=list(impact_counts.values()),
            hole=0.4,
            marker=dict(colors=['#ef5350', '#FFC107', '#26a69a']),
            textinfo='label+value',
            textfont=dict(size=12, color='white')
        )])

        fig.update_layout(
            title='Events by Impact',
            height=height,
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        return fig


# Singleton instance
interactive_chart_builder = InteractiveChartBuilder()
