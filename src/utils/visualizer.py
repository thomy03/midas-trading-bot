"""
Chart visualization module using Plotly for TradingView-like charts
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from datetime import datetime
from src.indicators.ema_analyzer import ema_analyzer
from src.data.market_data import market_data_fetcher
from src.utils.logger import logger


class ChartVisualizer:
    """Creates interactive TradingView-like charts"""

    def __init__(self):
        """Initialize the visualizer"""
        self.ema_analyzer = ema_analyzer
        self.market_data = market_data_fetcher

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_exit_signal(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        Detect exit signal based on RSI crossing below EMA when EMA < 50

        Rules:
        - RSI 14 periods
        - EMA 9 and EMA 14 on RSI
        - Exit when: At least one EMA < 50 AND RSI crosses below that EMA
        - Important: RSI must cross (not already be below)
        """
        if current_idx < 15:  # Need history for RSI and EMAs
            return False

        # Get current and previous RSI values
        current_rsi = df['RSI'].iloc[current_idx]
        prev_rsi = df['RSI'].iloc[current_idx - 1]

        # Get current RSI EMAs
        current_rsi_ema9 = df['RSI_EMA9'].iloc[current_idx]
        current_rsi_ema14 = df['RSI_EMA14'].iloc[current_idx]
        prev_rsi_ema9 = df['RSI_EMA9'].iloc[current_idx - 1]
        prev_rsi_ema14 = df['RSI_EMA14'].iloc[current_idx - 1]

        # Check if RSI crosses below EMA9 (when EMA9 < 50)
        if current_rsi_ema9 < 50:
            # RSI was above EMA9 and now crosses below
            if prev_rsi >= prev_rsi_ema9 and current_rsi < current_rsi_ema9:
                return True

        # Check if RSI crosses below EMA14 (when EMA14 < 50)
        if current_rsi_ema14 < 50:
            # RSI was above EMA14 and now crosses below
            if prev_rsi >= prev_rsi_ema14 and current_rsi < current_rsi_ema14:
                return True

        return False

    def detect_historical_signals(self, df: pd.DataFrame, crossovers: List[Dict], timeframe: str) -> List[Dict]:
        """
        Detect historical buy signals as TRADES (entry to exit)

        Args:
            df: DataFrame with price data and EMAs
            crossovers: List of EMA crossovers (not used anymore)
            timeframe: 'daily' or 'weekly'

        Returns:
            List of trade zones (entry to exit)
        """
        # Calculate RSI and its EMAs
        df['RSI'] = self.calculate_rsi(df, period=14)
        df['RSI_EMA9'] = df['RSI'].ewm(span=9, adjust=False).mean()
        df['RSI_EMA14'] = df['RSI'].ewm(span=14, adjust=False).mean()

        trades = []
        in_trade = False
        trade_start_idx = None
        trade_type = None
        trade_support_ema = None  # Track which EMA was the support for exit detection

        # Only check points where we have all indicators calculated
        start_idx = 62  # Start after EMA 62 is fully calculated

        # Adjusted thresholds based on timeframe
        if timeframe == 'weekly':
            strong_buy_threshold = 1.5  # <= 1.5%
            buy_threshold = 3.5          # <= 3.5%
            watch_threshold = 6.0        # <= 6%
        else:
            strong_buy_threshold = 1.0   # <= 1%
            buy_threshold = 2.0          # <= 2%
            watch_threshold = 4.0        # <= 4%

        for i in range(start_idx, len(df)):
            current_price = float(df['Close'].iloc[i])
            ema24 = float(df['EMA_24'].iloc[i])
            ema38 = float(df['EMA_38'].iloc[i])
            ema62 = float(df['EMA_62'].iloc[i])

            # Check for EXIT if we're in a trade
            if in_trade:
                # ONLY exit condition: RSI exit signal
                rsi_exit = self.detect_exit_signal(df, i)

                if rsi_exit:
                    # Close the trade
                    trades.append({
                        'type': trade_type,
                        'start_date': df.index[trade_start_idx],
                        'end_date': df.index[i],
                        'entry_price': float(df['Close'].iloc[trade_start_idx]),
                        'exit_price': current_price,
                        'start_idx': trade_start_idx,
                        'end_idx': i,
                        'exit_reason': 'RSI'
                    })
                    in_trade = False
                    trade_start_idx = None
                    trade_type = None
                    trade_support_ema = None
                continue  # Don't look for new entries while in trade

            # Check for ENTRY if not in a trade
            if not in_trade:
                # Check EMA alignment (at least 2 EMAs in bullish order)
                alignment_count = 0
                if ema24 > ema38:
                    alignment_count += 1
                if ema24 > ema62:
                    alignment_count += 1
                if ema38 > ema62:
                    alignment_count += 1

                if alignment_count < 2:
                    continue  # Not aligned enough

                # IMPORTANT: Bougie doit clôturer en baissière (Close < Open)
                current_open = float(df['Open'].iloc[i])
                is_bearish_candle = current_price < current_open

                if not is_bearish_candle:
                    continue  # Signal only on bearish candles

                # IMPORTANT: Use LOW (wick) to detect touch, not Close
                current_low = float(df['Low'].iloc[i])

                # Calculate distance from LOW to each EMA (to detect wick touching)
                # Positive = low above EMA, Negative = low below EMA (wick touched/broke support)
                distances_low_to_emas = [
                    ((current_low - ema24) / ema24) * 100,  # % distance from LOW to EMA24
                    ((current_low - ema38) / ema38) * 100,  # % distance from LOW to EMA38
                    ((current_low - ema62) / ema62) * 100   # % distance from LOW to EMA62
                ]

                # Find the EMA that the LOW is closest to (considering both above and below)
                min_distance_idx = min(range(len(distances_low_to_emas)), key=lambda x: abs(distances_low_to_emas[x]))
                distance_low_to_nearest_ema = distances_low_to_emas[min_distance_idx]
                nearest_ema_value = [ema24, ema38, ema62][min_distance_idx]

                # Determine signal strength based on distance from LOW
                # Negative distance (low below EMA) = STRONG signal (wick touched/broke support)
                # Small positive distance (low above EMA) = Good signal (approaching support)
                recommendation = None

                if distance_low_to_nearest_ema <= 0:
                    # LOW AT or BELOW EMA = strongest signal (wick touched/broke support)
                    if abs(distance_low_to_nearest_ema) <= 3.0:  # Within 3% below
                        recommendation = 'STRONG_BUY'
                    elif abs(distance_low_to_nearest_ema) <= 5.0:  # Within 5% below
                        recommendation = 'BUY'
                else:
                    # LOW ABOVE EMA = approaching support
                    if distance_low_to_nearest_ema <= strong_buy_threshold:
                        recommendation = 'STRONG_BUY'
                    elif distance_low_to_nearest_ema <= buy_threshold:
                        recommendation = 'BUY'
                    elif distance_low_to_nearest_ema <= watch_threshold:
                        recommendation = 'WATCH'

                if recommendation:
                    # Start a new trade
                    in_trade = True
                    trade_start_idx = i
                    trade_type = recommendation
                    trade_support_ema = nearest_ema_value  # Remember which EMA was the support

        # Close any open trade at the end
        if in_trade:
            trades.append({
                'type': trade_type,
                'start_date': df.index[trade_start_idx],
                'end_date': df.index[-1],
                'entry_price': float(df['Close'].iloc[trade_start_idx]),
                'exit_price': float(df['Close'].iloc[-1]),
                'start_idx': trade_start_idx,
                'end_idx': len(df) - 1,
                'exit_reason': 'Ongoing'  # Trade still active
            })

        return trades

    def create_chart(
        self,
        symbol: str,
        timeframe: str = 'daily',
        period: str = '1y',
        show_volume: bool = True
    ) -> Optional[go.Figure]:
        """
        Create an interactive chart with EMAs and support zones

        Args:
            symbol: Stock symbol
            timeframe: 'daily' or 'weekly'
            period: Data period ('1y', '2y', '6mo', etc.)
            show_volume: Whether to show volume subplot

        Returns:
            Plotly Figure object or None if failed
        """
        try:
            # Get interval based on timeframe
            interval = '1wk' if timeframe == 'weekly' else '1d'

            # Fetch historical data
            df = self.market_data.get_historical_data(symbol, period=period, interval=interval)

            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return None

            # Calculate EMAs
            df = self.ema_analyzer.calculate_emas(df)

            # Detect crossovers
            crossovers = self.ema_analyzer.detect_crossovers(df, timeframe)

            # Get current price
            current_price = float(df['Close'].iloc[-1])

            # Check EMA alignment
            is_aligned, alignment_desc = self.ema_analyzer.check_ema_alignment(df, for_buy=True)

            # Find support zones
            support_zones = self.ema_analyzer.find_support_zones(df, crossovers, current_price)

            # Detect historical signals
            historical_signals = self.detect_historical_signals(df, crossovers, timeframe)

            # Create figure with subplots: Price, RSI, Volume
            if show_volume:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=(f'{symbol} - {timeframe.upper()}', 'RSI (14)', 'Volume')
                )
            else:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f'{symbol} - {timeframe.upper()}', 'RSI (14)')
                )

            # Add candlestick chart
            candlestick = go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            )

            if show_volume:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)

            # Add EMAs
            ema_colors = {
                'EMA_24': '#2962FF',  # Blue
                'EMA_38': '#FF6D00',  # Orange
                'EMA_62': '#E91E63'   # Pink
            }

            for ema_col, color in ema_colors.items():
                if ema_col in df.columns:
                    ema_trace = go.Scatter(
                        x=df.index,
                        y=df[ema_col],
                        name=ema_col.replace('_', ' '),
                        line=dict(color=color, width=2),
                        hovertemplate='%{y:.2f}<extra></extra>'
                    )
                    if show_volume:
                        fig.add_trace(ema_trace, row=1, col=1)
                    else:
                        fig.add_trace(ema_trace)

            # Add crossover markers
            if crossovers:
                for crossover in crossovers[:10]:  # Show last 10 crossovers
                    cross_date = crossover['date']
                    cross_price = crossover['price']
                    cross_type = crossover['type']

                    # Find the closest date in the dataframe
                    if isinstance(cross_date, pd.Timestamp):
                        marker_color = '#00C853' if cross_type == 'bullish' else '#D50000'
                        marker_symbol = 'triangle-up' if cross_type == 'bullish' else 'triangle-down'

                        marker = go.Scatter(
                            x=[cross_date],
                            y=[cross_price],
                            mode='markers',
                            name=f'EMA {crossover["fast_ema"]}/{crossover["slow_ema"]} Cross',
                            marker=dict(
                                size=12,
                                color=marker_color,
                                symbol=marker_symbol,
                                line=dict(color='white', width=2)
                            ),
                            hovertemplate=f'Crossover: EMA {crossover["fast_ema"]}/{crossover["slow_ema"]}<br>' +
                                        f'Price: %{{y:.2f}}<br>' +
                                        f'{crossover["days_ago"]} days ago<extra></extra>',
                            showlegend=False
                        )
                        if show_volume:
                            fig.add_trace(marker, row=1, col=1)
                        else:
                            fig.add_trace(marker)

            # Add trade zones (entry to exit)
            if historical_signals:  # historical_signals now contains trades
                signal_colors = {
                    'STRONG_BUY': 'rgba(0, 200, 83, 0.3)',    # Green
                    'BUY': 'rgba(100, 221, 23, 0.25)',         # Light green
                    'WATCH': 'rgba(253, 216, 53, 0.2)'         # Yellow
                }

                signal_marker_colors = {
                    'STRONG_BUY': '#00C853',
                    'BUY': '#64DD17',
                    'WATCH': '#FDD835'
                }

                # Draw each trade as a zone from entry to exit
                for trade in historical_signals:
                    # Add vertical shaded region for the entire trade duration
                    zone_shape = go.Scatter(
                        x=[trade['start_date'], trade['start_date'], trade['end_date'], trade['end_date'], trade['start_date']],
                        y=[df['Low'].min(), df['High'].max(), df['High'].max(), df['Low'].min(), df['Low'].min()],
                        fill='toself',
                        fillcolor=signal_colors[trade['type']],
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                    if show_volume:
                        fig.add_trace(zone_shape, row=1, col=1)
                    else:
                        fig.add_trace(zone_shape)

                    # Calculate trade performance
                    trade_duration = (trade['end_date'] - trade['start_date']).days if hasattr(trade['end_date'] - trade['start_date'], 'days') else 0
                    trade_return = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100

                    # Add entry marker (star at entry)
                    entry_marker = go.Scatter(
                        x=[trade['start_date']],
                        y=[trade['entry_price']],
                        mode='markers+text',
                        name=f'{trade["type"]} Entry',
                        marker=dict(
                            size=15,
                            color=signal_marker_colors[trade['type']],
                            symbol='star',
                            line=dict(color='white', width=2)
                        ),
                        text=['ENTRY'],
                        textposition='top center',
                        textfont=dict(size=9, color=signal_marker_colors[trade['type']]),
                        hovertemplate=f'<b>{trade["type"]} ENTRY</b><br>' +
                                    f'Date: {trade["start_date"].strftime("%Y-%m-%d")}<br>' +
                                    f'Price: ${trade["entry_price"]:.2f}<br>' +
                                    f'Duration: {trade_duration} days<br>' +
                                    f'Return: {trade_return:+.2f}%<br>' +
                                    f'<extra></extra>',
                        showlegend=True,
                        legendgroup=trade['type']
                    )
                    if show_volume:
                        fig.add_trace(entry_marker, row=1, col=1)
                    else:
                        fig.add_trace(entry_marker)

                    # Add exit marker (X at exit)
                    exit_reason = trade.get('exit_reason', 'Unknown')
                    exit_symbol = 'circle' if exit_reason == 'Ongoing' else 'x'
                    exit_text = 'ACTIVE' if exit_reason == 'Ongoing' else 'EXIT'

                    exit_marker = go.Scatter(
                        x=[trade['end_date']],
                        y=[trade['exit_price']],
                        mode='markers+text',
                        name=f'{trade["type"]} Exit',
                        marker=dict(
                            size=12,
                            color='orange' if exit_reason == 'Ongoing' else ('red' if trade_return < 0 else 'green'),
                            symbol=exit_symbol,
                            line=dict(color='white', width=2)
                        ),
                        text=[exit_text],
                        textposition='bottom center',
                        textfont=dict(size=9, color='orange' if exit_reason == 'Ongoing' else ('red' if trade_return < 0 else 'green')),
                        hovertemplate=f'<b>{exit_text}</b><br>' +
                                    f'Date: {trade["end_date"].strftime("%Y-%m-%d")}<br>' +
                                    f'Price: ${trade["exit_price"]:.2f}<br>' +
                                    f'Return: {trade_return:+.2f}%<br>' +
                                    f'Exit: {exit_reason}<br>' +
                                    f'<extra></extra>',
                        showlegend=False
                    )
                    if show_volume:
                        fig.add_trace(exit_marker, row=1, col=1)
                    else:
                        fig.add_trace(exit_marker)

            # Add support zones as horizontal lines
            if support_zones:
                for i, zone in enumerate(support_zones[:5]):  # Show top 5 support zones
                    support_level = zone['level']
                    strength = zone['strength']

                    # Add horizontal line
                    line_trace = go.Scatter(
                        x=[df.index[0], df.index[-1]],
                        y=[support_level, support_level],
                        mode='lines',
                        name=f'Support ${support_level:.2f} (Strength: {strength:.0f}%)',
                        line=dict(
                            color='rgba(0, 200, 83, 0.5)',
                            width=2,
                            dash='dash'
                        ),
                        hovertemplate=f'Support: ${support_level:.2f}<br>Strength: {strength:.0f}%<extra></extra>'
                    )
                    if show_volume:
                        fig.add_trace(line_trace, row=1, col=1)
                    else:
                        fig.add_trace(line_trace)

                    # Add shaded area (zone tolerance)
                    from config.settings import ZONE_TOLERANCE
                    zone_upper = support_level * (1 + ZONE_TOLERANCE / 100)
                    zone_lower = support_level * (1 - ZONE_TOLERANCE / 100)

                    zone_area = go.Scatter(
                        x=list(df.index) + list(df.index[::-1]),
                        y=[zone_upper] * len(df.index) + [zone_lower] * len(df.index),
                        fill='toself',
                        fillcolor='rgba(0, 200, 83, 0.1)',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                    if show_volume:
                        fig.add_trace(zone_area, row=1, col=1)
                    else:
                        fig.add_trace(zone_area)

            # Add RSI subplot (row 2)
            # RSI line
            rsi_line = go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='#2962FF', width=2),
                showlegend=True
            )
            rsi_row = 2
            fig.add_trace(rsi_line, row=rsi_row, col=1)

            # RSI EMA 9
            rsi_ema9 = go.Scatter(
                x=df.index,
                y=df['RSI_EMA9'],
                name='RSI EMA9',
                line=dict(color='#FF6D00', width=1.5),
                showlegend=True
            )
            fig.add_trace(rsi_ema9, row=rsi_row, col=1)

            # RSI EMA 14
            rsi_ema14 = go.Scatter(
                x=df.index,
                y=df['RSI_EMA14'],
                name='RSI EMA14',
                line=dict(color='#00E676', width=1.5),
                showlegend=True
            )
            fig.add_trace(rsi_ema14, row=rsi_row, col=1)

            # Add RSI reference lines (50, 30, 70)
            for level, color, name in [(50, 'gray', '50'), (30, 'green', '30'), (70, 'red', '70')]:
                fig.add_hline(y=level, line_dash="dash", line_color=color, line_width=1,
                             annotation_text=name, annotation_position="right",
                             row=rsi_row, col=1)

            # Add volume bars if requested (row 3)
            if show_volume:
                colors = ['#26a69a' if close >= open else '#ef5350'
                         for close, open in zip(df['Close'], df['Open'])]

                volume_bars = go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                )
                fig.add_trace(volume_bars, row=3, col=1)

            # Update layout
            title_text = f'{symbol} - {timeframe.upper()}'
            if is_aligned:
                title_text += f' | ✅ EMAs Aligned ({alignment_desc})'
            else:
                title_text += f' | ⚠️ EMAs Not Aligned ({alignment_desc})'

            fig.update_layout(
                title=dict(
                    text=title_text,
                    font=dict(size=20)
                ),
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                template='plotly_dark',
                height=1000 if show_volume else 800,  # Increased for RSI subplot
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Update axes
            if show_volume:
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                fig.update_yaxes(title_text="Volume", row=3, col=1)
                fig.update_xaxes(title_text="Date", row=3, col=1)
            else:
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                fig.update_xaxes(title_text="Date", row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            return None

    def create_comparison_chart(
        self,
        symbols: List[str],
        timeframe: str = 'daily',
        period: str = '1y'
    ) -> Optional[go.Figure]:
        """
        Create a comparison chart for multiple symbols (normalized)

        Args:
            symbols: List of stock symbols to compare
            timeframe: 'daily' or 'weekly'
            period: Data period

        Returns:
            Plotly Figure object or None if failed
        """
        try:
            fig = go.Figure()

            interval = '1wk' if timeframe == 'weekly' else '1d'

            for symbol in symbols:
                df = self.market_data.get_historical_data(symbol, period=period, interval=interval)

                if df is None or df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Normalize to percentage change from start
                normalized = (df['Close'] / df['Close'].iloc[0] - 1) * 100

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    name=symbol,
                    mode='lines',
                    hovertemplate=f'{symbol}: %{{y:.2f}}%<extra></extra>'
                ))

            fig.update_layout(
                title=f'Price Comparison - {timeframe.upper()} ({period})',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                hovermode='x unified',
                template='plotly_dark',
                height=600,
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return None

    def create_alert_summary_chart(self, alerts: List[Dict]) -> Optional[go.Figure]:
        """
        Create a summary chart showing alert distribution

        Args:
            alerts: List of alert dictionaries

        Returns:
            Plotly Figure object or None if failed
        """
        try:
            if not alerts:
                return None

            # Count by recommendation
            recommendations = {}
            for alert in alerts:
                rec = alert.get('recommendation', 'UNKNOWN')
                recommendations[rec] = recommendations.get(rec, 0) + 1

            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(recommendations.keys()),
                values=list(recommendations.values()),
                hole=0.3,
                marker_colors=['#D50000', '#FF6D00', '#FDD835', '#00C853']
            )])

            fig.update_layout(
                title='Alerts by Recommendation',
                template='plotly_dark',
                height=400
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating alert summary chart: {e}")
            return None


# Singleton instance
chart_visualizer = ChartVisualizer()
