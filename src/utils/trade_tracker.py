"""
Trade Tracker for TradingBot V3

Tracks trade history and calculates performance metrics:
- P&L (realized and unrealized)
- Win rate
- Sharpe ratio
- Max drawdown
- Equity curve
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    trade_id: str
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    target_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'target', 'stop_loss', 'manual', 'trailing_stop'
    status: str = 'open'  # 'open', 'closed'
    notes: str = ''

    @property
    def is_open(self) -> bool:
        return self.status == 'open'

    @property
    def pnl(self) -> float:
        """Calculate P&L (0 if still open)"""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage"""
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0

    @property
    def risk_reward_actual(self) -> Optional[float]:
        """Calculate actual risk/reward achieved"""
        if self.exit_price is None:
            return None
        risk = self.entry_price - self.stop_loss
        if risk <= 0:
            return None
        reward = self.exit_price - self.entry_price
        return reward / risk


class TradeTracker:
    """Tracks trades and calculates performance metrics"""

    def __init__(self, data_file: str = "data/trades.json", initial_capital: float = 10000):
        """
        Initialize TradeTracker

        Args:
            data_file: Path to JSON file for persistence
            initial_capital: Starting capital for equity curve
        """
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self._load()

    def _load(self):
        """Load trades from JSON file"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.initial_capital = data.get('initial_capital', self.initial_capital)
                    trades_data = data.get('trades', [])
                    self.trades = [Trade(**t) for t in trades_data]
                    logger.info(f"Loaded {len(self.trades)} trades")
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.error(f"Error loading trades: {e}")
                self.trades = []
        else:
            self.trades = []
            self._save()
            logger.info("Created new trade history file")

    def _save(self):
        """Save trades to JSON file"""
        try:
            data = {
                'initial_capital': self.initial_capital,
                'trades': [asdict(t) for t in self.trades],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except IOError as e:
            logger.error(f"Error saving trades: {e}")

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        count = len(self.trades) + 1
        return f"T{timestamp}_{count:04d}"

    def open_trade(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        target_price: Optional[float] = None,
        entry_date: Optional[str] = None,
        notes: str = ''
    ) -> Trade:
        """
        Open a new trade

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            shares: Number of shares
            stop_loss: Stop loss price
            target_price: Optional target price
            entry_date: Entry date (defaults to today)
            notes: Optional notes

        Returns:
            Created Trade object
        """
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=symbol.upper(),
            entry_date=entry_date or date.today().isoformat(),
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            target_price=target_price,
            notes=notes,
            status='open'
        )
        self.trades.append(trade)
        self._save()
        logger.info(f"Opened trade {trade.trade_id}: {symbol} @ {entry_price}")
        return trade

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = 'manual',
        exit_date: Optional[str] = None
    ) -> Optional[Trade]:
        """
        Close an open trade

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_reason: Reason for exit ('target', 'stop_loss', 'manual', 'trailing_stop')
            exit_date: Exit date (defaults to today)

        Returns:
            Closed Trade or None if not found
        """
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.is_open:
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.exit_date = exit_date or date.today().isoformat()
                trade.status = 'closed'
                self._save()
                logger.info(f"Closed trade {trade_id}: P&L = {trade.pnl:.2f} ({trade.pnl_pct:.1f}%)")
                return trade
        return None

    def close_trade_by_symbol(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = 'manual',
        exit_date: Optional[str] = None
    ) -> Optional[Trade]:
        """Close the most recent open trade for a symbol"""
        symbol = symbol.upper()
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.is_open:
                return self.close_trade(trade.trade_id, exit_price, exit_reason, exit_date)
        return None

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        return [t for t in self.trades if t.is_open]

    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades"""
        return [t for t in self.trades if not t.is_open]

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a specific trade by ID"""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get all trades for a symbol"""
        return [t for t in self.trades if t.symbol == symbol.upper()]

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade from history"""
        for i, trade in enumerate(self.trades):
            if trade.trade_id == trade_id:
                del self.trades[i]
                self._save()
                logger.info(f"Deleted trade {trade_id}")
                return True
        return False

    # ==================== PERFORMANCE METRICS ====================

    def get_total_pnl(self) -> float:
        """Get total realized P&L from closed trades"""
        return sum(t.pnl for t in self.get_closed_trades())

    def get_win_rate(self) -> Tuple[float, int, int]:
        """
        Calculate win rate

        Returns:
            (win_rate_pct, wins, losses)
        """
        closed = self.get_closed_trades()
        if not closed:
            return 0.0, 0, 0

        wins = sum(1 for t in closed if t.is_winner)
        losses = len(closed) - wins
        win_rate = (wins / len(closed)) * 100
        return win_rate, wins, losses

    def get_average_pnl(self) -> Tuple[float, float, float]:
        """
        Calculate average P&L

        Returns:
            (avg_pnl, avg_win, avg_loss)
        """
        closed = self.get_closed_trades()
        if not closed:
            return 0.0, 0.0, 0.0

        wins = [t.pnl for t in closed if t.is_winner]
        losses = [t.pnl for t in closed if not t.is_winner]

        avg_pnl = sum(t.pnl for t in closed) / len(closed)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return avg_pnl, avg_win, avg_loss

    def get_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profits / gross losses)

        Returns:
            Profit factor (> 1 is profitable)
        """
        closed = self.get_closed_trades()
        if not closed:
            return 0.0

        gross_profit = sum(t.pnl for t in closed if t.is_winner)
        gross_loss = abs(sum(t.pnl for t in closed if not t.is_winner))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Generate equity curve DataFrame

        Returns:
            DataFrame with date, equity, drawdown columns
        """
        closed = sorted(self.get_closed_trades(), key=lambda t: t.exit_date or '')

        if not closed:
            return pd.DataFrame(columns=['date', 'equity', 'pnl', 'cumulative_pnl', 'drawdown_pct'])

        data = []
        cumulative_pnl = 0
        peak_equity = self.initial_capital

        for trade in closed:
            cumulative_pnl += trade.pnl
            equity = self.initial_capital + cumulative_pnl
            peak_equity = max(peak_equity, equity)
            drawdown_pct = ((peak_equity - equity) / peak_equity) * 100 if peak_equity > 0 else 0

            data.append({
                'date': trade.exit_date,
                'symbol': trade.symbol,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'equity': equity,
                'cumulative_pnl': cumulative_pnl,
                'drawdown_pct': drawdown_pct
            })

        return pd.DataFrame(data)

    def get_max_drawdown(self) -> Tuple[float, float]:
        """
        Calculate maximum drawdown

        Returns:
            (max_drawdown_amount, max_drawdown_pct)
        """
        equity_df = self.get_equity_curve()
        if equity_df.empty:
            return 0.0, 0.0

        max_dd_pct = equity_df['drawdown_pct'].max()

        # Calculate max drawdown in currency
        equities = [self.initial_capital] + equity_df['equity'].tolist()
        peak = equities[0]
        max_dd_amount = 0

        for eq in equities:
            peak = max(peak, eq)
            dd = peak - eq
            max_dd_amount = max(max_dd_amount, dd)

        return max_dd_amount, max_dd_pct

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (annualized)

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        closed = self.get_closed_trades()
        if len(closed) < 2:
            return 0.0

        returns = [t.pnl_pct / 100 for t in closed]  # Convert to decimal

        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize (assuming ~52 trades per year)
        trades_per_year = min(52, len(closed))
        annual_return = avg_return * trades_per_year
        annual_std = std_return * np.sqrt(trades_per_year)

        sharpe = (annual_return - risk_free_rate) / annual_std
        return sharpe

    def get_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sortino ratio
        """
        closed = self.get_closed_trades()
        if len(closed) < 2:
            return 0.0

        returns = [t.pnl_pct / 100 for t in closed]

        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float('inf') if avg_return > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        trades_per_year = min(52, len(closed))
        annual_return = avg_return * trades_per_year
        annual_downside_std = downside_std * np.sqrt(trades_per_year)

        sortino = (annual_return - risk_free_rate) / annual_downside_std
        return sortino

    def get_consecutive_stats(self) -> Dict[str, int]:
        """
        Get consecutive wins/losses stats

        Returns:
            Dict with max_consecutive_wins, max_consecutive_losses, current_streak
        """
        closed = sorted(self.get_closed_trades(), key=lambda t: t.exit_date or '')

        if not closed:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0, 'current_streak': 0}

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in closed:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        # Current streak (positive = wins, negative = losses)
        current_streak = current_wins if current_wins > 0 else -current_losses

        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak': current_streak
        }

    def get_performance_summary(self) -> Dict:
        """
        Get complete performance summary

        Returns:
            Dict with all performance metrics
        """
        win_rate, wins, losses = self.get_win_rate()
        avg_pnl, avg_win, avg_loss = self.get_average_pnl()
        max_dd_amount, max_dd_pct = self.get_max_drawdown()
        consecutive = self.get_consecutive_stats()

        closed = self.get_closed_trades()
        open_trades = self.get_open_trades()

        # Calculate current equity
        total_pnl = self.get_total_pnl()
        current_equity = self.initial_capital + total_pnl

        # Return on capital
        roi = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'current_equity': current_equity,
            'total_pnl': total_pnl,
            'roi_pct': roi,
            'total_trades': len(self.trades),
            'open_trades': len(open_trades),
            'closed_trades': len(closed),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': self.get_profit_factor(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'sortino_ratio': self.get_sortino_ratio(),
            'max_drawdown': max_dd_amount,
            'max_drawdown_pct': max_dd_pct,
            'max_consecutive_wins': consecutive['max_consecutive_wins'],
            'max_consecutive_losses': consecutive['max_consecutive_losses'],
            'current_streak': consecutive['current_streak']
        }

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Get monthly returns breakdown

        Returns:
            DataFrame with year, month, pnl, trades columns
        """
        closed = self.get_closed_trades()
        if not closed:
            return pd.DataFrame(columns=['year', 'month', 'pnl', 'trades', 'win_rate'])

        # Group by month
        monthly_data = {}
        for trade in closed:
            if trade.exit_date:
                dt = datetime.fromisoformat(trade.exit_date)
                key = (dt.year, dt.month)
                if key not in monthly_data:
                    monthly_data[key] = {'pnl': 0, 'trades': 0, 'wins': 0}
                monthly_data[key]['pnl'] += trade.pnl
                monthly_data[key]['trades'] += 1
                if trade.is_winner:
                    monthly_data[key]['wins'] += 1

        data = []
        for (year, month), stats in sorted(monthly_data.items()):
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            data.append({
                'year': year,
                'month': month,
                'pnl': stats['pnl'],
                'trades': stats['trades'],
                'win_rate': win_rate
            })

        return pd.DataFrame(data)

    def get_symbol_performance(self) -> pd.DataFrame:
        """
        Get performance breakdown by symbol

        Returns:
            DataFrame with symbol, trades, pnl, win_rate columns
        """
        closed = self.get_closed_trades()
        if not closed:
            return pd.DataFrame(columns=['symbol', 'trades', 'pnl', 'win_rate', 'avg_pnl'])

        symbol_data = {}
        for trade in closed:
            if trade.symbol not in symbol_data:
                symbol_data[trade.symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
            symbol_data[trade.symbol]['trades'] += 1
            symbol_data[trade.symbol]['pnl'] += trade.pnl
            if trade.is_winner:
                symbol_data[trade.symbol]['wins'] += 1

        data = []
        for symbol, stats in sorted(symbol_data.items(), key=lambda x: x[1]['pnl'], reverse=True):
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            data.append({
                'symbol': symbol,
                'trades': stats['trades'],
                'pnl': stats['pnl'],
                'win_rate': win_rate,
                'avg_pnl': avg_pnl
            })

        return pd.DataFrame(data)

    def set_initial_capital(self, capital: float):
        """Update initial capital"""
        self.initial_capital = capital
        self._save()

    def clear_all_trades(self):
        """Clear all trade history (use with caution!)"""
        self.trades = []
        self._save()
        logger.warning("Cleared all trade history")


# Singleton instance
trade_tracker = TradeTracker()
