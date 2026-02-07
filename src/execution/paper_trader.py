"""
Paper Trading Engine - Simulates trades without real money

Manages:
- Opening positions on BUY signals
- Tracking P&L in real-time
- Stop loss / Take profit execution
- Portfolio state persistence
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf
from .dynamic_stops import calculate_dynamic_stops, calculate_position_size_from_risk

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = Path("data/portfolio.json")
TRADES_HISTORY_PATH = Path("data/trades_history.json")


@dataclass
class Position:
    """A single position"""
    symbol: str
    entry_price: float
    quantity: int
    entry_date: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    score_at_entry: float = 0.0
    decision_type: str = "BUY"
    
    # Analysis data (4 piliers)
    pillar_technical: Optional[float] = None
    pillar_fundamental: Optional[float] = None
    pillar_sentiment: Optional[float] = None
    pillar_news: Optional[float] = None
    reasoning: Optional[str] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    # Computed fields
    current_price: float = 0.0
    pnl_amount: float = 0.0
    pnl_percent: float = 0.0
    
    def update_price(self, price: float):
        """Update current price and P&L"""
        self.current_price = price
        self.pnl_amount = (price - self.entry_price) * self.quantity
        self.pnl_percent = ((price - self.entry_price) / self.entry_price) * 100
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class TradeRecord:
    """Record of a completed trade"""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_date: str
    exit_date: str
    pnl_amount: float
    pnl_percent: float
    exit_reason: str  # stop_loss, take_profit, manual
    score_at_entry: float = 0.0


class PaperTrader:
    """
    Paper trading engine for simulating trades.
    """
    
    def __init__(self, initial_capital: float = 15000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        
        # Risk management
        self.max_position_pct = 0.10  # Max 10% per position
        self.default_stop_loss_pct = 0.05  # 5% stop loss
        self.default_take_profit_pct = 0.15  # 15% take profit
        
        self._load_state()
        logger.info(f"PaperTrader initialized: cash={self.cash:.2f}, positions={len(self.positions)}")
    
    def _load_state(self):
        """Load portfolio state from disk"""
        if PORTFOLIO_PATH.exists():
            try:
                with open(PORTFOLIO_PATH, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', data.get('total_capital', self.initial_capital))
                    self.initial_capital = data.get('total_capital', data.get('initial_capital', self.initial_capital))
                    
                    # Load positions (handle both list and dict formats)
                    positions_data = data.get('positions', [])
                    if isinstance(positions_data, dict):
                        # New dict format
                        for symbol, pos_data in positions_data.items():
                            self.positions[symbol] = Position(**pos_data)
                    else:
                        # Old list format (compatible with PortfolioTracker)
                        for pos_data in positions_data:
                            pos = Position(
                                symbol=pos_data['symbol'],
                                entry_price=pos_data['entry_price'],
                                quantity=pos_data.get('shares', pos_data.get('quantity', 0)),
                                entry_date=pos_data['entry_date'],
                                stop_loss=pos_data.get('stop_loss'),
                                take_profit=pos_data.get('take_profit'),
                                score_at_entry=pos_data.get('score_at_entry', 0),
                                pillar_technical=pos_data.get('pillar_technical'),
                                pillar_fundamental=pos_data.get('pillar_fundamental'),
                                pillar_sentiment=pos_data.get('pillar_sentiment'),
                                pillar_news=pos_data.get('pillar_news'),
                                reasoning=pos_data.get('reasoning'),
                                company_name=pos_data.get('company_name'),
                                sector=pos_data.get('sector'),
                                industry=pos_data.get('industry'),
                                current_price=pos_data.get('current_price', pos_data['entry_price']),
                                pnl_amount=pos_data.get('pnl_amount', 0),
                                pnl_percent=pos_data.get('pnl_percent', 0)
                            )
                            self.positions[pos.symbol] = pos
                logger.info(f"Loaded portfolio: cash={self.cash:.2f}, {len(self.positions)} positions")
            except Exception as e:
                logger.warning(f"Could not load portfolio: {e}")
        
        if TRADES_HISTORY_PATH.exists():
            try:
                with open(TRADES_HISTORY_PATH, 'r') as f:
                    data = json.load(f)
                    self.trade_history = [TradeRecord(**t) for t in data.get('trades', [])]
            except Exception as e:
                logger.warning(f"Could not load trade history: {e}")
    
    def save_state(self):
        """Save portfolio state to disk"""
        try:
            # Save portfolio (compatible with PortfolioTracker format)
            positions_list = []
            for p in self.positions.values():
                positions_list.append({
                    'symbol': p.symbol,
                    'shares': p.quantity,
                    'entry_price': p.entry_price,
                    'entry_date': p.entry_date,
                    'stop_loss': p.stop_loss or 0,
                    'position_value': p.quantity * p.entry_price,
                    # Extra fields for paper trading
                    'take_profit': p.take_profit,
                    'score_at_entry': p.score_at_entry,
                    'current_price': p.current_price,
                    'pnl_amount': p.pnl_amount,
                    'pnl_percent': p.pnl_percent,
                    # Analysis data (4 pillars)
                    'pillar_technical': p.pillar_technical,
                    'pillar_fundamental': p.pillar_fundamental,
                    'pillar_sentiment': p.pillar_sentiment,
                    'pillar_news': p.pillar_news,
                    'reasoning': p.reasoning,
                    'company_name': p.company_name,
                    'sector': p.sector,
                    'industry': p.industry
                })
            
            portfolio_data = {
                'total_capital': self.initial_capital,
                'cash': self.cash,
                'total_value': self.get_total_value(),
                'positions': positions_list,
                'updated': datetime.now().isoformat()
            }
            with open(PORTFOLIO_PATH, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            
            # Save trade history
            history_data = {
                'trades': [asdict(t) for t in self.trade_history],
                'total_trades': len(self.trade_history),
                'updated': datetime.now().isoformat()
            }
            with open(TRADES_HISTORY_PATH, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logger.info(f"Saved portfolio: {len(self.positions)} positions, {len(self.trade_history)} trades")
        except Exception as e:
            logger.error(f"Could not save portfolio: {e}")
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_position_value(self, symbol: str) -> float:
        """Get value of a specific position"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return pos.current_price * pos.quantity
        return 0.0
    
    def can_open_position(self, symbol: str, price: float) -> tuple[bool, str]:
        """Check if we can open a new position"""
        # Already have position?
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        
        # Calculate position size
        max_position_value = self.get_total_value() * self.max_position_pct
        
        if self.cash < max_position_value:
            return False, f"Not enough cash ({self.cash:.2f} < {max_position_value:.2f})"
        
        return True, "OK"
    
    def open_position(
        self, 
        symbol: str, 
        price: float, 
        score: float = 0.0,
        decision_type: str = "BUY",
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        pillar_technical: Optional[float] = None,
        pillar_fundamental: Optional[float] = None,
        pillar_sentiment: Optional[float] = None,
        pillar_news: Optional[float] = None,
        reasoning: Optional[str] = None,
        company_name: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None
    ) -> Optional[Position]:
        """
        Open a new paper position with ATR-based dynamic stops.
        
        Returns:
            Position if successful, None otherwise
        """
        can_open, reason = self.can_open_position(symbol, price)
        if not can_open:
            logger.warning(f"Cannot open position in {symbol}: {reason}")
            return None
        
        # Calculate dynamic stop loss and take profit based on ATR
        try:
            stop_loss, take_profit = calculate_dynamic_stops(
                symbol=symbol,
                entry_price=price,
                confidence_score=score
            )
            logger.info(f"{symbol}: Dynamic SL={stop_loss:.2f}, TP={take_profit:.2f} (ATR-based)")
        except Exception as e:
            logger.warning(f"Could not calculate dynamic stops for {symbol}: {e}, using defaults")
            sl_pct = stop_loss_pct or self.default_stop_loss_pct
            tp_pct = take_profit_pct or self.default_take_profit_pct
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        
        # Calculate position size based on confidence/score
        # Higher score = bigger position
        if score >= 90:
            risk_pct = 1.5  # High conviction
            max_pos_pct = 0.10  # 10%
        elif score >= 85:
            risk_pct = 1.2
            max_pos_pct = 0.08  # 8%
        elif score >= 80:
            risk_pct = 1.0
            max_pos_pct = 0.06  # 6%
        elif score >= 75:
            risk_pct = 0.7
            max_pos_pct = 0.05  # 5%
        else:
            risk_pct = 0.5
            max_pos_pct = 0.03  # 3%
        
        quantity = calculate_position_size_from_risk(
            capital=self.get_total_value(),
            entry_price=price,
            stop_loss=stop_loss,
            risk_per_trade_pct=risk_pct,
            max_position_pct=max_pos_pct * 100
        )
        logger.info(f"{symbol}: Score={score:.1f} -> Risk={risk_pct}%, MaxPos={max_pos_pct*100}%")
        
        if quantity < 1:
            logger.warning(f"Cannot afford even 1 share of {symbol} at {price}")
            return None
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            entry_date=datetime.now().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            score_at_entry=score,
            decision_type=decision_type,
            pillar_technical=pillar_technical,
            pillar_fundamental=pillar_fundamental,
            pillar_sentiment=pillar_sentiment,
            pillar_news=pillar_news,
            reasoning=reasoning,
            company_name=company_name,
            sector=sector,
            industry=industry,
            current_price=price
        )
        
        # Update cash
        cost = price * quantity
        self.cash -= cost
        
        # Store position
        self.positions[symbol] = position
        
        logger.info(
            f"📈 OPENED: {symbol} x{quantity} @ {price:.2f} "
            f"(cost: {cost:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f})"
        )
        
        self.save_state()
        return position
    
    def close_position(self, symbol: str, price: float, reason: str = "manual") -> Optional[TradeRecord]:
        """
        Close an existing position.
        
        Returns:
            TradeRecord if successful, None otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        proceeds = price * position.quantity
        pnl_amount = (price - position.entry_price) * position.quantity
        pnl_percent = ((price - position.entry_price) / position.entry_price) * 100
        
        # Create trade record
        trade = TradeRecord(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            entry_date=position.entry_date,
            exit_date=datetime.now().isoformat(),
            pnl_amount=pnl_amount,
            pnl_percent=pnl_percent,
            exit_reason=reason,
            score_at_entry=position.score_at_entry
        )
        
        # Update cash
        self.cash += proceeds
        
        # Remove position
        del self.positions[symbol]
        
        # Add to history
        self.trade_history.append(trade)
        
        emoji = "✅" if pnl_amount > 0 else "❌"
        logger.info(
            f"{emoji} CLOSED: {symbol} x{position.quantity} @ {price:.2f} "
            f"(P&L: {pnl_amount:+.2f} / {pnl_percent:+.1f}%, reason: {reason})"
        )
        
        self.save_state()
        return trade
    
    def update_prices(self):
        """Update all position prices and check stop loss/take profit"""
        if not self.positions:
            return
        
        symbols = list(self.positions.keys())
        
        try:
            # Batch fetch prices
            tickers = yf.Tickers(" ".join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if ticker:
                        info = ticker.fast_info
                        price = info.get('lastPrice') or info.get('regularMarketPrice', 0)
                        
                        if price and price > 0:
                            position = self.positions[symbol]
                            position.update_price(price)
                            
                            # Check stop loss
                            if position.stop_loss and price <= position.stop_loss:
                                logger.info(f"🛑 Stop loss triggered for {symbol} at {price:.2f}")
                                self.close_position(symbol, price, "stop_loss")
                            
                            # Check take profit
                            elif position.take_profit and price >= position.take_profit:
                                logger.info(f"🎯 Take profit triggered for {symbol} at {price:.2f}")
                                self.close_position(symbol, price, "take_profit")
                                
                except Exception as e:
                    logger.warning(f"Could not update price for {symbol}: {e}")
            
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
    
    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary for display"""
        total_value = self.get_total_value()
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        wins = sum(1 for t in self.trade_history if t.pnl_amount > 0)
        losses = sum(1 for t in self.trade_history if t.pnl_amount <= 0)
        win_rate = (wins / len(self.trade_history) * 100) if self.trade_history else 0
        
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions_count': len(self.positions),
            'positions': [p.to_dict() for p in self.positions.values()],
            'total_trades': len(self.trade_history),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate
        }


# Singleton
_paper_trader: Optional[PaperTrader] = None

def get_paper_trader() -> PaperTrader:
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
    return _paper_trader
