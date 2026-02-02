"""
Enhanced Telegram Trading Notifier
Sends formatted trading signals, portfolio updates, and bot status.
"""
import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal data"""
    symbol: str
    signal_type: str  # SELECTION, BUY, SELL
    price: float
    quantity: int = 0
    score: float = 0
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # For sells
    entry_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class TelegramTraderNotifier:
    """
    Enhanced Telegram notifier for trading signals.
    
    Message types:
    - 🔍 Selection: New stock detected by scanner
    - 🟢 Buy: Position opened
    - 🔴 Sell: Position closed with P&L
    - 📊 Daily Summary: Portfolio performance
    - ⚙️ Bot Status: Start/Stop/Error
    """
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured")
            return False
            
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                }
            ) as response:
                if response.status == 200:
                    logger.info(f"Telegram message sent")
                    return True
                else:
                    logger.error(f"Telegram error {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    # ==================== Signal Notifications ====================
    
    async def notify_selection(self, signal: TradeSignal):
        """Notify about a new stock selection"""
        msg = f"""🔍 <b>SÉLECTION</b>

<b>{signal.symbol}</b> - Score: {signal.score:.0f}/100

💡 <i>{signal.reason[:200]}</i>

💰 Prix: ${signal.price:.2f}
"""
        if signal.stop_loss:
            msg += f"🛡️ Stop Loss: ${signal.stop_loss:.2f}\n"
        if signal.take_profit:
            msg += f"🎯 Take Profit: ${signal.take_profit:.2f}\n"
            
        msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        await self.send_message(msg)
    
    async def notify_buy(self, signal: TradeSignal):
        """Notify about a buy execution"""
        total_value = signal.price * signal.quantity
        
        msg = f"""🟢 <b>ACHAT</b>

<b>{signal.symbol}</b>
📊 Score: {signal.score:.0f}/100

💵 Prix: ${signal.price:.2f}
📦 Quantité: {signal.quantity}
💰 Total: ${total_value:.2f}

🛡️ Stop Loss: ${signal.stop_loss:.2f if signal.stop_loss else 'N/A'}
🎯 Take Profit: ${signal.take_profit:.2f if signal.take_profit else 'N/A'}

💡 <i>{signal.reason[:150]}</i>

⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(msg)
    
    async def notify_sell(self, signal: TradeSignal):
        """Notify about a sell execution with P&L"""
        total_value = signal.price * signal.quantity
        
        # P&L emoji
        if signal.pnl and signal.pnl > 0:
            pnl_emoji = "📈"
            pnl_color = "+"
        else:
            pnl_emoji = "📉"
            pnl_color = ""
        
        msg = f"""🔴 <b>VENTE</b>

<b>{signal.symbol}</b>

💵 Prix vente: ${signal.price:.2f}
📦 Quantité: {signal.quantity}
💰 Total: ${total_value:.2f}
"""
        
        if signal.entry_price:
            msg += f"\n📥 Prix achat: ${signal.entry_price:.2f}"
        
        if signal.pnl is not None:
            msg += f"\n\n{pnl_emoji} <b>P&L: {pnl_color}${signal.pnl:.2f}</b>"
        if signal.pnl_pct is not None:
            msg += f" ({pnl_color}{signal.pnl_pct:.1f}%)"
            
        msg += f"\n\n💡 <i>{signal.reason[:150]}</i>"
        msg += f"\n\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        await self.send_message(msg)
    
    # ==================== Portfolio Notifications ====================
    
    async def notify_daily_summary(
        self,
        total_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        positions: List[Dict],
        trades_today: int
    ):
        """Send daily portfolio summary"""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        
        msg = f"""📊 <b>RÉSUMÉ JOURNALIER</b>

💰 Portefeuille: ${total_value:,.2f}
{pnl_emoji} P&L Jour: {pnl_sign}${daily_pnl:.2f} ({pnl_sign}{daily_pnl_pct:.1f}%)

📋 Positions ({len(positions)}):
"""
        
        for pos in positions[:5]:  # Max 5 positions
            pos_emoji = "🟢" if pos.get('pnl', 0) >= 0 else "🔴"
            msg += f"  {pos_emoji} {pos['symbol']}: {pos.get('pnl_pct', 0):+.1f}%\n"
        
        if len(positions) > 5:
            msg += f"  ... et {len(positions) - 5} autres\n"
            
        msg += f"\n🔄 Trades aujourd'hui: {trades_today}"
        msg += f"\n⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        
        await self.send_message(msg)
    
    # ==================== Bot Status Notifications ====================
    
    async def notify_bot_started(self, mode: str = "live"):
        """Notify bot started"""
        msg = f"""⚙️ <b>BOT DÉMARRÉ</b>

🚀 Mode: {mode.upper()}
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

Le bot de trading est maintenant actif."""
        
        await self.send_message(msg)
    
    async def notify_bot_stopped(self, reason: str = "Manual"):
        """Notify bot stopped"""
        msg = f"""⚙️ <b>BOT ARRÊTÉ</b>

🛑 Raison: {reason}
⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        await self.send_message(msg)
    
    async def notify_error(self, error: str):
        """Notify about an error"""
        msg = f"""🚨 <b>ERREUR</b>

❌ {error[:500]}

⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(msg)
    
    # ==================== Analysis Notifications ====================
    
    async def notify_analysis(
        self,
        symbol: str,
        score: float,
        technical: float,
        fundamental: float,
        sentiment: float,
        recommendation: str,
        key_factors: List[str]
    ):
        """Notify about a completed analysis"""
        # Score color
        if score >= 70:
            score_emoji = "🟢"
        elif score >= 50:
            score_emoji = "🟡"
        else:
            score_emoji = "🔴"
        
        msg = f"""📈 <b>ANALYSE {symbol}</b>

{score_emoji} Score Global: {score:.0f}/100
📊 Technique: {technical:.0f}
💼 Fondamental: {fundamental:.0f}
🎭 Sentiment: {sentiment:.0f}

📌 Recommandation: <b>{recommendation}</b>

🔑 Facteurs clés:
"""
        for factor in key_factors[:3]:
            msg += f"• {factor}\n"
            
        msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        await self.send_message(msg)


# ==================== Singleton ====================

_notifier: Optional[TelegramTraderNotifier] = None

def get_telegram_notifier() -> TelegramTraderNotifier:
    """Get singleton instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramTraderNotifier()
    return _notifier


# ==================== Quick Send Functions ====================

async def send_selection(symbol: str, price: float, score: float, reason: str):
    """Quick function to send selection signal"""
    notifier = get_telegram_notifier()
    await notifier.notify_selection(TradeSignal(
        symbol=symbol,
        signal_type="SELECTION",
        price=price,
        score=score,
        reason=reason
    ))

async def send_buy(symbol: str, price: float, quantity: int, score: float, reason: str, stop_loss: float = None, take_profit: float = None):
    """Quick function to send buy signal"""
    notifier = get_telegram_notifier()
    await notifier.notify_buy(TradeSignal(
        symbol=symbol,
        signal_type="BUY",
        price=price,
        quantity=quantity,
        score=score,
        reason=reason,
        stop_loss=stop_loss,
        take_profit=take_profit
    ))

async def send_sell(symbol: str, price: float, quantity: int, entry_price: float, pnl: float, pnl_pct: float, reason: str):
    """Quick function to send sell signal"""
    notifier = get_telegram_notifier()
    await notifier.notify_sell(TradeSignal(
        symbol=symbol,
        signal_type="SELL",
        price=price,
        quantity=quantity,
        entry_price=entry_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        reason=reason
    ))


# ==================== Test ====================

async def test_notifications():
    """Test all notification types"""
    notifier = get_telegram_notifier()
    
    # Test selection
    await notifier.notify_selection(TradeSignal(
        symbol="AAPL",
        signal_type="SELECTION",
        price=185.50,
        score=78,
        reason="Strong technical setup with RSI breakout and positive momentum",
        stop_loss=180.00,
        take_profit=200.00
    ))
    
    await asyncio.sleep(1)
    
    # Test buy
    await notifier.notify_buy(TradeSignal(
        symbol="AAPL",
        signal_type="BUY",
        price=185.50,
        quantity=10,
        score=78,
        reason="Confirmed breakout with volume",
        stop_loss=180.00,
        take_profit=200.00
    ))
    
    await asyncio.sleep(1)
    
    # Test sell
    await notifier.notify_sell(TradeSignal(
        symbol="AAPL",
        signal_type="SELL",
        price=195.00,
        quantity=10,
        entry_price=185.50,
        pnl=95.00,
        pnl_pct=5.12,
        reason="Take profit reached"
    ))
    
    await notifier.close()
    print("✅ All notifications sent!")


if __name__ == "__main__":
    asyncio.run(test_notifications())
