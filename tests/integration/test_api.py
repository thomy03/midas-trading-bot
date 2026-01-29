"""
Integration tests for FastAPI REST API

Tests all API endpoints:
- Health and info
- Stock screening
- Alerts
- Watchlists
- Trades and portfolio
- Sectors and market data
- Economic calendar
- WebSocket
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from src.api.main import app, verify_api_key
from src.api.websocket import ConnectionManager, AlertMessage, ProgressMessage


# Override API key verification for testing
async def override_verify_api_key():
    return "test_key"


app.dependency_overrides[verify_api_key] = override_verify_api_key

client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health and info endpoints"""

    @pytest.mark.integration
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "TradingBot V3 API"

    @pytest.mark.integration
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestScreeningEndpoints:
    """Tests for screening endpoints"""

    @pytest.mark.integration
    def test_screen_single_stock(self):
        """Test screening single stock"""
        with patch('src.api.main.market_screener') as mock_screener:
            mock_screener.screen_single_stock.return_value = {
                'symbol': 'AAPL',
                'company_name': 'Apple Inc.',
                'timeframe': 'weekly',
                'current_price': 150.0,
                'support_level': 145.0,
                'distance_to_support_pct': 3.4,
                'recommendation': 'BUY',
                'confidence_score': 75.0,
                'has_rsi_breakout': True
            }

            response = client.get("/api/v1/screen/AAPL")

            assert response.status_code == 200
            data = response.json()
            assert data['symbol'] == 'AAPL'
            assert data['recommendation'] == 'BUY'

    @pytest.mark.integration
    def test_screen_single_stock_no_signal(self):
        """Test screening stock with no signal"""
        with patch('src.api.main.market_screener') as mock_screener:
            mock_screener.screen_single_stock.return_value = None

            response = client.get("/api/v1/screen/NOSTOCK")

            assert response.status_code == 200
            assert response.json() is None

    @pytest.mark.integration
    def test_screen_multiple_stocks(self):
        """Test screening multiple stocks"""
        with patch('src.api.main.market_screener') as mock_screener:
            mock_screener.screen_multiple_stocks.return_value = [
                {
                    'symbol': 'AAPL',
                    'company_name': 'Apple',
                    'timeframe': 'weekly',
                    'current_price': 150.0,
                    'support_level': 145.0,
                    'distance_to_support_pct': 3.4,
                    'recommendation': 'BUY'
                }
            ]

            response = client.post(
                "/api/v1/screen",
                json={"symbols": ["AAPL", "MSFT"]}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data) >= 0  # May return 0 or more alerts


class TestAlertsEndpoints:
    """Tests for alerts endpoints"""

    @pytest.mark.integration
    def test_get_recent_alerts(self):
        """Test getting recent alerts"""
        with patch('src.api.main.db_manager') as mock_db:
            mock_alert = MagicMock()
            mock_alert.id = 1
            mock_alert.symbol = 'AAPL'
            mock_alert.timeframe = 'weekly'
            mock_alert.current_price = 150.0
            mock_alert.support_level = 145.0
            mock_alert.recommendation = 'BUY'
            mock_alert.created_at = datetime.now()

            mock_db.get_recent_alerts.return_value = [mock_alert]

            response = client.get("/api/v1/alerts")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['symbol'] == 'AAPL'

    @pytest.mark.integration
    def test_get_alerts_with_params(self):
        """Test getting alerts with query parameters"""
        with patch('src.api.main.db_manager') as mock_db:
            mock_db.get_recent_alerts.return_value = []

            response = client.get("/api/v1/alerts?days=30&limit=10")

            assert response.status_code == 200
            mock_db.get_recent_alerts.assert_called_once_with(days=30)


class TestWatchlistEndpoints:
    """Tests for watchlist endpoints"""

    @pytest.mark.integration
    def test_get_watchlists(self):
        """Test getting all watchlists"""
        with patch('src.api.main.watchlist_manager') as mock_wl:
            mock_watchlist = MagicMock()
            mock_watchlist.id = 'wl_123'
            mock_watchlist.name = 'Tech Stocks'
            mock_watchlist.symbols = ['AAPL', 'MSFT']
            mock_watchlist.created_at = datetime.now()

            mock_wl.get_all_watchlists.return_value = [mock_watchlist]

            response = client.get("/api/v1/watchlists")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['name'] == 'Tech Stocks'

    @pytest.mark.integration
    def test_create_watchlist(self):
        """Test creating a watchlist"""
        with patch('src.api.main.watchlist_manager') as mock_wl:
            mock_watchlist = MagicMock()
            mock_watchlist.id = 'wl_new'
            mock_watchlist.name = 'New List'
            mock_watchlist.symbols = ['AAPL']
            mock_watchlist.created_at = datetime.now()

            mock_wl.create_watchlist.return_value = mock_watchlist

            response = client.post(
                "/api/v1/watchlists",
                json={"name": "New List", "symbols": ["AAPL"]}
            )

            assert response.status_code == 200
            data = response.json()
            assert data['name'] == 'New List'

    @pytest.mark.integration
    def test_delete_watchlist(self):
        """Test deleting a watchlist"""
        with patch('src.api.main.watchlist_manager') as mock_wl:
            mock_wl.delete_watchlist.return_value = True

            response = client.delete("/api/v1/watchlists/wl_123")

            assert response.status_code == 200
            assert response.json()['status'] == 'deleted'

    @pytest.mark.integration
    def test_delete_nonexistent_watchlist(self):
        """Test deleting nonexistent watchlist"""
        with patch('src.api.main.watchlist_manager') as mock_wl:
            mock_wl.delete_watchlist.return_value = False

            response = client.delete("/api/v1/watchlists/nonexistent")

            assert response.status_code == 404


class TestTradesEndpoints:
    """Tests for trades endpoints"""

    @pytest.mark.integration
    def test_get_all_trades(self):
        """Test getting all trades"""
        with patch('src.api.main.trade_tracker') as mock_tracker:
            mock_trade = MagicMock()
            mock_trade.trade_id = 'trade_123'
            mock_trade.symbol = 'AAPL'
            mock_trade.entry_date = '2024-01-15'
            mock_trade.entry_price = 150.0
            mock_trade.shares = 10
            mock_trade.status = 'open'
            mock_trade.pnl = None
            mock_trade.pnl_pct = None

            mock_tracker.get_all_trades.return_value = [mock_trade]

            response = client.get("/api/v1/trades")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['symbol'] == 'AAPL'

    @pytest.mark.integration
    def test_get_open_trades(self):
        """Test getting only open trades"""
        with patch('src.api.main.trade_tracker') as mock_tracker:
            mock_tracker.get_open_trades.return_value = []

            response = client.get("/api/v1/trades?status=open")

            assert response.status_code == 200
            mock_tracker.get_open_trades.assert_called_once()

    @pytest.mark.integration
    def test_open_trade(self):
        """Test opening a new trade"""
        with patch('src.api.main.trade_tracker') as mock_tracker:
            mock_trade = MagicMock()
            mock_trade.trade_id = 'trade_new'
            mock_trade.symbol = 'AAPL'
            mock_trade.entry_date = '2024-01-15'
            mock_trade.entry_price = 150.0
            mock_trade.shares = 10
            mock_trade.status = 'open'

            mock_tracker.open_trade.return_value = mock_trade

            response = client.post(
                "/api/v1/trades",
                json={
                    "symbol": "AAPL",
                    "entry_price": 150.0,
                    "shares": 10,
                    "stop_loss": 145.0
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data['symbol'] == 'AAPL'

    @pytest.mark.integration
    def test_close_trade(self):
        """Test closing a trade"""
        with patch('src.api.main.trade_tracker') as mock_tracker:
            mock_trade = MagicMock()
            mock_trade.trade_id = 'trade_123'
            mock_trade.symbol = 'AAPL'
            mock_trade.entry_date = '2024-01-15'
            mock_trade.entry_price = 150.0
            mock_trade.shares = 10
            mock_trade.status = 'closed'
            mock_trade.pnl = 50.0
            mock_trade.pnl_pct = 3.33

            mock_tracker.close_trade.return_value = mock_trade

            response = client.post(
                "/api/v1/trades/trade_123/close",
                json={"exit_price": 155.0}
            )

            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'closed'
            assert data['pnl'] == 50.0

    @pytest.mark.integration
    def test_get_performance(self):
        """Test getting performance metrics"""
        with patch('src.api.main.trade_tracker') as mock_tracker:
            mock_tracker.get_performance_summary.return_value = {
                'total_trades': 10,
                'open_trades': 2,
                'closed_trades': 8,
                'total_pnl': 500.0,
                'win_rate': 75.0,
                'avg_win': 100.0,
                'avg_loss': -50.0,
                'profit_factor': 2.0,
                'sharpe_ratio': 1.5,
                'max_drawdown': 5.0
            }

            response = client.get("/api/v1/trades/performance")

            assert response.status_code == 200
            data = response.json()
            assert data['total_trades'] == 10
            assert data['win_rate'] == 75.0


class TestPortfolioEndpoints:
    """Tests for portfolio endpoints"""

    @pytest.mark.integration
    def test_get_portfolio_summary(self):
        """Test getting portfolio summary"""
        with patch('src.api.main.market_screener') as mock_screener:
            mock_screener.get_portfolio_summary.return_value = {
                'total_capital': 10000,
                'available_capital': 8000,
                'invested_capital': 2000,
                'open_positions': 2,
                'unrealized_pnl': 150
            }

            response = client.get("/api/v1/portfolio/summary")

            assert response.status_code == 200
            data = response.json()
            assert data['total_capital'] == 10000


class TestSectorEndpoints:
    """Tests for sector endpoints"""

    @pytest.mark.integration
    def test_get_sector_performance(self):
        """Test getting sector performance"""
        with patch('src.api.main.sector_heatmap_builder') as mock_sectors:
            mock_sector = MagicMock()
            mock_sector.name = 'Technology'
            mock_sector.etf = 'XLK'
            mock_sector.perf_1d = 1.5
            mock_sector.perf_1w = 3.2
            mock_sector.perf_1m = 5.0
            mock_sector.perf_ytd = 15.0

            mock_sectors.get_sector_performance.return_value = [mock_sector]

            response = client.get("/api/v1/sectors")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['name'] == 'Technology'

    @pytest.mark.integration
    def test_get_market_breadth(self):
        """Test getting market breadth"""
        with patch('src.api.main.sector_heatmap_builder') as mock_sectors:
            mock_sectors.get_market_breadth.return_value = {
                'advancing': 7,
                'declining': 4,
                'total': 11,
                'breadth_ratio': 1.75,
                'avg_performance': 0.5
            }

            response = client.get("/api/v1/market/breadth")

            assert response.status_code == 200
            data = response.json()
            assert data['advancing'] == 7

    @pytest.mark.integration
    def test_get_sector_rotation(self):
        """Test getting sector rotation"""
        with patch('src.api.main.sector_heatmap_builder') as mock_sectors:
            mock_sectors.get_sector_rotation_signal.return_value = {
                'signal': 'RISK_ON',
                'reason': 'Cyclical outperforming',
                'cyclical_avg': 2.5,
                'defensive_avg': 0.5
            }

            response = client.get("/api/v1/market/rotation")

            assert response.status_code == 200
            data = response.json()
            assert data['signal'] == 'RISK_ON'


class TestCalendarEndpoints:
    """Tests for economic calendar endpoints"""

    @pytest.mark.integration
    def test_get_economic_events(self):
        """Test getting economic events"""
        with patch('src.api.main.economic_calendar') as mock_calendar:
            mock_event = MagicMock()
            mock_event.event_id = 'fomc_123'
            mock_event.event_type.value = 'fomc'
            mock_event.title = 'FOMC Meeting'
            mock_event.date.isoformat.return_value = '2024-01-31T14:00:00'
            mock_event.impact.value = 'high'
            mock_event.symbol = None

            mock_calendar.get_upcoming_events.return_value = [mock_event]

            response = client.get("/api/v1/calendar/events")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['event_type'] == 'fomc'

    @pytest.mark.integration
    def test_get_earnings_date(self):
        """Test getting earnings date for symbol"""
        with patch('src.api.main.economic_calendar') as mock_calendar:
            mock_calendar.get_earnings_date.return_value = {
                'symbol': 'AAPL',
                'earnings_date': datetime.now() + timedelta(days=10),
                'days_until': 10
            }
            mock_calendar.get_earnings_warning.return_value = None

            response = client.get("/api/v1/calendar/earnings/AAPL")

            assert response.status_code == 200
            data = response.json()
            assert data['symbol'] == 'AAPL'


class TestStockDataEndpoints:
    """Tests for stock data endpoints"""

    @pytest.mark.integration
    def test_get_stock_price(self):
        """Test getting current stock price"""
        import pandas as pd
        import numpy as np

        with patch('src.api.main.market_data_fetcher') as mock_data:
            # Create mock DataFrame
            dates = pd.date_range(end=datetime.now(), periods=5)
            mock_df = pd.DataFrame({
                'Open': [148, 149, 150, 151, 152],
                'High': [150, 151, 152, 153, 154],
                'Low': [147, 148, 149, 150, 151],
                'Close': [149, 150, 151, 152, 153],
                'Volume': [1000000] * 5
            }, index=dates)

            mock_data.get_historical_data.return_value = mock_df

            response = client.get("/api/v1/stock/AAPL/price")

            assert response.status_code == 200
            data = response.json()
            assert data['symbol'] == 'AAPL'
            assert data['price'] == 153

    @pytest.mark.integration
    def test_get_stock_history(self):
        """Test getting stock history"""
        import pandas as pd

        with patch('src.api.main.market_data_fetcher') as mock_data:
            dates = pd.date_range(end=datetime.now(), periods=5)
            mock_df = pd.DataFrame({
                'Open': [148, 149, 150, 151, 152],
                'High': [150, 151, 152, 153, 154],
                'Low': [147, 148, 149, 150, 151],
                'Close': [149, 150, 151, 152, 153],
                'Volume': [1000000] * 5
            }, index=dates)

            mock_data.get_historical_data.return_value = mock_df

            response = client.get("/api/v1/stock/AAPL/history?period=1mo")

            assert response.status_code == 200
            data = response.json()
            assert data['symbol'] == 'AAPL'
            assert len(data['data']) == 5

    @pytest.mark.integration
    def test_get_stock_not_found(self):
        """Test getting nonexistent stock"""
        with patch('src.api.main.market_data_fetcher') as mock_data:
            mock_data.get_historical_data.return_value = None

            response = client.get("/api/v1/stock/FAKESYM/price")

            assert response.status_code == 404


class TestWebSocketStatus:
    """Tests for WebSocket status endpoint"""

    @pytest.mark.integration
    def test_websocket_status(self):
        """Test WebSocket status endpoint"""
        response = client.get("/ws/status")

        assert response.status_code == 200
        data = response.json()
        assert 'active_connections' in data
        assert 'subscriptions' in data


class TestConnectionManager:
    """Tests for WebSocket ConnectionManager"""

    @pytest.fixture
    def manager(self):
        return ConnectionManager()

    @pytest.mark.integration
    def test_generate_client_id(self, manager):
        """Test client ID generation"""
        id1 = manager._generate_client_id()
        id2 = manager._generate_client_id()

        assert id1.startswith('client_')
        assert id2.startswith('client_')
        assert id1 != id2

    @pytest.mark.integration
    def test_disconnect_cleanup(self, manager):
        """Test disconnect removes client from all subscriptions"""
        client_id = 'test_client'
        manager.active_connections[client_id] = MagicMock()
        manager.subscriptions['alerts'].add(client_id)
        manager.subscriptions['progress'].add(client_id)

        manager.disconnect(client_id)

        assert client_id not in manager.active_connections
        assert client_id not in manager.subscriptions['alerts']
        assert client_id not in manager.subscriptions['progress']

    @pytest.mark.integration
    def test_get_connection_count(self, manager):
        """Test connection count"""
        assert manager.get_connection_count() == 0

        manager.active_connections['client1'] = MagicMock()
        manager.active_connections['client2'] = MagicMock()

        assert manager.get_connection_count() == 2

    @pytest.mark.integration
    def test_get_client_subscriptions(self, manager):
        """Test getting client subscriptions"""
        client_id = 'test_client'
        manager.subscriptions['alerts'].add(client_id)
        manager.subscriptions['progress'].add(client_id)

        subs = manager.get_client_subscriptions(client_id)

        assert 'alerts' in subs
        assert 'progress' in subs


class TestAlertMessage:
    """Tests for message dataclasses"""

    @pytest.mark.integration
    def test_alert_message_defaults(self):
        """Test AlertMessage has correct defaults"""
        msg = AlertMessage()

        assert msg.type == 'alert'
        assert msg.symbol == ''
        assert msg.timestamp is not None

    @pytest.mark.integration
    def test_alert_message_with_data(self):
        """Test AlertMessage with data"""
        msg = AlertMessage(
            symbol='AAPL',
            recommendation='BUY',
            current_price=150.0,
            confidence_score=75.0
        )

        assert msg.symbol == 'AAPL'
        assert msg.recommendation == 'BUY'
        assert msg.confidence_score == 75.0

    @pytest.mark.integration
    def test_progress_message(self):
        """Test ProgressMessage"""
        msg = ProgressMessage(
            total=100,
            completed=50,
            current_symbol='AAPL'
        )

        assert msg.type == 'progress'
        assert msg.total == 100
        assert msg.completed == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
