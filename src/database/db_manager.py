"""
Database manager for storing screening results and alerts
"""
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from config.settings import DATABASE_PATH
from src.utils.logger import logger

Base = declarative_base()


class StockAlert(Base):
    """Model for stock buy alerts"""
    __tablename__ = 'stock_alerts'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    company_name = Column(String(200))
    alert_date = Column(DateTime, default=datetime.utcnow, index=True)
    timeframe = Column(String(10), nullable=False)  # 'weekly' or 'daily'
    current_price = Column(Float, nullable=False)
    support_level = Column(Float, nullable=False)
    distance_to_support_pct = Column(Float, nullable=False)
    ema_24 = Column(Float)
    ema_38 = Column(Float)
    ema_62 = Column(Float)
    ema_alignment = Column(String(100))  # e.g., "24>38>62"
    crossover_info = Column(Text)  # JSON string with crossover details
    recommendation = Column(String(50))  # 'BUY', 'WATCH', etc.
    is_notified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ScreeningHistory(Base):
    """Model for keeping screening history"""
    __tablename__ = 'screening_history'

    id = Column(Integer, primary_key=True)
    screening_date = Column(DateTime, default=datetime.utcnow, index=True)
    total_stocks_analyzed = Column(Integer)
    total_alerts_generated = Column(Integer)
    timeframe = Column(String(10))
    execution_time_seconds = Column(Float)
    status = Column(String(20))  # 'SUCCESS', 'FAILED', 'PARTIAL'
    error_message = Column(Text, nullable=True)


class MarketCapCache(Base):
    """Cache for market cap data to avoid repeated API calls"""
    __tablename__ = 'market_cap_cache'

    symbol = Column(String(20), primary_key=True)
    market_cap = Column(Float)
    sector = Column(String(100))
    industry = Column(String(100))
    company_name = Column(String(200))
    updated = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manages database operations"""

    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        # Create data directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        # Thread-safe scoped session for concurrent access
        self._ScopedSession = scoped_session(self.SessionLocal)
        # Write lock for thread-safe modifications
        self._write_lock = threading.Lock()
        logger.info(f"Database initialized at {db_path} (thread-safe mode)")

    def get_session(self) -> Session:
        """Get a thread-local database session (thread-safe)"""
        return self._ScopedSession()
    
    def remove_session(self):
        """Remove the current thread-local session (call after operations)"""
        self._ScopedSession.remove()

    def save_alert(self, alert_data: Dict) -> StockAlert:
        """
        Save a new stock alert (thread-safe)

        Args:
            alert_data: Dictionary with alert information

        Returns:
            Created StockAlert object
        """
        with self._write_lock:
            session = self.get_session()
            try:
                alert = StockAlert(**alert_data)
                session.add(alert)
                session.commit()
                session.refresh(alert)
                logger.info(f"Alert saved for {alert_data.get('symbol')} on {alert_data.get('timeframe')}")
                return alert
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving alert: {e}")
                raise
            finally:
                self.remove_session()

    def save_screening_history(self, history_data: Dict) -> ScreeningHistory:
        """
        Save screening execution history (thread-safe)

        Args:
            history_data: Dictionary with screening history information

        Returns:
            Created ScreeningHistory object
        """
        with self._write_lock:
            session = self.get_session()
            try:
                history = ScreeningHistory(**history_data)
                session.add(history)
                session.commit()
                session.refresh(history)
                logger.info(f"Screening history saved: {history_data.get('total_alerts_generated')} alerts")
                return history
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving screening history: {e}")
                raise
            finally:
                self.remove_session()

    def get_recent_alerts(self, days: int = 7, timeframe: Optional[str] = None) -> List[StockAlert]:
        """
        Get recent alerts

        Args:
            days: Number of days to look back
            timeframe: Optional filter by timeframe ('weekly' or 'daily')

        Returns:
            List of StockAlert objects
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            query = session.query(StockAlert).filter(StockAlert.alert_date >= cutoff_date)

            if timeframe:
                query = query.filter(StockAlert.timeframe == timeframe)

            alerts = query.order_by(StockAlert.alert_date.desc()).all()
            return alerts
        finally:
            self.remove_session()

    def get_unnotified_alerts(self) -> List[StockAlert]:
        """
        Get alerts that haven't been notified yet

        Returns:
            List of unnotified StockAlert objects
        """
        session = self.get_session()
        try:
            alerts = session.query(StockAlert).filter(
                StockAlert.is_notified == False
            ).order_by(StockAlert.alert_date.desc()).all()
            return alerts
        finally:
            self.remove_session()

    def mark_alert_as_notified(self, alert_id: int):
        """
        Mark an alert as notified

        Args:
            alert_id: ID of the alert to mark
        """
        with self._write_lock:
            session = self.get_session()
        try:
            alert = session.query(StockAlert).filter(StockAlert.id == alert_id).first()
            if alert:
                alert.is_notified = True
                session.commit()
                logger.debug(f"Alert {alert_id} marked as notified")
        except Exception as e:
            session.rollback()
            logger.error(f"Error marking alert as notified: {e}")
            raise
        finally:
            self.remove_session()

    def get_alerts_by_symbol(self, symbol: str, days: int = 30) -> List[StockAlert]:
        """
        Get all alerts for a specific symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of StockAlert objects
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            alerts = session.query(StockAlert).filter(
                StockAlert.symbol == symbol,
                StockAlert.alert_date >= cutoff_date
            ).order_by(StockAlert.alert_date.desc()).all()
            return alerts
        finally:
            self.remove_session()

    def save_alerts_batch(self, alerts_data: List[Dict]) -> int:
        """
        Save multiple alerts in a single transaction (100x faster than individual saves)

        Args:
            alerts_data: List of dictionaries with alert information

        Returns:
            Number of alerts successfully saved
        """
        if not alerts_data:
            return 0

        with self._write_lock:
            session = self.get_session()
        try:
            # Use bulk_insert_mappings for optimal performance
            session.bulk_insert_mappings(StockAlert, alerts_data)
            session.commit()
            logger.info(f"Batch saved {len(alerts_data)} alerts in single transaction")
            return len(alerts_data)
        except Exception as e:
            session.rollback()
            logger.error(f"Error in batch save: {e}")
            # Fallback to individual saves to identify problematic records
            saved_count = 0
            for alert_data in alerts_data:
                try:
                    alert = StockAlert(**alert_data)
                    session.add(alert)
                    session.commit()
                    saved_count += 1
                except Exception as individual_error:
                    session.rollback()
                    logger.warning(f"Failed to save alert for {alert_data.get('symbol')}: {individual_error}")
            logger.info(f"Fallback: saved {saved_count}/{len(alerts_data)} alerts individually")
            return saved_count
        finally:
            self.remove_session()

    def delete_old_alerts(self, days_to_keep: int = 90) -> int:
        """
        Delete alerts older than specified days

        Args:
            days_to_keep: Number of days to keep alerts

        Returns:
            Number of alerts deleted
        """
        with self._write_lock:
            session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            deleted = session.query(StockAlert).filter(
                StockAlert.alert_date < cutoff_date
            ).delete()

            session.commit()
            logger.info(f"Deleted {deleted} alerts older than {days_to_keep} days")
            return deleted
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting old alerts: {e}")
            return 0
        finally:
            self.remove_session()

    # ==================== Market Cap Cache ====================

    def get_cached_market_cap(self, symbol: str, max_age_days: int = 7) -> Optional[Dict]:
        """
        Get cached market cap data if fresh enough

        Args:
            symbol: Stock symbol
            max_age_days: Maximum age of cache in days

        Returns:
            Dict with market_cap, sector, industry, company_name or None if not cached/stale
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

            cached = session.query(MarketCapCache).filter(
                MarketCapCache.symbol == symbol,
                MarketCapCache.updated >= cutoff_date
            ).first()

            if cached:
                return {
                    'market_cap': cached.market_cap,
                    'sector': cached.sector,
                    'industry': cached.industry,
                    'company_name': cached.company_name,
                    'cached': True
                }
            return None
        except Exception as e:
            logger.error(f"Error getting cached market cap for {symbol}: {e}")
            return None
        finally:
            self.remove_session()

    def save_market_cap(self, symbol: str, market_cap: float, sector: str = None,
                        industry: str = None, company_name: str = None):
        """
        Save market cap data to cache

        Args:
            symbol: Stock symbol
            market_cap: Market cap in actual value (not millions)
            sector: Sector name
            industry: Industry name
            company_name: Company name
        """
        with self._write_lock:
            session = self.get_session()
        try:
            # Upsert - update if exists, insert if not
            existing = session.query(MarketCapCache).filter(
                MarketCapCache.symbol == symbol
            ).first()

            if existing:
                existing.market_cap = market_cap
                existing.sector = sector
                existing.industry = industry
                existing.company_name = company_name
                existing.updated = datetime.utcnow()
            else:
                new_cache = MarketCapCache(
                    symbol=symbol,
                    market_cap=market_cap,
                    sector=sector,
                    industry=industry,
                    company_name=company_name
                )
                session.add(new_cache)

            session.commit()
            logger.debug(f"Cached market cap for {symbol}: ${market_cap/1e9:.2f}B")
        except Exception as e:
            session.rollback()
            logger.error(f"Error caching market cap for {symbol}: {e}")
        finally:
            self.remove_session()

    def get_cached_market_caps_batch(self, symbols: List[str], max_age_days: int = 7) -> Dict[str, Dict]:
        """
        Get cached market cap data for multiple symbols at once

        Args:
            symbols: List of stock symbols
            max_age_days: Maximum age of cache in days

        Returns:
            Dict mapping symbols to their cached data
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

            cached_entries = session.query(MarketCapCache).filter(
                MarketCapCache.symbol.in_(symbols),
                MarketCapCache.updated >= cutoff_date
            ).all()

            result = {}
            for entry in cached_entries:
                result[entry.symbol] = {
                    'market_cap': entry.market_cap,
                    'sector': entry.sector,
                    'industry': entry.industry,
                    'company_name': entry.company_name,
                    'cached': True
                }
            return result
        except Exception as e:
            logger.error(f"Error getting batch cached market caps: {e}")
            return {}
        finally:
            self.remove_session()

    def save_market_caps_batch(self, data: List[Dict]):
        """
        Save multiple market cap entries at once

        Args:
            data: List of dicts with symbol, market_cap, sector, industry, company_name
        """
        if not data:
            return

        with self._write_lock:
            session = self.get_session()
        try:
            for item in data:
                existing = session.query(MarketCapCache).filter(
                    MarketCapCache.symbol == item['symbol']
                ).first()

                if existing:
                    existing.market_cap = item.get('market_cap', 0)
                    existing.sector = item.get('sector')
                    existing.industry = item.get('industry')
                    existing.company_name = item.get('company_name')
                    existing.updated = datetime.utcnow()
                else:
                    new_cache = MarketCapCache(
                        symbol=item['symbol'],
                        market_cap=item.get('market_cap', 0),
                        sector=item.get('sector'),
                        industry=item.get('industry'),
                        company_name=item.get('company_name')
                    )
                    session.add(new_cache)

            session.commit()
            logger.info(f"Batch cached {len(data)} market cap entries")
        except Exception as e:
            session.rollback()
            logger.error(f"Error batch caching market caps: {e}")
        finally:
            self.remove_session()


# Singleton instance
db_manager = DatabaseManager()
