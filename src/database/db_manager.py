"""
Database manager for storing screening results and alerts
"""
import os
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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
        logger.info(f"Database initialized at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    def save_alert(self, alert_data: Dict) -> StockAlert:
        """
        Save a new stock alert

        Args:
            alert_data: Dictionary with alert information

        Returns:
            Created StockAlert object
        """
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
            session.close()

    def save_screening_history(self, history_data: Dict) -> ScreeningHistory:
        """
        Save screening execution history

        Args:
            history_data: Dictionary with screening history information

        Returns:
            Created ScreeningHistory object
        """
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
            session.close()

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
            session.close()

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
            session.close()

    def mark_alert_as_notified(self, alert_id: int):
        """
        Mark an alert as notified

        Args:
            alert_id: ID of the alert to mark
        """
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
            session.close()

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
            session.close()


# Singleton instance
db_manager = DatabaseManager()
