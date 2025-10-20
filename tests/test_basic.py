"""
Basic tests for the market screener
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.market_data import MarketDataFetcher
from src.indicators.ema_analyzer import EMAAnalyzer
from src.utils.logger import logger


def test_market_data_fetcher():
    """Test market data fetching"""
    logger.info("Testing market data fetcher...")

    fetcher = MarketDataFetcher()

    # Test getting S&P 500 tickers
    tickers = fetcher.get_sp500_tickers()
    assert len(tickers) > 0, "Should fetch S&P 500 tickers"
    logger.info(f"✓ Fetched {len(tickers)} S&P 500 tickers")

    # Test getting historical data for a known stock
    df = fetcher.get_historical_data('AAPL', period='1y', interval='1d')
    assert df is not None and not df.empty, "Should fetch historical data"
    logger.info(f"✓ Fetched {len(df)} days of AAPL data")

    # Test getting stock info
    info = fetcher.get_stock_info('AAPL')
    assert info is not None, "Should fetch stock info"
    logger.info(f"✓ Fetched AAPL info: {info.get('longName', 'N/A')}")

    logger.info("Market data fetcher tests passed! ✓\n")


def test_ema_analyzer():
    """Test EMA analyzer"""
    logger.info("Testing EMA analyzer...")

    fetcher = MarketDataFetcher()
    analyzer = EMAAnalyzer()

    # Get test data
    df = fetcher.get_historical_data('AAPL', period='1y', interval='1d')
    assert df is not None and not df.empty, "Need data for testing"

    # Test EMA calculation
    df_with_emas = analyzer.calculate_emas(df)
    assert 'EMA_24' in df_with_emas.columns, "Should calculate EMA 24"
    assert 'EMA_38' in df_with_emas.columns, "Should calculate EMA 38"
    assert 'EMA_62' in df_with_emas.columns, "Should calculate EMA 62"
    logger.info("✓ EMAs calculated successfully")

    # Test EMA alignment check
    is_aligned, alignment_desc = analyzer.check_ema_alignment(df_with_emas, for_buy=True)
    logger.info(f"✓ EMA alignment: {alignment_desc} (Aligned: {is_aligned})")

    # Test crossover detection
    crossovers = analyzer.detect_crossovers(df_with_emas, timeframe='daily')
    logger.info(f"✓ Detected {len(crossovers)} crossovers")

    if crossovers:
        logger.info(f"  Most recent crossover: {crossovers[0]['days_ago']} days ago")

    logger.info("EMA analyzer tests passed! ✓\n")


def test_filtering():
    """Test stock filtering"""
    logger.info("Testing stock filtering...")

    fetcher = MarketDataFetcher()

    # Test filtering a known stock
    passes, info = fetcher.filter_by_criteria('AAPL')

    if info:
        market_cap = info.get('marketCap', 0) / 1_000_000
        logger.info(f"AAPL Market Cap: ${market_cap:.1f}M")
        logger.info(f"AAPL passes filter: {passes}")
    else:
        logger.warning("Could not fetch AAPL info")

    logger.info("Stock filtering tests passed! ✓\n")


def run_all_tests():
    """Run all tests"""
    logger.info("="*60)
    logger.info("RUNNING BASIC TESTS")
    logger.info("="*60 + "\n")

    try:
        test_market_data_fetcher()
        test_ema_analyzer()
        test_filtering()

        logger.info("="*60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("="*60)

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
