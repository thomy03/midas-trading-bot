#!/usr/bin/env python
"""Test dual-timeframe simulation"""

from src.backtesting import PortfolioSimulator, BacktestConfig
from src.data.market_data import market_data_fetcher

# Config avec dual-timeframe
config = BacktestConfig(
    initial_capital=10000,
    max_positions=3,
    use_daily_fallback=True,
    min_ema_conditions_for_fallback=2,
    min_confidence_score=30,  # Plus permissif
    require_volume_confirmation=False  # Pas de volume requis
)

print('Configuration:')
print(f'  use_daily_fallback: {config.use_daily_fallback}')
print(f'  min_ema_conditions_for_fallback: {config.min_ema_conditions_for_fallback}')

# Test avec plus de symboles (US + Europe)
symbols = [
    # US Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # CAC40
    'MC.PA', 'OR.PA', 'AI.PA', 'SAN.PA', 'BNP.PA'
]
print(f'\nTelechargement donnees pour {len(symbols)} symboles...')

# Charger weekly (5 ans pour avoir assez de lookback)
print('  Weekly...')
data_weekly = market_data_fetcher.get_batch_historical_data(
    symbols, period='5y', interval='1wk', batch_size=10
)
print(f'  -> {len(data_weekly)} symboles weekly')

# Charger daily
print('  Daily...')
data_daily = market_data_fetcher.get_batch_historical_data(
    symbols, period='5y', interval='1d', batch_size=10
)
print(f'  -> {len(data_daily)} symboles daily')

# Lancer simulation
print('\nSimulation 2 ans (2023-2024)...')
simulator = PortfolioSimulator(config)

result = simulator.run_simulation(
    all_data=data_weekly,
    start_date='2023-01-01',  # Plus longue periode
    end_date='2024-12-01',
    all_data_daily=data_daily
)

print('\n=== RESULTATS ===')
print(f'Signaux detectes: {result.signals_detected}')
print(f'Signaux pris:     {result.signals_taken}')
print(f'Trades fermes:    {len(result.trades)}')
print(f'Capital final:    ${result.metrics.final_capital:,.2f}')
print(f'Return:           {result.metrics.total_return:+.2f}%')
print(f'Win rate:         {result.metrics.win_rate:.1f}%')

# Verifier timeframes des trades
if result.trades:
    print('\nTrades (5 premiers):')
    for t in result.trades[:5]:
        print(f'  {t.symbol}: {t.entry_date.strftime("%Y-%m-%d")} -> {t.exit_date.strftime("%Y-%m-%d")} ({t.exit_reason})')

print('\nTest dual-timeframe OK!')
