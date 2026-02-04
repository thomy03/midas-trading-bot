from src.execution.paper_trader import get_paper_trader

pt = get_paper_trader()

signals = [
    ("EVRG", 76.25, 86.3),
    ("ENTG", 120.73, 74.1),
    ("COKE", 154.55, 69.0),
]

for symbol, price, score in signals:
    print(f"Opening {symbol}...")
    position = pt.open_position(symbol, price, score, "BUY")
    if position:
        print(f"  OK: {symbol} x{position.quantity} @ {position.entry_price:.2f}")
        print(f"  SL: {position.stop_loss:.2f}, TP: {position.take_profit:.2f}")
    else:
        print(f"  SKIP: {symbol}")

summary = pt.get_portfolio_summary()
print(f"\nTotal: {summary['total_value']:.2f}, Cash: {summary['cash']:.2f}, Positions: {summary['positions_count']}")
